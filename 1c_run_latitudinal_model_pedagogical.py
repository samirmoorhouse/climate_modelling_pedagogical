import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time
from numba import jit
import datetime
import os
import pandas as pd

start_time = time.time()

### --- 1. FUNDAMENTAL CONSTANTS --- ###
sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
S0 = 1361              # Solar constant (W/m^2)
Q = S0 / 4             # Average solar insolation (W/m^2)
g = 9.81               # Gravity (m/s^2)
Na = 6.02214076e23     # Avogadro's number (mol^-1)
m_air = 28.97e-3       # Molar mass of dry air (kg/mol)
k_B = 1.380649e-23     # Boltzmann constant (J/K)
c = 2.99792458e8       # Speed of light (m/s)
h = 6.62607015e-34     # Planck constant (J*s)

R_dry = 287.05         # Specific gas constant for dry air (J/kg/K)
L_v = 2.501e6          # Latent heat of vaporization (J/kg)
R_vap = 461.5          # Specific gas constant for water vapor (J/kg/K)
cp_dry_air = 1004      # Specific heat of dry air (J/kg/K)
cp_h2o = 4186.0        # Specific heat of liquid water (J/kg/K)
cp_h2o_vap = 1860      # Specific heat of water vapor (J/kg/K)

T_sun = 5778.0              # Sun surface temperature (K)
sun_radius = 695700000      # Radius of the Sun (m)
earth_sun_distance = 149597870700  # Mean Earth-Sun distance (m)


### --- 2. SIMULATION PARAMETERS & TOGGLES --- ###

# -- Time & Grid Settings --
start_year, start_month, start_day = 2010, 1, 1
years = 2
dt_days = 0.3  # Time step (days); keep small for numerical stability across all latitudes
N = 25         # Number of vertical atmospheric layers per latitude column

# -- Atmospheric Composition --
initial_co2_ppm = 0      # Initial CO2 concentration (ppm)
co2_ppm_rate = 0         # Annual CO2 increase (ppm/year)
initial_ch4_ppm = 1.8    # Initial CH4 concentration (ppm)
ch4_ppm_rate = 0         # Annual CH4 increase (ppm/year)
ozone_peak_ppm = 7       # Peak stratospheric ozone concentration (ppm)
relative_humidity = 0.77 # Baseline surface relative humidity (fraction, 0–1)

# -- 2D (Latitudinal) Physics Toggles --
diffusion = True           # Enable meridional (North-South) heat transport
include_topography = True  # Modify surface pressure from CSV elevation data
heat_capacity_factor = 10  # Divisor applied to CSV heat capacities; increase for faster spin-up
D_coeff = 0.6              # Diffusion coefficient for meridional heat transport (W/m^2/K)

# -- General Physics Toggles --
convection = True          # Enable convective adjustment
log_pressure = True        # True = logarithmic layer spacing; False = linear spacing
include_CIA = True         # Include Collision-Induced Absorption
seasonal_variation = True  # Account for Earth's axial tilt and orbital eccentricity
include_albedo = True      # Enable surface albedo and Rayleigh SW reflection
include_clouds = True      # Enable cloud shortwave reflection
include_blanket = True     # Enable cloud longwave trapping

# -- Experimental Forcing Toggles --
perturb = False                   # If True, doubles CO2 halfway through the run
fixed_water_feedback = False      # Lock water vapour to pre-perturbation climatology
dynamic_ice_albedo = False        # Allow albedo to vary with surface temperature (ice feedback)
lock_surface_after_switch = False # Fix surface temperature after the CO2 perturbation

diffusivity_factor = 1.66  # Diffusivity factor for LW angular integration


### --- 3. SEASONAL & TOPOGRAPHIC DATA LOADING --- ###
print("Loading seasonal and topographic data...")
df_params = pd.read_csv('parameters/latitude_params_monthly.csv')
df_params = df_params[df_params['Latitude'] != 'Global']
df_params['Latitude'] = df_params['Latitude'].astype(float)
unique_lats = np.sort(df_params['Latitude'].unique())
num_lats = len(unique_lats)

# Cloud/albedo arrays are stored as monthly time series and interpolated each time step.
# The arrays are padded with wrap-around values at day -15 and day 380 to avoid edge
# artefacts when interpolating near the start and end of the year.
month_days = np.array([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345])
days_padded = np.concatenate([[-15], month_days, [380]])


def get_data_grid(variable_name):
    """Pulls monthly parameter data into a (14 × num_lats) grid for time interpolation.

    Row 0 is a copy of December (month 12) and row 13 is a copy of January (month 1),
    providing the same wrap-around padding as days_padded.
    """
    grid = np.zeros((14, num_lats))
    for i, m in enumerate(range(1, 13)):
        m_data = df_params[df_params['Month'] == m].sort_values('Latitude')
        grid[i + 1, :] = m_data[variable_name].values
    grid[0, :], grid[13, :] = grid[12, :], grid[1, :]
    return grid


albedo_grid    = get_data_grid('Albedo')
cld_albedo_grid = get_data_grid('Cloud_Albedo')
cld_frac_grid  = get_data_grid('Cloud_Fraction')
cld_press_grid = get_data_grid('Cloud_Pressure')

static_data  = df_params[df_params['Month'] == 1].sort_values('Latitude')
Cs_arr       = static_data['Heat_Capacity'].values / heat_capacity_factor  # J/m^2/K
rh_arr       = static_data['Relative_Humidity'].values                     # fraction
ts_init_arr  = static_data['Initial_Ts'].values                            # K
lats_grid    = static_data['Latitude'].values                              # degrees N

elev_arr = static_data['Elevation'].values if include_topography else np.zeros(num_lats)
if include_topography:
    print("Topography enabled: surface pressures will vary by latitude.")


### --- 4. ATMOSPHERIC GRID SETUP (2D) --- ###
p_top        = 50       # Top-of-atmosphere pressure (Pa)
p_tropopause = 20000.0  # Approximate tropopause pressure (Pa)
p_stratopause = 100     # Approximate stratopause pressure (Pa)

# Barometric formula: surface pressure adjusted for local elevation
p_surface_arr = 101325.0 * np.exp(-g * elev_arr / (R_dry * 288.15))

# Standard-atmosphere column (used only for the initial temperature profile)
p_int_std = np.geomspace(p_top, 101325.0, N) if log_pressure else np.linspace(p_top, 101325.0, N)
p_mid_std = 0.5 * (p_int_std[:-1] + p_int_std[1:])

# Build per-latitude pressure grids
p_int = np.zeros((num_lats, N))
for i in range(num_lats):
    p_int[i, :] = (np.geomspace(p_top, p_surface_arr[i], N) if log_pressure
                   else np.linspace(p_top, p_surface_arr[i], N))

p_mid  = 0.5 * (p_int[:, :-1] + p_int[:, 1:])  # Layer midpoint pressures (Pa), shape (num_lats, N-1)
dp     = np.diff(p_int, axis=1)                   # Layer pressure thicknesses (Pa), shape (num_lats, N-1)
C_air  = cp_dry_air * dp / g                      # Layer heat capacities (J/m^2/K)

print(f"Equator surface pressure:    {p_surface_arr[num_lats // 2] / 100:.1f} hPa")
print(f"South Pole surface pressure: {p_surface_arr[0] / 100:.1f} hPa")

# Initial temperature profile following the standard atmospheric structure
n_meso  = np.sum(p_mid_std < p_stratopause)
n_strat = np.sum((p_mid_std >= p_stratopause) & (p_mid_std < p_tropopause))
n_trop  = N - n_meso - n_strat - 1  # N-1 midpoints total; -1 accounts for the boundary

T_initial = np.concatenate((
    np.linspace(220, 270, n_meso),
    np.linspace(270, 215, n_strat),
    np.linspace(215, 290, n_trop)
))


### --- 5. DATA LOADING & INTERPOLATION (HITRAN) --- ###
print("Loading pre-computed K tables (HITRAN)...")
ktable_co2    = np.load("data/hitran_tables/k_table_co2_lw_new_gauss_8.npz",    allow_pickle=True)['k_table'].item()
ktable_h2o    = np.load("data/hitran_tables/k_table_h2o_lw_gauss_8.npz",        allow_pickle=True)['k_table'].item()
ktable_h2o_sw = np.load("data/hitran_tables/k_table_h2o_sw_new_gauss_8.npz",    allow_pickle=True)['k_table'].item()
ktable_ch4    = np.load("data/hitran_tables/k_table_ch4_new_gauss_8.npz",        allow_pickle=True)['k_table'].item()
ktable_o3     = np.load("data/hitran_tables/k_table_o3_lw_new_gauss_8.npz",     allow_pickle=True)['k_table'].item()
ktable_o3_sw  = np.load("data/hitran_tables/k_table_o3_sw_new_gauss_8_ex.npz",  allow_pickle=True)['k_table'].item()

cia_data   = np.load("data/hitran_tables/cia_table_combined.npz", allow_pickle=True)
cia_t_grid = cia_data['T_grid']
cia_lw, cia_sw = cia_data['cia_lw'].item(), cia_data['cia_sw'].item()

# Build interpolators so the model can look up absorption coefficients at any (P, T, ppm).
co2_interps = [
    RegularGridInterpolator((np.log10(ktable_co2['P_grid']), ktable_co2['T_grid']),
                            band['k_P_T_g'], bounds_error=False)
    for band in ktable_co2['bands']]

h2o_interps = [
    RegularGridInterpolator((np.log10(ktable_h2o['P_grid']), ktable_h2o['T_grid'],
                             np.log10(ktable_h2o['ppm_grid'])),
                            band['k_P_T_Q_g'], bounds_error=False)
    for band in ktable_h2o['bands']]

h2o_sw_interps = [
    RegularGridInterpolator((np.log10(ktable_h2o_sw['P_grid']), ktable_h2o_sw['T_grid'],
                             np.log10(ktable_h2o_sw['ppm_grid'])),
                            band['k_P_T_Q_g'], bounds_error=False)
    for band in ktable_h2o_sw['bands']]

ch4_interps = [
    RegularGridInterpolator((np.log10(ktable_ch4['P_grid']), ktable_ch4['T_grid']),
                            band['k_P_T_g'], bounds_error=False)
    for band in ktable_ch4['bands']]

o3_interps = [
    RegularGridInterpolator((np.log10(ktable_o3['P_grid']), ktable_o3['T_grid']),
                            band['k_P_T_g'], bounds_error=False)
    for band in ktable_o3['bands']]

o3_sw_interps = [
    RegularGridInterpolator((ktable_o3_sw['T_grid'],), band['k_T_g'], bounds_error=False)
    for band in ktable_o3_sw['bands']]

print("Tables interpolated successfully.")


### --- 6. GAS DISTRIBUTIONS --- ###
def get_h2o_ppm(T, P_Pa, local_rh):
    """
    Returns water vapour concentration (ppm) at each model layer.

    Troposphere: relative humidity scaled by the Magnus-Tetens saturation vapour pressure,
    with RH decreasing to zero above the tropopause.
    Stratosphere: fixed background set by CH4 oxidation. In the real stratosphere, methane
    is oxidised to produce water vapour; each mole of CH4 destroyed yields ~2 moles of H2O.
    This is approximated here with a fixed CH4 decay profile.
    The two regimes are blended smoothly across the tropopause.
    """
    T_c = T - 273.15
    # Magnus-Tetens formula: saturation vapour pressure (Pa)
    exp_term = (18.678 - T_c / 234.5) * (T_c / (257.14 + T_c))
    p_sat = 0.61121 * np.exp(exp_term) * 1000.0

    p0 = np.max(P_Pa)
    rh = local_rh * np.clip((P_Pa / p0 - 0.02) / (1.0 - 0.02), 0.0, 1.0)
    q_local = (rh * p_sat / P_Pa) * 1e6  # Tropospheric water vapour (ppm)
    q_entry = 3.5                         # Water vapour at tropopause entry (ppm)

    # Stratospheric H2O from CH4 oxidation: H2O_strat ≈ q_entry + 2*(CH4_surface - CH4(z))
    ch4_surface, ch4_top = 1.8, 0.3
    log_p = np.log(np.clip(P_Pa, 1.0, None))
    ch4_weight  = 1.0 / (1.0 + np.exp(-(log_p - np.log(3000.0)) / 0.75))
    current_ch4 = ch4_top + (ch4_surface - ch4_top) * ch4_weight
    q = q_entry + 2.0 * (ch4_surface - current_ch4)

    # Smooth blend: w_trop=1 in troposphere, w_trop=0 in stratosphere
    w_trop = 1.0 / (1.0 + np.exp((log_p - np.log(20000.0)) / 0.2))

    return np.zeros_like(T) if local_rh == 0 else np.maximum(w_trop * q + (1.0 - w_trop) * q_local, q)


def get_ozone_ppm(P_Pa):
    """Returns ozone concentration (ppm) as a Gaussian peak in the stratosphere."""
    base_ozone   = 0.04       # Tropospheric background (ppm)
    peak_pressure = 8 * 100   # Peak ozone pressure (~8 hPa, mid-stratosphere)
    width = np.where(P_Pa < peak_pressure, 2.5, 2)
    strat_ozone = ozone_peak_ppm * np.exp(
        -np.abs((np.log(P_Pa) - np.log(peak_pressure)) / width) ** 2)
    return base_ozone + strat_ozone


def get_ch4_ppm(P_Pa, current_surface_ppm):
    """Returns methane concentration (ppm) decaying from the surface value to a stratospheric background."""
    strat_background = 0.2  # Stratospheric CH4 background (ppm)
    weight = 1.0 / (1.0 + np.exp(-(np.log(P_Pa) - np.log(1000.0)) / 1))
    return strat_background + (current_surface_ppm - strat_background) * weight


### --- 7. 2D SPATIAL PHYSICS --- ###
def spherical_grid(lats):
    """
    Returns the geometric widths and boundary properties of latitude bands.

    Uses the sine-latitude coordinate x = sin(lat), in which equal increments of x
    correspond to equal areas on the sphere.  This is the natural coordinate for the
    diffusion operator.

    Returns: x, dx, x_bounds, boundary_lengths
    """
    x = np.sin(np.radians(lats))
    x_bounds = np.zeros(num_lats + 1)
    x_bounds[0], x_bounds[-1] = -1.0, 1.0
    for i in range(1, num_lats):
        x_bounds[i] = 0.5 * (x[i - 1] + x[i])
    return x, np.diff(x_bounds), x_bounds, 1.0 - x_bounds**2


def get_dynamic_albedo(T_surf, baseline_albedo):
    """
    Returns the effective surface albedo, optionally adjusted for ice-albedo feedback.

    When dynamic_ice_albedo is True, the albedo transitions smoothly between an
    ice-covered value (~0.6) and an ice-free value (~0.12) around the freezing point
    (266.575 K), using a tanh function with a transition width of 3 K.
    """
    if not dynamic_ice_albedo:
        return baseline_albedo
    ice_free_alb = baseline_albedo if baseline_albedo < 0.3 else 0.12
    ice_alb = max(0.6, baseline_albedo)
    return 0.5 * (ice_alb + ice_free_alb) - 0.5 * (ice_alb - ice_free_alb) * np.tanh((T_surf - 266.575) / 3.0)


def apply_diffusion(Ts, Cs_arr, x, dx, boundary_lengths, dt, p_surface):
    """
    Applies one step of meridional (North-South) heat diffusion to the surface temperature.

    Solves the implicit diffusion equation:
        Cs * dTs/dt = D * d/dx [(1-x^2) * dTs/dx] * (p_surface/p0)^(R/cp)
    where x = sin(latitude) and the potential-temperature factor (f_theta) accounts for
    varying surface pressure with topography.  The implicit scheme is unconditionally
    stable and avoids the need for a very short time step.
    """
    num_lats = len(Ts)
    M = np.zeros((num_lats, num_lats))
    f_theta = (101325.0 / p_surface) ** (R_dry / cp_dry_air)

    for i in range(num_lats):
        coeff_north = ((D_coeff * boundary_lengths[i + 1] * dt) /
                       (Cs_arr[i] * dx[i] * (x[i + 1] - x[i]))) if i < num_lats - 1 else 0.0
        coeff_south = ((D_coeff * boundary_lengths[i] * dt) /
                       (Cs_arr[i] * dx[i] * (x[i] - x[i - 1]))) if i > 0 else 0.0

        M[i, i]     =  1.0 + coeff_north * f_theta[i] + coeff_south * f_theta[i]
        if i < num_lats - 1: M[i, i + 1] = -coeff_north * f_theta[i + 1]
        if i > 0:            M[i, i - 1] = -coeff_south * f_theta[i - 1]

    return np.linalg.solve(M, Ts)


def cloud_profiles(cld_press, cld_frac, cld_albedo, p_mid_local, dp_local):
    """
    Builds per-layer SW reflectivity and LW optical depth profiles for one latitude column.

    A single cloud layer is placed at the pressure level nearest to cld_press.
    Rayleigh scattering is distributed across all layers proportionally to their
    pressure thickness.
    """
    local_reflectivity = 0.1 * (dp_local / np.sum(dp_local))  # Rayleigh component
    local_tau_cloud    = np.zeros(len(p_mid_local))

    if not include_albedo:
        local_reflectivity[:] = 0

    if include_clouds:
        idx_cloud = (np.abs(p_mid_local - cld_press)).argmin()
        if include_albedo:
            local_reflectivity[idx_cloud] = min(
                local_reflectivity[idx_cloud] + cld_frac * cld_albedo, 0.999)
        if include_blanket:
            # Effective LW transmittance: clear-sky fraction + cloud fraction × Beer-Lambert
            # Fixed LW optical depth of 1.25 for the single cloud layer
            local_tau_cloud[idx_cloud] = -np.log(
                max((1 - cld_frac) + cld_frac * np.exp(-1.25), 1e-10))

    return local_reflectivity, local_tau_cloud


### --- 8. PLANCK & SOLAR GEOMETRY --- ###
def planck_integration(v_min, v_max, T):
    """Returns band-integrated Planck emission (W/m^2) for a layer at temperature T."""
    v_center = 0.5 * (v_min + v_max) * 100  # Band centre (m^-1)
    width_m  = (v_max - v_min) * 100         # Band width (m^-1)
    B_nu = (2 * h * c**2 * v_center**3) / (np.exp((h * c / k_B) * v_center / T) - 1)
    return np.pi * B_nu * width_m


def get_sun_distance_factor(day):
    """Returns (a/r)^2 to account for Earth's elliptical orbit (r = distance on given day)."""
    r = 1 - 0.01672 * np.cos((np.pi / 180) * (360 / 365.25 * (day - 4)))
    return 1.0 / r**2


def solar_flux_toa(v_min, v_max, distance_factor):
    """Returns the top-of-atmosphere solar flux (W/m^2) in a given spectral band."""
    flux_bb = planck_integration(v_min, v_max, T_sun)
    normalization = S0 / (sigma * (T_sun**4) * ((sun_radius / earth_sun_distance)**2))
    return flux_bb * ((sun_radius / earth_sun_distance)**2) * normalization * (
        distance_factor if seasonal_variation else 1)


def get_spherical_path(mu):
    """Returns the slant optical path length relative to vertical for solar zenith angle cos(mu)."""
    if mu <= 0:
        return 0
    return (np.sqrt((6371 + 50)**2 - (6371**2) * (1 - mu**2)) - 6371 * mu) / 50


def calculate_solar_parameters(day, lat):
    """
    Returns (insol_factor, path_factor, mu) for the given day and latitude.

    - insol_factor: daily-mean insolation fraction (geometry factor, 0–0.5)
    - path_factor:  optical path length relative to vertical (for slant-path absorption)
    - mu:           cosine of the solar zenith angle at local noon
    """
    declination = 23.44 * np.sin((np.pi / 180.0) * 360 / 365 * (day - 81))
    delta_rad, lat_rad = np.radians(declination), np.radians(lat)
    tan_prod = -np.tan(lat_rad) * np.tan(delta_rad)

    h0 = 0.0 if tan_prod >= 1.0 else (np.pi if tan_prod <= -1.0 else np.arccos(tan_prod))
    daily_avg_mu = (1.0 / np.pi) * (
        h0 * np.sin(lat_rad) * np.sin(delta_rad) +
        np.cos(lat_rad) * np.cos(delta_rad) * np.sin(h0))

    elevation_noon = 90 - abs(lat - declination)
    mu_noon = np.sin(np.radians(elevation_noon)) if elevation_noon > 0 else 0
    return max(0.0, daily_avg_mu), get_spherical_path(mu_noon), mu_noon


### --- 9. CONVECTIVE ADJUSTMENT --- ###
# Numba's @jit compiler is used here because the inner loops iterate vertically through
# the atmosphere on every time step for every latitude — standard Python loops would be
# prohibitively slow.

@jit(nopython=True)
def lnT_lnP_fast(T, P, R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap):
    """Returns the moist adiabatic lapse rate expressed as d(lnT)/d(lnP)."""
    P_vapour_sat = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    rs = (R_dry / R_vap) * P_vapour_sat / (P - P_vapour_sat)
    numerator   = 1 + (L_v * rs) / (R_dry * T)
    denominator = 1 + ((cp_h2o_vap / cp_dry_air) +
                       (L_v / (R_vap * T) - 1) * (L_v / (cp_dry_air * T)) * rs)
    return (R_dry / cp_dry_air) * (numerator / denominator)


@jit(nopython=True)
def convective_adjustment_fast(T_atm, T_surf, p_mid_local, p_surface_local,
                                C_air_local, C_s, R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap):
    """
    Restores convective stability layer by layer, conserving energy.

    If the actual lapse rate between two adjacent layers exceeds the moist adiabatic
    lapse rate, the two layers are mixed to the critical slope while conserving their
    combined enthalpy.  Passes over the column up to N^2 times until fully stable.
    Note: this version always uses the moist critical slope (mixed_convection is fixed
    True in the latitudinal model).
    """
    N = len(T_atm)
    T_column   = np.empty(N + 1); T_column[:N]   = T_atm;         T_column[N]   = T_surf
    P_column   = np.empty(N + 1); P_column[:N]   = p_mid_local;   P_column[N]   = p_surface_local
    Cap_column = np.empty(N + 1); Cap_column[:N] = C_air_local;   Cap_column[N] = C_s

    for _ in range(N**2):
        stable = True
        for i in range(N, 0, -1):
            actual_slope = ((np.log(T_column[i]) - np.log(T_column[i - 1])) /
                            (np.log(P_column[i]) - np.log(P_column[i - 1])))
            moist_slope = lnT_lnP_fast(
                0.5 * (T_column[i] + T_column[i - 1]),
                0.5 * (P_column[i] + P_column[i - 1]),
                R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap)

            critical_slope = max(moist_slope, 0.192)
            if actual_slope > critical_slope:
                stable = False
                target_ratio = (P_column[i] / P_column[i - 1]) ** critical_slope
                E_total = Cap_column[i] * T_column[i] + Cap_column[i - 1] * T_column[i - 1]
                T_column[i - 1] = E_total / (Cap_column[i - 1] + Cap_column[i] * target_ratio)
                T_column[i]     = T_column[i - 1] * target_ratio
        if stable:
            break

    return T_column[:-1], T_column[-1]


### --- 10. RADIATIVE TRANSFER --- ###

def _get_column_amounts(T_atm, ppm_co2, ppm_h2o, ppm_ch4, ppm_o3, dp_local):
    """
    Computes per-layer column amounts (molecules/cm^2) and CIA weighting factors
    for one latitude column.

    Both the LW and SW solvers need these quantities; centralising them here avoids
    repeating identical code in each solver.

    Returns
    -------
    u_total                                    : air column amount per layer (molecules/cm^2)
    u_co2, u_h2o, u_ch4, u_o3                 : per-species column amounts (molecules/cm^2)
    factor_n2n2, factor_n2o2, factor_o2o2, factor_co2co2 : CIA path factors
    """
    u_total = (dp_local / (g * m_air)) * Na * 1e-4
    u_co2   = u_total * (ppm_co2 * 1e-6)
    u_h2o   = u_total * (ppm_h2o * 1e-6)
    u_ch4   = u_total * (ppm_ch4 * 1e-6)
    u_o3    = u_total * (ppm_o3  * 1e-6)

    # CIA factors: product of number density and column amount for each colliding pair
    n_air         = (p_mid_local_ref / (k_B * T_atm)) * 1e-6  # Number density (cm^-3)
    factor_n2n2   = (0.78 * 0.78) * (n_air * u_total)
    factor_n2o2   = (0.78 * 0.21) * (n_air * u_total)
    factor_o2o2   = (0.21 * 0.21) * (n_air * u_total)
    factor_co2co2 = (ppm_co2 * 1e-6)**2 * (n_air * u_total)

    return u_total, u_co2, u_h2o, u_ch4, u_o3, factor_n2n2, factor_n2o2, factor_o2o2, factor_co2co2


# Module-level mutable reference used by _get_column_amounts to access the current
# latitude's p_mid without threading it through every call signature.
p_mid_local_ref = None


def calc_lw_radiation(T_atm, T_surf, ppm_co2, ppm_ch4, local_rh, tau_cloud_profile,
                      p_mid_local, dp_local, T_atm_h2o=None):
    """
    Computes longwave (thermal infrared) upward and downward flux profiles (W/m^2)
    for one latitude column.

    Uses the correlated-k method: for each spectral band, absorption coefficients are
    looked up from pre-computed HITRAN k-tables, optical depths are accumulated, and
    the two-stream equations are solved with calc_lw_fluxes.
    """
    global p_mid_local_ref
    p_mid_local_ref = p_mid_local

    T_water = T_atm if T_atm_h2o is None else T_atm_h2o
    ppm_h2o = get_h2o_ppm(T_water, p_mid_local, local_rh)
    ppm_o3  = get_ozone_ppm(p_mid_local)

    u_total, u_co2, u_h2o, u_ch4, u_o3, f_n2n2, f_n2o2, f_o2o2, f_co2co2 = \
        _get_column_amounts(T_atm, ppm_co2, ppm_h2o, ppm_ch4, ppm_o3, dp_local)

    log_p       = np.log10(p_mid_local / 101325.0)
    log_h2o_ppm = np.log10(np.maximum(ppm_h2o, 1e-10))
    co2_coords  = np.column_stack((log_p, T_atm))
    h2o_coords  = np.column_stack((log_p, T_atm, log_h2o_ppm))

    Fd_lw_total, Fu_lw_total = np.zeros(N + 1), np.zeros(N + 1)

    for band_index, band in enumerate(ktable_co2['bands']):
        # 1. Look up absorption cross-sections (k) from HITRAN tables
        k_co2 = co2_interps[band_index](co2_coords)
        k_h2o = h2o_interps[band_index](h2o_coords)
        k_ch4 = ch4_interps[band_index](co2_coords)
        k_o3  = o3_interps[band_index](co2_coords)

        # 2. Optical depth = absorption cross-section × column amount
        tau_co2 = k_co2 * u_co2[:, None]
        tau_h2o = k_h2o * u_h2o[:, None]
        tau_ch4 = k_ch4 * u_ch4[:, None]
        tau_o3  = k_o3  * u_o3[:, None]

        tau_cia = 0.0
        if include_CIA:
            tau_cia = (np.interp(T_atm, cia_t_grid, cia_lw['N2-N2'][band_index])   * f_n2n2 +
                       np.interp(T_atm, cia_t_grid, cia_lw['N2-O2'][band_index])   * f_n2o2 +
                       np.interp(T_atm, cia_t_grid, cia_lw['O2-O2'][band_index])   * f_o2o2 +
                       np.interp(T_atm, cia_t_grid, cia_lw['CO2-CO2'][band_index]) * f_co2co2)

        # 3. Total optical depth including diffusivity factor and cloud contribution
        tau = diffusivity_factor * (
            tau_ch4 + tau_co2 + tau_h2o + tau_o3 + tau_cia[:, None] + tau_cloud_profile[:, None])

        # 4. Solve the two-stream equations for this band
        B_atm  = planck_integration(band['v_min'], band['v_max'], T_atm)
        B_surf = planck_integration(band['v_min'], band['v_max'], T_surf)
        w_g    = ktable_co2['bands'][band_index]['w_g']

        Fd_band, Fu_band = calc_lw_fluxes(N, np.exp(-tau), 1 - np.exp(-tau), B_atm, B_surf, w_g)
        Fd_lw_total += Fd_band
        Fu_lw_total += Fu_band

    return Fd_lw_total, Fu_lw_total


def calc_sw_radiation(T_atm, Ts, day_of_year, ppm_co2, ppm_ch4, insol_factor, path_factor, mu,
                      local_albedo, local_rh, local_layer_reflectivity, p_mid_local, dp_local,
                      T_atm_h2o=None):
    """
    Computes shortwave (solar) downward and upward flux profiles (W/m^2)
    for one latitude column.

    Uses the same correlated-k framework as calc_lw_radiation, but accounts for
    scattering and reflection via the adding-doubling scheme in calc_sw_fluxes_reflect.
    The solar beam travels at an angle set by path_factor; reflected diffuse light
    uses the fixed diffusivity factor of 1.66.
    """
    if insol_factor <= 0:
        return np.zeros(N + 1), np.zeros(N + 1)

    global p_mid_local_ref
    p_mid_local_ref = p_mid_local

    T_water = T_atm if T_atm_h2o is None else T_atm_h2o
    ppm_h2o = get_h2o_ppm(T_water, p_mid_local, local_rh)
    ppm_o3  = get_ozone_ppm(p_mid_local)

    # SW only uses H2O and O3 as absorbers (plus CIA); CO2 and CH4 are negligible in the solar
    u_total, _, u_h2o, _, u_o3, f_n2n2, f_n2o2, f_o2o2, f_co2co2 = \
        _get_column_amounts(T_atm, ppm_co2, ppm_h2o, ppm_ch4, ppm_o3, dp_local)

    h2o_coords_sw = np.column_stack((np.log10(p_mid_local / 101325.0), T_atm,
                                     np.log10(np.maximum(ppm_h2o, 1e-10))))
    o3_sw_coords  = np.column_stack([T_atm])

    Fd_sw_total, Fu_sw_total = np.zeros(N + 1), np.zeros(N + 1)
    dist_factor  = get_sun_distance_factor(day_of_year)
    local_albedo = local_albedo if include_albedo else 0.0

    for band_index, band in enumerate(ktable_h2o_sw['bands']):
        S_toa = solar_flux_toa(band['v_min'], band['v_max'], dist_factor) * insol_factor
        k_h2o = h2o_sw_interps[band_index](h2o_coords_sw)
        k_o3  = o3_sw_interps[band_index](o3_sw_coords)

        # Separate down/up paths: solar beam uses the geometric path_factor; diffuse uses 1.66
        tau_h2o_down = k_h2o * u_h2o[:, None] * path_factor
        tau_o3_down  = k_o3  * u_o3[:, None]  * path_factor
        tau_h2o_up   = k_h2o * u_h2o[:, None] * 1.66
        tau_o3_up    = k_o3  * u_o3[:, None]  * 1.66

        tau_cia = 0.0
        if include_CIA:
            tau_cia = (np.interp(T_atm, cia_t_grid, cia_sw['N2-N2'][band_index])   * f_n2n2 +
                       np.interp(T_atm, cia_t_grid, cia_sw['N2-O2'][band_index])   * f_n2o2 +
                       np.interp(T_atm, cia_t_grid, cia_sw['O2-O2'][band_index])   * f_o2o2 +
                       np.interp(T_atm, cia_t_grid, cia_sw['CO2-CO2'][band_index]) * f_co2co2)

        tau_down = tau_h2o_down + tau_o3_down + (tau_cia * path_factor)[:, None]
        tau_up   = tau_h2o_up   + tau_o3_up   + (tau_cia * 1.66)[:, None]

        w_g = ktable_h2o_sw['bands'][band_index]['w_g']
        Fd_band, Fu_band = calc_sw_fluxes_reflect(
            N, np.exp(-tau_down), np.exp(-tau_up),
            local_layer_reflectivity, S_toa, local_albedo, w_g)
        Fd_sw_total += Fd_band
        Fu_sw_total += Fu_band

    return Fd_sw_total, Fu_sw_total


### --- 11. FLUX PROPAGATION SOLVERS --- ###
# Numba JIT is used for the same reason as in section 9: these solvers are called once
# per band per latitude per time step, making their inner loops the innermost bottleneck.

@jit(nopython=True)
def calc_lw_fluxes(N, transmission, emissivity, B_atm, B_surf, w_g):
    """
    Sweeps up and down through the atmosphere to compute LW flux at each interface.

    The downward sweep propagates emission from the top down; the upward sweep
    propagates surface emission and atmospheric re-emission from the bottom up.
    Gaussian quadrature weights (w_g) sum over the g-point distribution.
    """
    Fd_g, Fd_band = np.zeros(transmission.shape[1]), np.zeros(N + 1)
    for i in range(N):
        Fd_g = transmission[i, :] * Fd_g + emissivity[i, :] * B_atm[i]
        Fd_band[i + 1] = np.sum(Fd_g * w_g)

    Fu_g, Fu_band = np.full(transmission.shape[1], B_surf), np.zeros(N + 1)
    Fu_band[N] = B_surf
    for i in range(N - 1, -1, -1):
        Fu_g = transmission[i, :] * Fu_g + emissivity[i, :] * B_atm[i]
        Fu_band[i] = np.sum(Fu_g * w_g)
    return Fd_band, Fu_band


@jit(nopython=True)
def calc_sw_fluxes_reflect(N, transmission_down, transmission_up, reflectivity,
                            S_toa, albedo_surface, w_g):
    """
    Solves SW flux through a stack of partially reflecting and absorbing layers.

    Uses an adding-doubling approach: first computes the effective combined albedo
    looking downward from each interface, then propagates the solar beam downward
    accounting for reflection at each layer and the surface.
    """
    alpha_layer = reflectivity.reshape((N, 1))
    t_eff_down  = transmission_down * (1.0 - alpha_layer)
    t_eff_up    = transmission_up   * (1.0 - alpha_layer)

    # Build combined albedo from the bottom up (surface → TOA)
    alpha_combined = np.zeros((N + 1, transmission_down.shape[1]))
    alpha_combined[N, :] = albedo_surface
    for i in range(N - 1, -1, -1):
        alpha_combined[i, :] = alpha_layer[i, 0] + (
            (alpha_combined[i + 1, :] * t_eff_down[i, :] * t_eff_up[i, :]) /
            (1.0 - alpha_layer[i, 0] * alpha_combined[i + 1, :]))

    # Propagate downward flux; upward flux = downward flux × combined albedo below
    Fd = np.zeros((N + 1, transmission_down.shape[1]))
    Fd[0, :] = S_toa
    for i in range(N):
        Fd[i + 1, :] = Fd[i, :] * t_eff_down[i, :] / (
            1.0 - alpha_layer[i, 0] * alpha_combined[i + 1, :])
    Fu = Fd * alpha_combined

    Fd_band, Fu_band = np.zeros(N + 1), np.zeros(N + 1)
    for i in range(N + 1):
        Fd_band[i] = np.sum(Fd[i, :] * w_g)
        Fu_band[i] = np.sum(Fu[i, :] * w_g)
    return Fd_band, Fu_band


### --- 12. MAIN SIMULATION ENGINE --- ###
def run_model():
    """Integrates the 2D model forward in time across all latitudes."""
    dt = dt_days * 24 * 3600  # Time step (seconds)
    nsteps         = int(years * 365.25 * 24 * 3600 / dt)
    steps_per_year = int(365.25 * 24 * 3600 / dt)
    switch_step    = int(nsteps * 0.5) if perturb else nsteps + 1

    lats = lats_grid
    x_grid, dx_grid, x_bounds, bound_lens = spherical_grid(lats)

    T_atm_curr = np.zeros((num_lats, N))
    for i in range(num_lats):
        T_atm_curr[i, :] = T_initial.copy()
    Ts_curr = ts_init_arr.astype(float)

    # History arrays (written at every time step for post-run plotting)
    date_history    = []
    hist_Ts         = np.zeros((nsteps, num_lats))
    hist_Tatm       = np.zeros((nsteps, num_lats, N))
    hist_imbalance  = np.zeros((nsteps, num_lats))
    hist_albedo     = np.zeros((nsteps, num_lats))
    hist_Fd_sw      = np.zeros((nsteps, num_lats, N + 1))
    hist_Fu_sw      = np.zeros((nsteps, num_lats, N + 1))
    hist_Fd_lw      = np.zeros((nsteps, num_lats, N + 1))
    hist_Fu_lw      = np.zeros((nsteps, num_lats, N + 1))
    hist_heating_sw = np.zeros((nsteps, num_lats, N))
    hist_heating_lw = np.zeros((nsteps, num_lats, N))

    # "Ghost year" climatology: records the full seasonal cycle of the year immediately
    # before the CO2 perturbation, used to lock the surface temperature or water vapour
    # profile in ERF experiments.
    Ts_climatology   = np.zeros((steps_per_year, num_lats))
    T_atm_climatology = np.zeros((steps_per_year, num_lats, N))

    start_date      = datetime.datetime(start_year, start_month, start_day)
    initial_doy     = start_date.timetuple().tm_yday - 1
    seconds_in_year = 365.25 * 86400

    print(f"\nIntegration: {dt_days} day steps")
    if perturb:
        print(f"ERF: CO2 doubles at step {switch_step}/{nsteps}")
    print(f"Running model from {start_date.strftime('%Y-%m-%d')} for {years} years...\n")

    for n in range(nsteps):
        time_elapsed = n * dt
        step_in_year = n % steps_per_year
        doy = (initial_doy + time_elapsed / 86400) % 365.25
        date_history.append(start_date + datetime.timedelta(seconds=time_elapsed))

        # Record the ghost-year climatology in the last year before the perturbation
        if perturb and (switch_step - steps_per_year <= n < switch_step):
            Ts_climatology[step_in_year, :] = Ts_curr.copy()
            if fixed_water_feedback:
                T_atm_climatology[step_in_year, :, :] = T_atm_curr.copy()

        if perturb and n >= switch_step:
            current_co2          = initial_co2_ppm * 2
            current_ch4_surface  = initial_ch4_ppm
            lock_surface         = lock_surface_after_switch
        else:
            current_co2         = initial_co2_ppm + (time_elapsed / seconds_in_year) * co2_ppm_rate
            current_ch4_surface = initial_ch4_ppm + (time_elapsed / seconds_in_year) * ch4_ppm_rate
            lock_surface        = False

        effective_day = doy if seasonal_variation else initial_doy

        # Interpolate cloud and albedo fields to today's day-of-year for every latitude
        curr_albedos    = np.array([np.interp(effective_day, days_padded, albedo_grid[:, i])
                                    for i in range(num_lats)])
        curr_cld_albedos = np.array([np.interp(effective_day, days_padded, cld_albedo_grid[:, i])
                                     for i in range(num_lats)])
        curr_cld_fracs  = np.array([np.interp(effective_day, days_padded, cld_frac_grid[:, i])
                                    for i in range(num_lats)])
        curr_cld_press  = np.array([np.interp(effective_day, days_padded, cld_press_grid[:, i])
                                    for i in range(num_lats)])

        rate_atm_all = np.zeros((num_lats, N))
        rate_s_all   = np.zeros(num_lats)
        temp_Fd_sw_0 = np.zeros(num_lats)
        temp_Fu_sw_0 = np.zeros(num_lats)
        temp_Fd_lw_0 = np.zeros(num_lats)
        temp_Fu_lw_0 = np.zeros(num_lats)

        # --- Loop over all latitude columns ---
        for i in range(num_lats):
            lat           = lats[i]
            loc_albedo    = get_dynamic_albedo(Ts_curr[i], curr_albedos[i])
            loc_Cs        = Cs_arr[i]
            loc_rh        = rh_arr[i]
            local_p_mid   = p_mid[i, :]
            local_dp      = dp[i, :]

            ch4_profile_local = get_ch4_ppm(local_p_mid, current_ch4_surface)
            loc_reflect, loc_tau = cloud_profiles(
                curr_cld_press[i], curr_cld_fracs[i], curr_cld_albedos[i], local_p_mid, local_dp)
            insol_fac, path_fac, mu_val = calculate_solar_parameters(effective_day, lat)

            # Use locked water vapour profile during perturbation run if requested
            T_h2o_pass = (T_atm_climatology[step_in_year, i, :]
                          if (fixed_water_feedback and perturb and n >= switch_step) else None)

            # --- Radiative transfer ---
            Fd_lw, Fu_lw = calc_lw_radiation(
                T_atm_curr[i, :], Ts_curr[i], current_co2, ch4_profile_local,
                loc_rh, loc_tau, local_p_mid, local_dp, T_atm_h2o=T_h2o_pass)
            Fd_sw, Fu_sw = calc_sw_radiation(
                T_atm_curr[i, :], Ts_curr[i], doy, current_co2, ch4_profile_local,
                insol_fac, path_fac, mu_val, loc_albedo, loc_rh, loc_reflect,
                local_p_mid, local_dp, T_atm_h2o=T_h2o_pass)

            temp_Fd_sw_0[i] = Fd_sw[0]; temp_Fu_sw_0[i] = Fu_sw[0]
            temp_Fd_lw_0[i] = Fd_lw[0]; temp_Fu_lw_0[i] = Fu_lw[0]

            # --- Heating rates: divergence of net flux in each layer (W/m^2) ---
            heating_lw = (Fu_lw - Fd_lw)[1:] - (Fu_lw - Fd_lw)[:-1]
            heating_sw = (Fu_sw - Fd_sw)[1:] - (Fu_sw - Fd_sw)[:-1]

            hist_Fd_sw[n, i, :] = Fd_sw;      hist_Fu_sw[n, i, :] = Fu_sw
            hist_Fd_lw[n, i, :] = Fd_lw;      hist_Fu_lw[n, i, :] = Fu_lw
            hist_heating_sw[n, i, :] = heating_sw
            hist_heating_lw[n, i, :] = heating_lw

            rate_atm_all[i, :] = (heating_lw + heating_sw) / C_air[i, :]
            rate_s_all[i] = (((Fd_lw[-1] - Fu_lw[-1]) + (Fd_sw[-1] - Fu_sw[-1])) / loc_Cs
                             if not lock_surface else 0.0)

        # --- Update temperatures ---
        T_atm_curr += rate_atm_all * dt

        # Surface: free integration or replay the locked ghost-year climatology
        Ts_curr = (Ts_curr + rate_s_all * dt if not lock_surface
                   else Ts_climatology[step_in_year, :].copy())

        # Meridional diffusion (applied before convection so convection can correct any
        # horizontal gradients introduced by diffusion)
        if diffusion and not lock_surface:
            Ts_curr = apply_diffusion(Ts_curr, Cs_arr, x_grid, dx_grid, bound_lens, dt, p_surface_arr)

        # Vertical convective adjustment for every latitude column
        for i in range(num_lats):
            if convection:
                T_atm_curr[i, :], Ts_curr[i] = convective_adjustment_fast(
                    T_atm_curr[i, :], Ts_curr[i], p_mid[i, :], p_surface_arr[i],
                    C_air[i, :], Cs_arr[i], R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap)

            hist_Ts[n, i]        = Ts_curr[i]
            hist_Tatm[n, i, :]   = T_atm_curr[i, :]
            hist_imbalance[n, i] = ((temp_Fd_sw_0[i] - temp_Fu_sw_0[i]) -
                                    (temp_Fu_lw_0[i] - temp_Fd_lw_0[i]))
            hist_albedo[n, i]    = (temp_Fu_sw_0[i] / temp_Fd_sw_0[i]
                                    if temp_Fd_sw_0[i] > 0 else 0)

        if n % 10 == 0:
            status       = f" ({'LOCKED' if lock_surface else 'FREE'})" if perturb else ""
            instant_avg  = np.average(Ts_curr, weights=dx_grid)
            rolling_avg  = np.average(
                np.mean(hist_Ts[max(0, n + 1 - steps_per_year):n + 1, :], axis=0),
                weights=dx_grid)
            print(f"Step {n}/{nsteps}: Instant Ts = {instant_avg:.2f} K{status}, "
                  f"CO2 = {current_co2:.1f} ppm, 1-yr rolling Ts = {rolling_avg:.2f} K")

    return (date_history, lats, rh_arr, hist_Ts, hist_Tatm, hist_albedo, hist_imbalance,
            hist_Fd_sw, hist_Fu_sw, hist_Fd_lw, hist_Fu_lw, hist_heating_sw, hist_heating_lw)


### --- 13. EXECUTION & SAVE --- ###
(dates, lats_out, rh_out, Ts_history, Tatm_history, albedo_history,
 imbalance_history, Fd_sw_hist, Fu_sw_hist, Fd_lw_hist, Fu_lw_hist,
 heat_sw_hist, heat_lw_hist) = run_model()

print("\n" + "=" * 50)
print("            MODEL SIMULATION COMPLETE")
print("=" * 50)

print(f"\nSIMULATION SETTINGS")
print(f"{'Total Duration:':<25} {years} years")
print(f"{'Time Step (dt):':<25} {dt_days} days")
print(f"{'Vertical Layers (N):':<25} {N}")

print(f"\nMODEL PHYSICS (FLAGS)")
physics_flags = {
    "Topography":     include_topography,
    "Convection":     convection,
    "CIA":            include_CIA,
    "Albedo":         include_albedo,
    "Clouds":         include_clouds,
    "Blanket":        include_blanket,
    "Seasonal":       seasonal_variation,
    "Dyn Ice":        dynamic_ice_albedo,
    "Diffusion":      diffusion,
}
for name, status in physics_flags.items():
    print(f"  {name:<15}: {'ON' if status else 'OFF'}")

print(f"\nRun time: {time.time() - start_time:.2f} seconds")

print("\nCompressing and saving model output...")
os.makedirs("results", exist_ok=True)
save_path = "results/latitudinal_model_output.npz"

run_params = {
    'start_year': start_year, 'start_month': start_month, 'start_day': start_day,
    'years': years, 'dt_days': dt_days, 'N': N,
    'convection': convection, 'log_pressure': log_pressure, 'include_CIA': include_CIA,
    'seasonal_variation': seasonal_variation,
    'include_albedo': include_albedo, 'include_clouds': include_clouds, 'include_blanket': include_blanket,
    'perturb': perturb,
    'fixed_water_feedback': fixed_water_feedback, 'dynamic_ice_albedo': dynamic_ice_albedo,
    'lock_surface_after_switch': lock_surface_after_switch,
    'diffusion': diffusion, 'include_topography': include_topography,
    'heat_capacity_factor': heat_capacity_factor, 'D_coeff': D_coeff,
    'initial_co2_ppm': initial_co2_ppm, 'co2_ppm_rate': co2_ppm_rate,
    'initial_ch4_ppm': initial_ch4_ppm, 'ch4_ppm_rate': ch4_ppm_rate,
    'ozone_peak_ppm': ozone_peak_ppm, 'relative_humidity': relative_humidity,
    'diffusivity_factor': diffusivity_factor,
}

np.savez_compressed(save_path,
                    dates       = np.asarray([d.strftime('%Y-%m-%d') for d in dates]),
                    lats        = np.asarray(lats_out),
                    p_mid       = np.asarray(p_mid),
                    p_int       = np.asarray(p_int),
                    Ts          = np.asarray(Ts_history),
                    albedo      = np.asarray(albedo_history),
                    imbalance   = np.asarray(imbalance_history),
                    Tatm        = np.asarray(Tatm_history),
                    Fd_sw       = np.asarray(Fd_sw_hist),
                    Fu_sw       = np.asarray(Fu_sw_hist),
                    Fd_lw       = np.asarray(Fd_lw_hist),
                    Fu_lw       = np.asarray(Fu_lw_hist),
                    heating_sw  = np.asarray(heat_sw_hist),
                    heating_lw  = np.asarray(heat_lw_hist),
                    params      = run_params)

print(f"\nSUCCESS: Simulation data saved to '{save_path}'")
print("-> Please run '2b_plot_latitudinal.py' to visualize the results.")