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

T_sun = 5778.0         # Sun surface temperature (K)
sun_radius = 695700000        # Radius of the Sun (m)
earth_sun_distance = 149597870700  # Mean Earth-Sun distance (m)


### --- 2. SIMULATION PARAMETERS & TOGGLES --- ###

# -- Time & Grid Settings --
years = 0.1
dt_days = 0.2   # Time step in days (max ~0.5 days for 50 layers to remain stable)
N = 50          # Number of vertical atmospheric layers
Ts_initial = 287  # Initial surface temperature (K)
C_s = 1e6         # Surface heat capacity (J/m^2/K); lowered for faster equilibration

# -- Location Settings --
global_average = True  # If True, uses global-mean geometry; if False, uses specific latitude
latitude = 45          # Latitude to simulate (degrees N; only used if global_average=False)
start_year, start_month, start_day = 2014, 1, 1

# -- Atmospheric Composition --
initial_co2_ppm = 400    # Initial CO2 concentration (ppm)
co2_ppm_rate = 0         # Annual CO2 increase (ppm/year)
initial_ch4_ppm = 2      # Initial CH4 concentration (ppm)
ch4_ppm_rate = 0         # Annual CH4 increase (ppm/year)
ozone_peak_ppm = 7       # Peak stratospheric ozone concentration (ppm)
relative_humidity = 0.77 # Surface relative humidity (fraction, 0–1)

# -- Physics Toggles --
convection = True         # Enable convective adjustment
mixed_convection = True   # Use mixed dry/moist adiabatic lapse rate (vs. pure moist)
log_pressure = True       # True = logarithmic layer spacing; False = linear spacing
seasonal_variation = False  # If False, solar geometry is fixed to the start date
include_CIA = True          # Include Collision-Induced Absorption

# -- Cloud & Albedo Toggles --
include_albedo = True    # Enable surface albedo and Rayleigh SW reflection
include_clouds = True    # Enable cloud shortwave reflection
include_blanket = True   # Enable cloud longwave trapping
observation_data = False # Load cloud/albedo data from CSV (else use hardcoded defaults)

# -- Experimental Forcing Toggles --
perturb = False               # If True, doubles CO2 halfway through the run
fixed_water_feedback = False  # If True, locks the water vapour profile after perturbation
lock_surface_after_switch = False  # If True, fixes Ts after the CO2 perturbation

diffusivity_factor = 1.66       # Diffusivity factor for angular integration of LW flux
turn_on_random_overlap = False  # Use random overlap assumption for gas bands


### --- 3. ATMOSPHERIC GRID SETUP --- ###
# Pressure interfaces from the top of the atmosphere (p_top) down to the surface (p_surface).
p_surface = 1.0e5   # Surface pressure (Pa)
p_top = 10          # Top-of-atmosphere pressure (Pa)
p_tropopause = 20000.0  # Approximate tropopause pressure (Pa)
p_stratopause = 100     # Approximate stratopause pressure (Pa)

if log_pressure:
    p_int = np.geomspace(p_top, p_surface, N)  # Logarithmically spaced interfaces
else:
    p_int = np.linspace(p_top, p_surface, N)   # Linearly spaced interfaces

p_mid = 0.5 * (p_int[:-1] + p_int[1:])  # Midpoint pressure of each layer (Pa)
dp = np.diff(p_int)                       # Pressure thickness of each layer (Pa)

# Pre-compute the column air mass per layer (molecules/m^2); used throughout for optical depths.
# Also pre-compute the layer heat capacity (J/m^2/K) used in the time-stepping.
u_total = (dp / (g * m_air)) * Na * 1e-4  # Column number density per layer (molecules/cm^2)
C_air = cp_dry_air * dp / g               # Heat capacity per layer (J/m^2/K)

print(f"Bottom Layer Thickness: {dp[-1] / 100:.1f} hPa")
print(f"Top Layer Thickness:    {dp[0] / 100:.1f} hPa")

# Initial temperature profile following the standard atmospheric structure
n_meso  = np.sum(p_mid < p_stratopause)
n_strat = np.sum((p_mid >= p_stratopause) & (p_mid < p_tropopause))
n_trop  = len(p_mid) - n_meso - n_strat

T_meso  = np.linspace(220, 270, n_meso)
T_strat = np.linspace(270, 215, n_strat)
T_trop  = np.linspace(215, 290, n_trop)
T_initial = np.concatenate((T_meso, T_strat, T_trop))


### --- 4. CLOUD & ALBEDO PROFILES --- ###
total_rayleigh_reflectivity = 0.1  # Total column SW reflectivity due to Rayleigh scattering

# Cloud and albedo arrays are stored as time series (monthly or two-point) and interpolated
# each time step. The arrays are padded with wrap-around values at day -15 and day 380 to
# avoid edge artefacts when interpolating near the start and end of the year.
if observation_data:
    df_params = pd.read_csv('parameters/latitude_params_monthly.csv')
    df_global = df_params[df_params['Latitude'] == 'Global']
    month_days = np.array([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345])
    days_padded = np.concatenate([[-15], month_days, [380]])

    def pad_data(arr):
        """Wraps the first/last month to both ends for smooth year-round interpolation."""
        return np.concatenate([[arr[-1]], arr, [arr[0]]])

    albedo_arr   = pad_data(df_global['Albedo'].values)
    cld_press_arrs = [pad_data(df_global['Cloud_Pressure'].values)]
    cld_frac_arrs  = [pad_data(df_global['Cloud_Fraction'].values)]
    cld_alb_arrs   = [pad_data(df_global['Cloud_Albedo'].values)]
    cld_tau_arrs   = [pad_data(np.full(12, 2.0))]
else:
    # Two-point (constant) profiles: values at day -15, 180, and 380 are all identical,
    # giving a flat seasonal profile by default.
    days_padded    = np.array([-15, 180, 380])
    albedo_arr     = np.array([0.135, 0.135, 0.135])
    cld_press_arrs = [np.full(3, 85000.0), np.full(3, 25000.0)]  # Low and high cloud (Pa)
    cld_frac_arrs  = [np.full(3, 0.3),     np.full(3, 0.2)]      # Cloud fractions
    cld_alb_arrs   = [np.full(3, 0.7),     np.full(3, 0.2)]      # SW cloud albedos
    cld_tau_arrs   = [np.full(3, 2.0),     np.full(3, 1.5)]      # LW optical depths

    if not include_clouds:
        cld_frac_arrs = [np.full(3, 0.0), np.full(3, 0.0)]


def get_cloud_profiles(cld_pressures, cld_fractions, cld_albedos, cld_taus, p_mid, dp):
    """Builds per-layer SW reflectivity and LW optical depth profiles from cloud parameters."""
    loc_reflectivity = total_rayleigh_reflectivity * (dp / np.sum(dp))
    loc_tau_lw = np.zeros(len(p_mid))

    if not include_albedo:
        loc_reflectivity[:] = 0

    if include_clouds:
        for i in range(len(cld_pressures)):
            idx_cld = (np.abs(p_mid - cld_pressures[i])).argmin()

            if include_albedo:
                loc_reflectivity[idx_cld] += cld_fractions[i] * cld_albedos[i]
                loc_reflectivity = np.minimum(loc_reflectivity, 0.999)

            if include_blanket:
                # Effective LW transmittance: clear-sky fraction + cloud fraction * Beer-Lambert
                trans = (1 - cld_fractions[i]) + cld_fractions[i] * np.exp(-cld_taus[i])
                loc_tau_lw[idx_cld] += -np.log(np.maximum(trans, 1e-10))

    return loc_reflectivity, loc_tau_lw


### --- 5. DATA LOADING & INTERPOLATION (HITRAN) --- ###
print("Loading pre-computed K tables (HITRAN)...")
ktable_co2    = np.load("data/hitran_tables/k_table_co2_lw_gauss_8.npz", allow_pickle=True)['k_table'].item()
ktable_h2o    = np.load("data/hitran_tables/k_table_h2o_lw_gauss_8.npz",        allow_pickle=True)['k_table'].item()
ktable_h2o_sw = np.load("data/hitran_tables/k_table_h2o_sw_gauss_8.npz", allow_pickle=True)['k_table'].item()
ktable_ch4    = np.load("data/hitran_tables/k_table_ch4_gauss_8.npz", allow_pickle=True)['k_table'].item()
ktable_o3     = np.load("data/hitran_tables/k_table_o3_lw_gauss_8.npz", allow_pickle=True)['k_table'].item()
ktable_o3_sw  = np.load("data/hitran_tables/k_table_o3_sw_gauss_8.npz", allow_pickle=True)['k_table'].item()

cia_data   = np.load("data/hitran_tables/cia_table_combined.npz", allow_pickle=True)
cia_t_grid = cia_data['T_grid']
cia_lw     = cia_data['cia_lw'].item()
cia_sw     = cia_data['cia_sw'].item()

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


### --- 6. GAS DISTRIBUTIONS --- ###
def get_h2o_ppm(T, P_Pa):
    """
    Returns water vapour concentration (ppm) at each model layer.

    Troposphere: relative humidity scaled by the Magnus-Tetens saturation vapour pressure.
    Stratosphere: fixed background controlled by CH4 oxidation. In the real stratosphere,
    methane is oxidised to produce water vapour; each mole of CH4 destroyed yields ~2 moles
    of H2O. This is approximated here with a fixed CH4 decay profile.
    The two regimes are blended smoothly across the tropopause.
    """
    T_c = T - 273.15
    # Magnus-Tetens formula: saturation vapour pressure (Pa)
    exp_term = (18.678 - T_c / 234.5) * (T_c / (257.14 + T_c))
    p_sat = 0.61121 * np.exp(exp_term) * 1000.0

    # Relative humidity decreases to zero above the tropopause
    p0 = np.max(P_Pa)
    pressure_factor = np.clip((P_Pa / p0 - 0.02) / (1.0 - 0.02), 0.0, 1.0)
    rh = relative_humidity * pressure_factor
    q_local = (rh * p_sat / P_Pa) * 1e6  # Tropospheric water vapour (ppm)

    # Stratospheric H2O from CH4 oxidation: H2O_strat ≈ q_entry + 2*(CH4_surface - CH4(z))
    q_entry = 3.5  # Water vapour at tropopause entry (ppm)
    ch4_surface, ch4_top = 1.8, 0.3
    log_p = np.log(np.clip(P_Pa, 1.0, None))
    ch4_weight = 1.0 / (1.0 + np.exp(-(log_p - np.log(3000.0)) / 0.75))
    current_ch4 = ch4_top + (ch4_surface - ch4_top) * ch4_weight
    q = q_entry + 2.0 * (ch4_surface - current_ch4)

    # Smooth blend: w_trop=1 in troposphere, w_trop=0 in stratosphere
    w_trop = 1.0 / (1.0 + np.exp((log_p - np.log(20000.0)) / 0.2))
    final_ppm = np.maximum(w_trop * q + (1.0 - w_trop) * q_local, q)

    return np.zeros_like(T) if relative_humidity == 0 else final_ppm


def get_ozone_ppm(P_Pa):
    """Returns ozone concentration (ppm) as a Gaussian peak in the stratosphere."""
    base_ozone = 0.04        # Tropospheric background (ppm)
    peak_pressure = 8 * 100  # Peak ozone pressure (~8 hPa, mid-stratosphere)
    width = np.where(P_Pa < peak_pressure, 1.5, 2)
    stratospheric_ozone = ozone_peak_ppm * np.exp(
        -np.abs((np.log(P_Pa) - np.log(peak_pressure)) / width) ** 2)
    return base_ozone + stratospheric_ozone


def get_ch4_ppm(P_Pa, current_surface_ppm):
    """Returns methane concentration (ppm) decaying from the surface value to a stratospheric background."""
    strat_background = 0.2  # Stratospheric CH4 background (ppm)
    weight = 1.0 / (1.0 + np.exp(-(np.log(P_Pa) - np.log(1000.0)) / 1))
    return strat_background + (current_surface_ppm - strat_background) * weight


### --- 7. PLANCK & SOLAR GEOMETRY --- ###
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
    flux_blackbody = planck_integration(v_min, v_max, T_sun)
    total_theoretical_flux = sigma * (T_sun**4) * ((sun_radius / earth_sun_distance)**2)
    normalization = S0 / total_theoretical_flux  # Normalise Planck spectrum to observed S0

    distance_factor = distance_factor if seasonal_variation else 1
    return flux_blackbody * ((sun_radius / earth_sun_distance)**2) * normalization * distance_factor


def calculate_solar_parameters(day, lat, is_global):
    """
    Returns (insol_factor, path_factor, mu) for the given day and latitude.
    - insol_factor: fraction of S0 reaching this column (geometry factor)
    - path_factor:  optical path length relative to vertical (for slant-path absorption)
    - mu:           cosine of the solar zenith angle at local noon
    """
    if is_global:
        return 0.25, 1.85, 0.5  # Global-average geometric constants

    declination = 23.44 * np.sin((np.pi / 180.0) * 360 / 365 * (day - 81))
    delta_rad, lat_rad = np.radians(declination), np.radians(lat)
    tan_prod = -np.tan(lat_rad) * np.tan(delta_rad)

    h0 = 0.0 if tan_prod >= 1.0 else (np.pi if tan_prod <= -1.0 else np.arccos(tan_prod))
    daily_avg_mu = (1.0 / np.pi) * (
        h0 * np.sin(lat_rad) * np.sin(delta_rad) + np.cos(lat_rad) * np.cos(delta_rad) * np.sin(h0))

    elevation_noon = 90 - abs(lat - declination)
    mu_noon = np.sin(np.radians(elevation_noon)) if elevation_noon > 0 else 0

    path_factor = 0 if mu_noon <= 0 else (
        np.sqrt((6371 + 50)**2 - (6371**2) * (1 - mu_noon**2)) - 6371 * mu_noon) / 50
    return max(0.0, daily_avg_mu), path_factor, mu_noon


### --- 8. CONVECTIVE ADJUSTMENT --- ###
# Numba's @jit compiler is used here because the inner loops iterate vertically through
# the atmosphere on every time step — standard Python loops would be prohibitively slow.

@jit(nopython=True)
def lnT_lnP_fast(T, P, R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap):
    """Returns the moist adiabatic lapse rate expressed as d(lnT)/d(lnP)."""
    P_vapour_sat = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    rs = (R_dry / R_vap) * P_vapour_sat / (P - P_vapour_sat)
    numerator   = 1 + (L_v * rs) / (R_dry * T)
    denominator = 1 + ((cp_h2o_vap / cp_dry_air) + (L_v / (R_vap * T) - 1) * (L_v / (cp_dry_air * T)) * rs)
    return (R_dry / cp_dry_air) * (numerator / denominator)


@jit(nopython=True)
def convective_adjustment_fast(T_atm, T_surf, p_mid, p_surface, C_air, C_s,
                                R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap):
    """
    Restores convective stability layer by layer, conserving energy.

    If the actual lapse rate between two adjacent layers exceeds the critical
    (moist or mixed) adiabatic lapse rate, the two layers are mixed to the
    critical slope while conserving their combined enthalpy.
    Passes over the column up to N^2 times until it is fully stable.
    """
    N = len(T_atm)
    T_column   = np.empty(N + 1); T_column[:N]   = T_atm;    T_column[N]   = T_surf
    P_column   = np.empty(N + 1); P_column[:N]   = p_mid;    P_column[N]   = p_surface
    Cap_column = np.empty(N + 1); Cap_column[:N] = C_air;    Cap_column[N] = C_s

    for _ in range(N**2):
        stable = True
        for i in range(N, 0, -1):
            d_lnP = np.log(P_column[i]) - np.log(P_column[i - 1])
            d_lnT = np.log(T_column[i]) - np.log(T_column[i - 1])
            actual_slope = d_lnT / d_lnP

            moist_slope = lnT_lnP_fast(
                0.5 * (T_column[i] + T_column[i - 1]),
                0.5 * (P_column[i] + P_column[i - 1]),
                R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap)
            critical_slope = max(moist_slope, 0.192) if mixed_convection else moist_slope

            if actual_slope > critical_slope:
                stable = False
                E_total      = Cap_column[i] * T_column[i] + Cap_column[i - 1] * T_column[i - 1]
                target_ratio = (P_column[i] / P_column[i - 1]) ** critical_slope
                T_column[i - 1] = E_total / (Cap_column[i - 1] + Cap_column[i] * target_ratio)
                T_column[i]     = T_column[i - 1] * target_ratio
        if stable:
            break

    return T_column[:-1], T_column[-1]


### --- 9. RADIATIVE TRANSFER --- ###

def _get_column_amounts(T_atm, ppm_co2, ppm_h2o, ppm_ch4, ppm_o3):
    """
    Computes per-layer column amounts (molecules/cm^2) and CIA weighting factors.

    These quantities are needed by both the LW and SW solvers.  Centralising the
    calculation here avoids repeating identical code in each solver.

    Returns
    -------
    u_co2, u_h2o, u_ch4, u_o3 : per-layer column amounts (molecules/cm^2)
    factor_n2n2, factor_n2o2, factor_o2o2, factor_co2co2 : CIA path factors
    """
    u_co2 = u_total * (ppm_co2 * 1e-6)
    u_h2o = u_total * (ppm_h2o * 1e-6)
    u_ch4 = u_total * (ppm_ch4 * 1e-6)
    u_o3  = u_total * (ppm_o3  * 1e-6)

    # CIA factors: product of number density and column amount for each colliding pair
    n_air = (p_mid / (k_B * T_atm)) * 1e-6   # Number density (cm^-3)
    factor_n2n2   = (0.78 * 0.78) * (n_air * u_total)
    factor_n2o2   = (0.78 * 0.21) * (n_air * u_total)
    factor_o2o2   = (0.21 * 0.21) * (n_air * u_total)
    factor_co2co2 = (ppm_co2 * 1e-6)**2 * (n_air * u_total)

    return u_co2, u_h2o, u_ch4, u_o3, factor_n2n2, factor_n2o2, factor_o2o2, factor_co2co2


def calc_lw_radiation(T_atm, T_surf, ppm_co2, ppm_ch4, local_tau_lw, T_atm_h2o=None):
    """
    Computes longwave (thermal infrared) upward and downward flux profiles (W/m^2).

    Uses the correlated-k method: for each spectral band, absorption coefficients are
    looked up from pre-computed HITRAN k-tables, optical depths are accumulated, and
    the two-stream equations are solved with calc_lw_fluxes.
    """
    T_atm_h2o = T_atm if T_atm_h2o is None else T_atm_h2o
    ppm_h2o = get_h2o_ppm(T_atm_h2o, p_mid)
    ppm_o3  = get_ozone_ppm(p_mid)

    u_co2, u_h2o, u_ch4, u_o3, f_n2n2, f_n2o2, f_o2o2, f_co2co2 = \
        _get_column_amounts(T_atm, ppm_co2, ppm_h2o, ppm_ch4, ppm_o3)

    log_p       = np.log10(p_mid / 101325.0)
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

        if turn_on_random_overlap:
            tau = diffusivity_factor * (
                tau_ch4[:, :, None, None] + tau_co2[:, :, None, None] +
                tau_h2o[:, None, :, None] + tau_o3[:, None, None, :] +
                tau_cia[:, None, None, None] + local_tau_lw[:, None, None, None]
            ).reshape(N, -1)
            w_g_combined = (
                ktable_co2['bands'][band_index]['w_g'][:, None, None] *
                ktable_h2o['bands'][band_index]['w_g'][None, :, None] *
                ktable_o3['bands'][band_index]['w_g'][None, None, :]
            ).reshape(-1)
        else:
            tau = diffusivity_factor * (
                tau_ch4 + tau_co2 + tau_h2o + tau_o3 + tau_cia[:, None] + local_tau_lw[:, None])
            w_g_combined = ktable_co2['bands'][band_index]['w_g']

        # 3. Solve the two-stream equations for this band
        B_atm  = planck_integration(band['v_min'], band['v_max'], T_atm)
        B_surf = planck_integration(band['v_min'], band['v_max'], T_surf)
        Fd_band, Fu_band = calc_lw_fluxes(N, np.exp(-tau), 1 - np.exp(-tau), B_atm, B_surf, w_g_combined)

        Fd_lw_total += Fd_band
        Fu_lw_total += Fu_band

    return Fd_lw_total, Fu_lw_total


def calc_sw_radiation(T_atm, Ts, day_of_year, ppm_co2, ppm_ch4, insol_factor, path_factor, mu,
                      local_reflectivity, local_albedo, T_atm_h2o=None):
    """
    Computes shortwave (solar) downward and upward flux profiles (W/m^2).

    Uses the same correlated-k framework as calc_lw_radiation, but accounts for
    scattering and reflection via the adding-doubling scheme in calc_sw_fluxes_reflect.
    The solar beam travels at an angle set by path_factor; reflected diffuse light
    uses the fixed diffusivity factor of 1.66.
    """
    if insol_factor <= 0:
        return np.zeros(N + 1), np.zeros(N + 1)

    T_atm_h2o = T_atm if T_atm_h2o is None else T_atm_h2o
    ppm_h2o = get_h2o_ppm(T_atm_h2o, p_mid)
    ppm_o3  = get_ozone_ppm(p_mid)

    # SW only uses H2O and O3 as absorbers (plus CIA); CO2 and CH4 are negligible in the solar
    _, u_h2o, _, u_o3, f_n2n2, f_n2o2, f_o2o2, f_co2co2 = \
        _get_column_amounts(T_atm, ppm_co2, ppm_h2o, ppm_ch4, ppm_o3)

    h2o_coords_sw = np.column_stack((np.log10(p_mid / 101325.0), T_atm,
                                     np.log10(np.maximum(ppm_h2o, 1e-10))))
    o3_sw_coords  = np.column_stack([T_atm])

    Fd_sw_total, Fu_sw_total = np.zeros(N + 1), np.zeros(N + 1)
    dist_factor = get_sun_distance_factor(day_of_year)

    for band_index, band in enumerate(ktable_h2o_sw['bands']):
        S_toa = solar_flux_toa(band['v_min'], band['v_max'], dist_factor) * insol_factor
        k_h2o = h2o_sw_interps[band_index](h2o_coords_sw)
        k_o3  = o3_sw_interps[band_index](o3_sw_coords)

        # Separate down/up paths: solar beam uses the geometric path_factor; diffuse uses 1.66
        tau_h2o_down = k_h2o * u_h2o[:, None] * path_factor
        tau_o3_down  = k_o3  * u_o3[:, None]  * path_factor
        tau_h2o_up   = k_h2o * u_h2o[:, None] * 1.66
        tau_o3_up    = k_o3  * u_o3[:, None]  * 1.66

        tau_cia_vertical = 0.0
        if include_CIA:
            tau_cia_vertical = (
                np.interp(T_atm, cia_t_grid, cia_sw['N2-N2'][band_index])   * f_n2n2 +
                np.interp(T_atm, cia_t_grid, cia_sw['N2-O2'][band_index])   * f_n2o2 +
                np.interp(T_atm, cia_t_grid, cia_sw['O2-O2'][band_index])   * f_o2o2 +
                np.interp(T_atm, cia_t_grid, cia_sw['CO2-CO2'][band_index]) * f_co2co2)

        if turn_on_random_overlap:
            tau_down = (tau_h2o_down[:, :, None] + tau_o3_down[:, None, :] +
                        (tau_cia_vertical * path_factor)[:, None, None]).reshape(N, -1)
            tau_up   = (tau_h2o_up[:, :, None]   + tau_o3_up[:, None, :] +
                        (tau_cia_vertical * 1.66)[:, None, None]).reshape(N, -1)
            w_g_combined_sw = (ktable_h2o_sw['bands'][band_index]['w_g'][:, None] *
                               ktable_o3_sw['bands'][band_index]['w_g'][None, :]).reshape(-1)
        else:
            tau_down = tau_h2o_down + tau_o3_down + (tau_cia_vertical * path_factor)[:, None]
            tau_up   = tau_h2o_up   + tau_o3_up   + (tau_cia_vertical * 1.66)[:, None]
            w_g_combined_sw = ktable_h2o_sw['bands'][band_index]['w_g']

        Fd_band, Fu_band = calc_sw_fluxes_reflect(
            N, np.exp(-tau_down), np.exp(-tau_up),
            local_reflectivity, S_toa, local_albedo, w_g_combined_sw)
        Fd_sw_total += Fd_band
        Fu_sw_total += Fu_band

    return Fd_sw_total, Fu_sw_total


# --- Numba JIT solvers for radiative flux integration ---
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
    t_eff_down = transmission_down * (1.0 - alpha_layer)
    t_eff_up   = transmission_up   * (1.0 - alpha_layer)

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


### --- 10. MAIN SIMULATION ENGINE --- ###
def run_model():
    """Integrates the model forward in time, updating atmospheric and surface temperatures."""
    dt = dt_days * 24 * 3600  # Time step (seconds)
    nsteps = int(years * 365.25 * 24 * 3600 / dt)
    switch_step = int(nsteps * 0.5) if perturb else nsteps + 1

    T_atm_curr, Ts_curr = T_initial.copy(), Ts_initial

    date_history = []
    hist_Ts        = np.zeros(nsteps)
    hist_Tatm      = np.zeros((nsteps, N))
    hist_imbalance = np.zeros(nsteps)
    hist_albedo    = np.zeros(nsteps)

    start_date      = datetime.datetime(start_year, start_month, start_day)
    initial_doy     = start_date.timetuple().tm_yday - 1
    seconds_in_year = 365.25 * 86400

    print(f"\nIntegration: {dt_days} day steps")
    if perturb:
        print(f"ERF: CO2 doubles and surface temp locks at step {switch_step}/{nsteps}")
    print(f"Running model from {start_date.strftime('%Y-%m-%d')} for {years} years...\n")

    T_atm_locked = None
    for n in range(nsteps):
        time_elapsed = n * dt
        doy = (initial_doy + time_elapsed / 86400) % 365.25
        date_history.append(start_date + datetime.timedelta(seconds=time_elapsed))

        current_co2  = initial_co2_ppm + (time_elapsed / seconds_in_year) * co2_ppm_rate
        ch4_profile  = get_ch4_ppm(p_mid, initial_ch4_ppm + (time_elapsed / seconds_in_year) * ch4_ppm_rate)

        if n == switch_step - 1:
            T_atm_locked = T_atm_curr.copy()

        lock_surface = False
        if perturb:
            current_co2  = initial_co2_ppm * 2 if n >= switch_step else initial_co2_ppm
            lock_surface = lock_surface_after_switch if n >= switch_step else False

        effective_day = doy if seasonal_variation else initial_doy

        curr_albedo = np.interp(effective_day, days_padded, albedo_arr) if include_albedo else 0.0
        curr_press  = [np.interp(effective_day, days_padded, arr) for arr in cld_press_arrs]
        curr_frac   = [np.interp(effective_day, days_padded, arr) for arr in cld_frac_arrs]
        curr_alb    = [np.interp(effective_day, days_padded, arr) for arr in cld_alb_arrs]
        curr_tau    = [np.interp(effective_day, days_padded, arr) for arr in cld_tau_arrs]

        loc_reflect, loc_tau_lw = get_cloud_profiles(curr_press, curr_frac, curr_alb, curr_tau, p_mid, dp)
        T_h2o_pass = T_atm_locked[:] if (fixed_water_feedback and T_atm_locked is not None) else None
        insol_fac, path_fac, mu_val = calculate_solar_parameters(effective_day, latitude, global_average)

        # --- Compute radiative fluxes ---
        Fd_lw, Fu_lw = calc_lw_radiation(T_atm_curr, Ts_curr, current_co2, ch4_profile,
                                          loc_tau_lw, T_atm_h2o=T_h2o_pass)
        Fd_sw, Fu_sw = calc_sw_radiation(T_atm_curr, Ts_curr, doy, current_co2, ch4_profile,
                                          insol_fac, path_fac, mu_val, loc_reflect, curr_albedo,
                                          T_atm_h2o=T_h2o_pass)

        # --- Heating rates: divergence of net flux in each layer (W/m^2) ---
        heating_lw = (Fu_lw - Fd_lw)[1:] - (Fu_lw - Fd_lw)[:-1]
        heating_sw = (Fu_sw - Fd_sw)[1:] - (Fu_sw - Fd_sw)[:-1]

        # --- Update temperatures ---
        T_atm_curr += ((heating_lw + heating_sw) / C_air) * dt
        if not lock_surface:
            Ts_curr += (((Fd_lw[-1] - Fu_lw[-1]) + (Fd_sw[-1] - Fu_sw[-1])) / C_s) * dt

        if convection:
            T_atm_curr, new_Ts = convective_adjustment_fast(
                T_atm_curr, Ts_curr, p_mid, p_surface, C_air, C_s,
                R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap)
            Ts_curr = Ts_curr if lock_surface else new_Ts

        # --- Store diagnostics ---
        hist_Ts[n]        = Ts_curr
        hist_Tatm[n, :]   = T_atm_curr
        hist_imbalance[n] = (Fd_sw[0] - Fu_sw[0]) - (Fu_lw[0] - Fd_lw[0])
        hist_albedo[n]    = Fu_sw[0] / Fd_sw[0] if Fd_sw[0] > 0 else 0

        if n % 10 == 0:
            status = f" ({'LOCKED' if lock_surface else 'FREE'})" if perturb else ""
            print(f"Step {n}/{nsteps}: Ts = {Ts_curr:.2f} K{status}, "
                  f"CO2 = {current_co2:.1f} ppm, Imbalance = {hist_imbalance[n]:.3f} W/m2")

    return date_history, hist_Ts, hist_Tatm, hist_albedo, hist_imbalance


### --- 11. PRE-RUN DIAGNOSTICS --- ###
print("\n--- INPUT PARAMETERS ---")
print(f"Time step:          {dt_days} days | Layers: {N} | Total time: {years} years")
print(f"Initial CO2:        {initial_co2_ppm} ppm | Annual increase: {co2_ppm_rate} ppm/year")
print(f"Initial CH4:        {initial_ch4_ppm} ppm | Annual increase: {ch4_ppm_rate} ppm/year")
print(f"Ozone peak:         {ozone_peak_ppm} ppm | Column mean: {np.sum(get_ozone_ppm(p_mid) * dp) / np.sum(dp):.3f} ppm")
print(f"Relative humidity:  {relative_humidity} (fraction)")

# Execute the simulation
dates, Ts, T_atm, albedo, imbalance = run_model()

### --- 12. POST-RUN SUMMARY & HANDOFF --- ###
print("\n" + "=" * 50)
print("            MODEL SIMULATION COMPLETE")
print("=" * 50)
print(f"\nRun time: {time.time() - start_time:.2f} seconds")

print("\nCompressing and saving model output...")
os.makedirs("results", exist_ok=True)
save_path = "results/global_model_output.npz"

np.savez_compressed(save_path,
                    dates      = np.asarray([d.strftime('%Y-%m-%d') for d in dates]),
                    Ts         = np.asarray(Ts),
                    T_atm      = np.asarray(T_atm),
                    albedo     = np.asarray(albedo),
                    imbalance  = np.asarray(imbalance),
                    p_mid      = np.asarray(p_mid),
                    p_int      = np.asarray(p_int),
                    dp         = np.asarray(dp))

print(f"\nSUCCESS: Simulation data saved to '{save_path}'")
print("-> Please run '2a_plot_global.py' to visualize the results.")