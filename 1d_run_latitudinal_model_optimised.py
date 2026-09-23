import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import time
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from numba import jit
import datetime
import matplotlib.dates as mdates
import os
import plotting
import pandas as pd
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded

start_time = time.time()

plotting.set_publication_style()

# <editor-fold desc="constants">
sigma = 5.670374419e-8
S0 = 1361
Q = S0 / 4
g = 9.81
Na = 6.02214076e23
m_air = 28.97e-3  # kg/mol
k_B = 1.380649e-23
c = 2.99792458e8
h = 6.62607015e-34

R_dry = 287.05
L_v = 2.501e6
R_vap = 461.5
cp_dry_air = 1004
cp_h2o = 4186.0
cp_h2o_vap = 1860

T_sun = 5778.0
sun_radius = 695700000
earth_sun_distance = 149597870700
earth_radius = 6371
model_height = 55
# </editor-fold>

hist_df = pd.read_csv("temperature data/Zonal_Timeseries_50Years_Kelvin.csv")

# <editor-fold desc="parameters">
show_plotting = True
save_data = True
data_save_name = 'test.npz'
save_figures_to_desktop = True
run_single_latitude = False
plotting_latitude = 0  # controls what we plot

start_year = 2010
start_month = 1
start_day = 1

years = 10
dt_days = 0.3
N = 25 # the number of layers

convection = True
linear_pressure = True
include_CIA = True
seasonal_variation = True
include_albedo = True
include_clouds = True
include_blanket = True

perturb = False
fixed_water_feedback = False
dynamic_ice_albedo = False
lock_surface_after_switch = False

diffusion = False
include_topography = False

heat_capacity_factor = 1
D_coeff = 0.6

initial_co2_ppm = 400
co2_ppm_rate = 0
initial_ch4_ppm = 1.8
ch4_ppm_rate = 0
ozone_peak_ppm = 7
relative_humidity = 0.77

diffusivity_factor = 1.66
# </editor-fold>

# <editor-fold desc="seasonal data">
print("Loading seasonal data...")
df_params = pd.read_csv('Making latitude package/latitude_params_monthly.csv')
df_params = df_params[df_params['Latitude'] != 'Global']
df_params['Latitude'] = df_params['Latitude'].astype(float)
unique_lats = np.sort(df_params['Latitude'].unique())
num_lats = len(unique_lats)

month_days = np.array([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345])
days_padded = np.concatenate([[-15], month_days, [380]])

def get_data_grid(variable_name):
    grid = np.zeros((14, num_lats))
    for i, m in enumerate(range(1, 13)):
        m_data = df_params[df_params['Month'] == m].sort_values('Latitude')
        grid[i + 1, :] = m_data[variable_name].values
    grid[0, :] = grid[12, :]
    grid[13, :] = grid[1, :]
    return grid

albedo_grid = get_data_grid('Albedo')
cld_albedo_grid = get_data_grid('Cloud_Albedo')
cld_frac_grid = get_data_grid('Cloud_Fraction')
cld_press_grid = get_data_grid('Cloud_Pressure')

static_data = df_params[df_params['Month'] == 1].sort_values('Latitude')
Cs_arr = static_data['Heat_Capacity'].values / heat_capacity_factor
rh_arr = static_data['Relative_Humidity'].values
ts_init_arr = static_data['Initial_Ts'].values
lats_grid = static_data['Latitude'].values

if include_topography:
    elev_arr = static_data['Elevation'].values
    print("Topography enabled!")
else:
    elev_arr = np.zeros(num_lats)
# </editor-fold>

# <editor-fold desc="Grid setup">
p_top = 50
p_tropopause = 20000.0
p_stratopause = 100

p_int_std = np.geomspace(p_top, 101325.0, N) if linear_pressure else np.linspace(p_top, 101325.0, N)
p_mid_std = 0.5 * (p_int_std[:-1] + p_int_std[1:])
N_layers = len(p_mid_std)

p_surface_arr = 101325.0 * np.exp(-g * elev_arr / (R_dry * 288.15))  # Barometric formula

p_int = np.zeros((num_lats, N))
for i in range(num_lats):
    if linear_pressure:
        p_int[i, :] = np.geomspace(p_top, p_surface_arr[i], N)
    else:
        p_int[i, :] = np.linspace(p_top, p_surface_arr[i], N)

p_mid = 0.5 * (p_int[:, :-1] + p_int[:, 1:])
dp = np.diff(p_int, axis=1)

C_air = cp_dry_air * dp / g
N = N_layers

print(f"Equator Surface Pressure: {p_surface_arr[num_lats // 2] / 100:.1f} hPa")
print(f"South Pole Surface Pressure: {p_surface_arr[0] / 100:.1f} hPa")
# </editor-fold>

# <editor-fold desc="temps">
n_meso = np.sum(p_mid_std < p_stratopause)
n_strat = np.sum((p_mid_std >= p_stratopause) & (p_mid_std < p_tropopause))
n_trop = N - n_meso - n_strat

T_stratopause = 270
T_tropopause = 215
T_surface_air = 290
T_top = 220

T_strat = np.linspace(T_stratopause, T_tropopause, n_strat)
T_trop = np.linspace(T_tropopause, T_surface_air, n_trop)
T_meso = np.linspace(T_top, T_stratopause, n_meso)
T_initial = np.concatenate((T_meso, T_strat, T_trop))
# </editor-fold>

# <editor-fold desc="data">
ktable_co2 = np.load("HITRAN/K tables gauss final/k_table_co2_lw_gauss_8.npz", allow_pickle=True)['k_table'].item()
ktable_h2o = np.load("HITRAN/K tables gauss final/k_table_h2o_lw_gauss_8.npz", allow_pickle=True)['k_table'].item()
ktable_h2o_sw = np.load("HITRAN/K tables gauss final/k_table_h2o_sw_gauss_8.npz", allow_pickle=True)['k_table'].item()
ktable_ch4 = np.load("HITRAN/K tables gauss final/k_table_ch4_gauss_8.npz", allow_pickle=True)['k_table'].item()
ktable_o3 = np.load("HITRAN/K tables gauss final/k_table_o3_lw_gauss_8.npz", allow_pickle=True)['k_table'].item()
ktable_o3_sw = np.load("HITRAN/K tables gauss final/k_table_o3_sw_new_gauss_8_ex.npz", allow_pickle=True)['k_table'].item()

cia_data = np.load("HITRAN/K tables final/cia_table_combined.npz", allow_pickle=True)
cia_t_grid = cia_data['T_grid']
cia_lw = cia_data['cia_lw'].item()
cia_sw = cia_data['cia_sw'].item()

print("K tables loaded")

co2_interps = []
for band in ktable_co2['bands']:
    interp = RegularGridInterpolator((np.log10(ktable_co2['P_grid']), ktable_co2['T_grid']), band['k_P_T_g'],
                                     bounds_error=False, fill_value=None)
    co2_interps.append(interp)

h2o_interps = []
for band in ktable_h2o['bands']:
    interp = RegularGridInterpolator(
        (np.log10(ktable_h2o['P_grid']), ktable_h2o['T_grid'], np.log10(ktable_h2o['ppm_grid'])), band['k_P_T_Q_g'],
        bounds_error=False, fill_value=None)
    h2o_interps.append(interp)

h2o_sw_interps = []
for band in ktable_h2o_sw['bands']:
    interp = RegularGridInterpolator(
        (np.log10(ktable_h2o_sw['P_grid']), ktable_h2o_sw['T_grid'], np.log10(ktable_h2o_sw['ppm_grid'])), band['k_P_T_Q_g'],
        bounds_error=False, fill_value=None)
    h2o_sw_interps.append(interp)

ch4_interps = []
for band in ktable_ch4['bands']:
    interp = RegularGridInterpolator((np.log10(ktable_ch4['P_grid']), ktable_ch4['T_grid']), band['k_P_T_g'],
                                     bounds_error=False, fill_value=None)
    ch4_interps.append(interp)

o3_interps = []
for band in ktable_o3['bands']:
    interp = RegularGridInterpolator((np.log10(ktable_o3['P_grid']), ktable_o3['T_grid']), band['k_P_T_g'],
                                     bounds_error=False, fill_value=None)
    o3_interps.append(interp)

o3_sw_interps = []
for band in ktable_o3_sw['bands']:
    interp = RegularGridInterpolator((ktable_o3_sw['T_grid'],), band['k_T_g'], bounds_error=False, fill_value=None)
    o3_sw_interps.append(interp)

print("tables interpolated")
# </editor-fold>

# <editor-fold desc="gas distributions">
def get_h2o_ppm(T, P_Pa, local_rh):
    T_c = T - 273.15
    exp_term = (18.678 - T_c / 234.5) * (T_c / (257.14 + T_c))
    p_sat = 0.61121 * np.exp(exp_term) * 1000.0

    rh0 = local_rh
    p0 = np.max(P_Pa)
    pressure_factor = np.clip((P_Pa / p0 - 0.02) / (1.0 - 0.02), 0.0, 1.0)
    rh = rh0 * pressure_factor

    q_local = (rh * p_sat / P_Pa) * 1e6
    q_entry = 3.5

    ch4_surface = 1.8
    ch4_top = 0.3

    p_mid_ch4 = 3000.0
    scale_ch4 = 0.75

    log_p = np.log(np.clip(P_Pa, 1.0, None))
    log_mid = np.log(p_mid_ch4)

    ch4_weight = 1.0 / (1.0 + np.exp(-(log_p - log_mid) / scale_ch4))
    current_ch4 = ch4_top + (ch4_surface - ch4_top) * ch4_weight

    q = q_entry + 2.0 * (ch4_surface - current_ch4)

    p_trop = 200 * 100
    w_trop = 1.0 / (1.0 + np.exp((log_p - np.log(p_trop)) / 0.2))

    final_ppm = w_trop * q + (1.0 - w_trop) * q_local
    final_ppm = np.maximum(final_ppm, q)

    if local_rh == 0:
        return np.zeros_like(T)
    return final_ppm

def get_ozone_ppm(P_Pa):
    base_ozone = 0.04
    peak_pressure = 8 * 100

    width_top = 2.5
    width_bottom = 2
    shape_factor = 2

    log_p = np.log(P_Pa)
    log_peak = np.log(peak_pressure)

    width = np.where(P_Pa < peak_pressure, width_top, width_bottom)
    stratospheric_ozone = ozone_peak_ppm * np.exp(-np.abs((log_p - log_peak) / width) ** shape_factor)

    return base_ozone + stratospheric_ozone

def get_ch4_ppm(P_Pa, current_surface_ppm):
    surface_val = current_surface_ppm
    strat_background = 0.2

    p_mid_val = 1000.0
    scale = 1

    log_p = np.log(P_Pa)
    log_mid = np.log(p_mid_val)

    weight = 1.0 / (1.0 + np.exp(-(log_p - log_mid) / scale))

    return strat_background + (surface_val - strat_background) * weight
# </editor-fold>

# <editor-fold desc="precomputed constants">
print("Pre-computing constant arrays...")

# --- Ozone profiles per latitude (constant — only depends on p_mid) ---
ozone_profiles = np.zeros((num_lats, N))
for i in range(num_lats):
    ozone_profiles[i, :] = get_ozone_ppm(p_mid[i, :])

# --- Column density per latitude (constant — only depends on dp) ---
u_total_all = (dp / (g * m_air)) * Na * 1e-4  # shape (num_lats, N)

# --- Ozone column density (constant) ---
u_o3_precomp = u_total_all * (ozone_profiles * 1e-6)

# --- log10(p/p0) per latitude (constant) ---
log_p_all = np.log10(p_mid / 101325.0)  # shape (num_lats, N)

# --- Gas fractions for CIA (constant) ---
f_n2 = 0.78
f_o2 = 0.21
cia_f_n2n2 = f_n2 * f_n2
cia_f_n2o2 = f_n2 * f_o2
cia_f_o2o2 = f_o2 * f_o2

# --- Rayleigh reflectivity base per latitude (constant) ---
rayleigh_base = np.zeros((num_lats, N))
for i in range(num_lats):
    rayleigh_base[i, :] = 0.1 * (dp[i, :] / np.sum(dp[i, :]))

# --- Pre-compute Gauss-Legendre weight products per band (constant) ---
num_bands_lw = len(ktable_co2['bands'])
num_bands_sw = len(ktable_h2o_sw['bands'])

w_g_lw = []  # combined weights for LW bands
for b in range(num_bands_lw):
    w_co2 = ktable_co2['bands'][b]['w_g']
    w_h2o = ktable_h2o['bands'][b]['w_g']
    w_o3 = ktable_o3['bands'][b]['w_g']
    w_tensor = w_co2[:, None, None] * w_h2o[None, :, None] * w_o3[None, None, :]
    w_g_lw.append(w_tensor.reshape(-1))

w_g_sw = []  # combined weights for SW bands
for b in range(num_bands_sw):
    w_h2o = ktable_h2o_sw['bands'][b]['w_g']
    w_o3 = ktable_o3_sw['bands'][b]['w_g']
    w_tensor = w_h2o[:, None] * w_o3[None, :]
    w_g_sw.append(w_tensor.reshape(-1))

# --- Pre-compute Planck band constants (constant) ---
planck_v_center_lw = np.zeros(num_bands_lw)
planck_width_lw = np.zeros(num_bands_lw)
for b in range(num_bands_lw):
    band = ktable_co2['bands'][b]
    planck_v_center_lw[b] = 0.5 * (band['v_min'] + band['v_max']) * 100
    planck_width_lw[b] = (band['v_max'] - band['v_min']) * 100

planck_v_center_sw = np.zeros(num_bands_sw)
planck_width_sw = np.zeros(num_bands_sw)
planck_v_min_sw = np.zeros(num_bands_sw)
planck_v_max_sw = np.zeros(num_bands_sw)
for b in range(num_bands_sw):
    band = ktable_h2o_sw['bands'][b]
    planck_v_center_sw[b] = 0.5 * (band['v_min'] + band['v_max']) * 100
    planck_width_sw[b] = (band['v_max'] - band['v_min']) * 100
    planck_v_min_sw[b] = band['v_min']
    planck_v_max_sw[b] = band['v_max']

# --- Solar normalization (constant) ---
_solar_norm_total = sigma * (T_sun ** 4) * ((sun_radius / earth_sun_distance) ** 2)
_solar_norm = S0 / _solar_norm_total
_solar_geom = (sun_radius / earth_sun_distance) ** 2

# Planck constants for fast evaluation
_planck_c1 = 2 * h * c ** 2
_planck_c2 = h * c / k_B

# --- Pre-compute CIA arrays as contiguous numpy arrays for faster interp ---
cia_lw_arrays = {}
for pair in ['N2-N2', 'N2-O2', 'O2-O2', 'CO2-CO2']:
    cia_lw_arrays[pair] = np.array(cia_lw[pair])  # shape (num_bands_lw, len(cia_t_grid))

cia_sw_arrays = {}
for pair in ['N2-N2', 'N2-O2', 'O2-O2', 'CO2-CO2']:
    cia_sw_arrays[pair] = np.array(cia_sw[pair])

# --- Pre-compute CH4 profile if rate is zero (constant) ---
if ch4_ppm_rate == 0:
    ch4_profiles_precomp = np.zeros((num_lats, N))
    for i in range(num_lats):
        ch4_profiles_precomp[i, :] = get_ch4_ppm(p_mid[i, :], initial_ch4_ppm)
    u_ch4_precomp = u_total_all * (ch4_profiles_precomp * 1e-6)

print("Pre-computation complete.")
# </editor-fold>

# <editor-fold desc="2D Physics">
def spherical_grid(lats):
    x = np.sin(np.radians(lats))
    num_lats = len(lats)
    x_bounds = np.zeros(num_lats + 1)
    x_bounds[0] = -1.0
    x_bounds[-1] = 1.0
    for i in range(1, num_lats):
        x_bounds[i] = 0.5 * (x[i - 1] + x[i])

    dx = np.diff(x_bounds)
    boundary_lengths = 1.0 - x_bounds ** 2
    return x, dx, x_bounds, boundary_lengths

def get_dynamic_albedo_vec(T_surf, baseline_albedo):
    """Vectorized version across all latitudes at once."""
    if not dynamic_ice_albedo:
        return baseline_albedo.copy()

    ice_free_alb = np.where(baseline_albedo < 0.3, baseline_albedo, 0.12)
    ice_alb = np.maximum(0.6, baseline_albedo)

    T_center = 266.575
    scale = 3.0

    albedo = 0.5 * (ice_alb + ice_free_alb) - 0.5 * (ice_alb - ice_free_alb) * np.tanh((T_surf - T_center) / scale)
    return albedo

def apply_diffusion(Ts, Cs_arr, x, dx, boundary_lengths, dt, p_surface):
    num_lats = len(Ts)

    kappa = R_dry / cp_dry_air
    f_theta = (101325.0 / p_surface) ** kappa

    # Build tridiagonal system using banded storage: ab[0]=upper, ab[1]=diag, ab[2]=lower
    ab = np.zeros((3, num_lats))

    for i in range(num_lats):
        if i < num_lats - 1:
            dist_north = x[i + 1] - x[i]
            coeff_north = (D_coeff * boundary_lengths[i + 1] * dt) / (Cs_arr[i] * dx[i] * dist_north)
        else:
            coeff_north = 0.0

        if i > 0:
            dist_south = x[i] - x[i - 1]
            coeff_south = (D_coeff * boundary_lengths[i] * dt) / (Cs_arr[i] * dx[i] * dist_south)
        else:
            coeff_south = 0.0

        ab[1, i] = 1.0 + coeff_north * f_theta[i] + coeff_south * f_theta[i]

        if i < num_lats - 1:
            ab[0, i + 1] = -coeff_north * f_theta[i + 1]  # upper diagonal
        if i > 0:
            ab[2, i - 1] = -coeff_south * f_theta[i - 1]  # lower diagonal

    Ts_new = solve_banded((1, 1), ab, Ts)
    return Ts_new

def cloud_profiles_batch(cld_press, cld_frac, cld_albedo, p_mid_local, dp_local, lat_idx):
    """Single-latitude cloud profile using precomputed Rayleigh base."""
    local_reflectivity = rayleigh_base[lat_idx, :].copy()
    local_tau_cloud = np.zeros(N)

    if not include_albedo:
        local_reflectivity[:] = 0

    if include_clouds:
        idx_cloud = (np.abs(p_mid_local - cld_press)).argmin()

        if include_albedo:
            local_reflectivity[idx_cloud] += (cld_frac * cld_albedo)
            local_reflectivity = np.minimum(local_reflectivity, 0.999)

        if include_blanket:
            tau_cloud_base = 1.25
            trans_cloud = (1 - cld_frac) + cld_frac * np.exp(-tau_cloud_base)
            local_tau_cloud[idx_cloud] = -np.log(np.maximum(trans_cloud, 1e-10))

    return local_reflectivity, local_tau_cloud
# </editor-fold>

# <editor-fold desc="Solar and geometric">
def planck_integration(v_min, v_max, T):
    v_center = 0.5 * (v_min + v_max) * 100
    width_m = (v_max - v_min) * 100
    B_nu = _planck_c1 * v_center ** 3 / (np.exp(_planck_c2 * v_center / T) - 1)
    return np.pi * B_nu * width_m

def planck_fast(v_center, width_m, T):
    """Fast Planck using precomputed band constants."""
    B_nu = _planck_c1 * v_center ** 3 / (np.exp(_planck_c2 * v_center / T) - 1)
    return np.pi * B_nu * width_m

def get_sun_distance_factor(day):
    rad_factor = np.pi / 180
    r = 1 - 0.01672 * np.cos(rad_factor * (360 / 365.25 * (day - 4)))
    return 1.0 / r ** 2

def solar_flux_toa_fast(v_center, width_m, distance_factor):
    """Fast solar flux using precomputed constants."""
    B_nu = _planck_c1 * v_center ** 3 / (np.exp(_planck_c2 * v_center / T_sun) - 1)
    flux_blackbody = np.pi * B_nu * width_m

    if not seasonal_variation:
        distance_factor = 1

    return flux_blackbody * _solar_geom * _solar_norm * distance_factor

def get_spherical_path(mu):
    if mu <= 0: return 0

    sin_sq = 1 - mu ** 2
    r_val = 6371
    a_val = 50

    term1 = (r_val + a_val) ** 2
    term2 = (r_val ** 2) * sin_sq
    term3 = r_val * mu

    L = np.sqrt(term1 - term2) - term3
    return L / a_val

def calculate_solar_parameters(day, lat):
    rad = np.pi / 180.0
    declination = 23.44 * np.sin(rad * 360 / 365 * (day - 81))
    delta_rad = np.radians(declination)
    lat_rad = np.radians(lat)

    tan_prod = -np.tan(lat_rad) * np.tan(delta_rad)

    if tan_prod >= 1.0:
        h0 = 0.0
    elif tan_prod <= -1.0:
        h0 = np.pi
    else:
        h0 = np.arccos(tan_prod)

    daily_avg_mu = (1.0 / np.pi) * (h0 * np.sin(lat_rad) * np.sin(delta_rad) +
                                    np.cos(lat_rad) * np.cos(delta_rad) * np.sin(h0))

    insol_factor = max(0.0, daily_avg_mu)

    elevation_noon = 90 - abs(lat - declination)
    if elevation_noon <= 0:
        mu_noon = 0
        path_factor = 0
    else:
        mu_noon = np.sin(np.radians(elevation_noon))
        path_factor = get_spherical_path(mu_noon)
    mu = mu_noon
    return insol_factor, path_factor, mu
# </editor-fold>

# <editor-fold desc="convection">
@jit(nopython=True)
def lnT_lnP_fast(T, P, R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap):
    P_vapour_sat = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))

    rs = (R_dry / R_vap) * P_vapour_sat / (P - P_vapour_sat)

    numerator = 1 + (L_v * rs) / (R_dry * T)
    denominator = 1 + ((cp_h2o_vap / cp_dry_air) + (L_v / (R_vap * T) - 1) * (L_v / (cp_dry_air * T)) * rs)

    return (R_dry / cp_dry_air) * (numerator / denominator)

@jit(nopython=True)
def convective_adjustment_fast(T_atm, T_surf, p_mid_local, p_surface_local, C_air_local, C_s,
                               R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap):
    N = len(T_atm)

    T_column = np.empty(N + 1)
    T_column[:N] = T_atm
    T_column[N] = T_surf

    P_column = np.empty(N + 1)
    P_column[:N] = p_mid_local
    P_column[N] = p_surface_local

    Cap_column = np.empty(N + 1)
    Cap_column[:N] = C_air_local
    Cap_column[N] = C_s

    max_loops = N ** 2
    for _ in range(max_loops):
        stable = True

        for i in range(N, 0, -1):
            T_lower = T_column[i]
            T_upper = T_column[i - 1]
            P_lower = P_column[i]
            P_upper = P_column[i - 1]

            d_lnP = np.log(P_lower) - np.log(P_upper)
            d_lnT = np.log(T_lower) - np.log(T_upper)
            actual_slope = d_lnT / d_lnP

            T_mid = 0.5 * (T_lower + T_upper)
            P_mid_val = 0.5 * (P_lower + P_upper)

            moist_slope = lnT_lnP_fast(T_mid, P_mid_val, R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap)
            critical_slope = max(moist_slope, 0.192)
            if actual_slope > critical_slope:
                stable = False

                C_up = Cap_column[i - 1]
                C_down = Cap_column[i]

                E_total = C_down * T_lower + C_up * T_upper

                target_ratio = (P_lower / P_upper) ** critical_slope

                T_up_new = E_total / (C_up + C_down * target_ratio)
                T_down_new = T_up_new * target_ratio

                T_column[i - 1] = T_up_new
                T_column[i] = T_down_new

        if stable:
            break

    return T_column[:-1], T_column[-1]
# </editor-fold>

# <editor-fold desc="radiation">
def calc_lw_radiation_batch(T_atm_all, Ts_all, ppm_co2, ppm_ch4_surface, rh_arr_local,
                            tau_cloud_all, T_atm_h2o_all=None):
    """Compute LW radiation for ALL latitudes at once, batching interpolation."""

    # --- Per-latitude gas profiles ---
    ppm_h2o_all = np.zeros((num_lats, N))
    for i in range(num_lats):
        T_water = T_atm_all[i, :] if T_atm_h2o_all is None else T_atm_h2o_all[i, :]
        ppm_h2o_all[i, :] = get_h2o_ppm(T_water, p_mid[i, :], rh_arr_local[i])

    u_h2o_all = u_total_all * (ppm_h2o_all * 1e-6)
    u_co2_all = u_total_all * (ppm_co2 * 1e-6)

    if ch4_ppm_rate == 0:
        u_ch4_all = u_ch4_precomp
    else:
        ch4_profiles = np.zeros((num_lats, N))
        for i in range(num_lats):
            ch4_profiles[i, :] = get_ch4_ppm(p_mid[i, :], ppm_ch4_surface)
        u_ch4_all = u_total_all * (ch4_profiles * 1e-6)

    # Flatten coords for batch interpolation: shape (num_lats * N, ndim)
    log_p_flat = log_p_all.reshape(-1)
    T_flat = T_atm_all.reshape(-1)
    log_h2o_flat = np.log10(np.maximum(ppm_h2o_all, 1e-10)).reshape(-1)

    co2_coords_batch = np.column_stack((log_p_flat, T_flat))
    h2o_coords_batch = np.column_stack((log_p_flat, T_flat, log_h2o_flat))
    ch4_coords_batch = co2_coords_batch  # same axes
    o3_coords_batch = co2_coords_batch

    # CIA factors (temperature-dependent part)
    n_air_all = (p_mid / (k_B * T_atm_all)) * 1e-6  # (num_lats, N)
    f_co2_cia = ppm_co2 * 1e-6

    factor_n2n2 = cia_f_n2n2 * (n_air_all * u_total_all)
    factor_n2o2 = cia_f_n2o2 * (n_air_all * u_total_all)
    factor_o2o2 = cia_f_o2o2 * (n_air_all * u_total_all)
    factor_co2co2 = (f_co2_cia * f_co2_cia) * (n_air_all * u_total_all)

    Fd_lw_total = np.zeros((num_lats, N + 1))
    Fu_lw_total = np.zeros((num_lats, N + 1))

    total_pts = num_lats * N

    for band_index in range(num_bands_lw):
        band = ktable_co2['bands'][band_index]

        # Batch interpolation — single call for all latitudes
        k_co2_flat = co2_interps[band_index](co2_coords_batch)
        k_h2o_flat = h2o_interps[band_index](h2o_coords_batch)
        k_ch4_flat = ch4_interps[band_index](ch4_coords_batch)
        k_o3_flat = o3_interps[band_index](o3_coords_batch)

        # Reshape back: (num_lats, N, n_g)
        n_g = k_co2_flat.shape[1] if k_co2_flat.ndim > 1 else k_co2_flat.shape[0] // total_pts
        k_co2 = k_co2_flat.reshape(num_lats, N, -1)
        k_h2o = k_h2o_flat.reshape(num_lats, N, -1)
        k_ch4 = k_ch4_flat.reshape(num_lats, N, -1)
        k_o3 = k_o3_flat.reshape(num_lats, N, -1)

        # CIA interpolation (batch across all latitudes)
        T_flat_1d = T_atm_all.reshape(-1)
        k_cia_n2n2 = np.interp(T_flat_1d, cia_t_grid, cia_lw_arrays['N2-N2'][band_index]).reshape(num_lats, N)
        k_cia_n2o2 = np.interp(T_flat_1d, cia_t_grid, cia_lw_arrays['N2-O2'][band_index]).reshape(num_lats, N)
        k_cia_o2o2 = np.interp(T_flat_1d, cia_t_grid, cia_lw_arrays['O2-O2'][band_index]).reshape(num_lats, N)
        k_cia_co2co2 = np.interp(T_flat_1d, cia_t_grid, cia_lw_arrays['CO2-CO2'][band_index]).reshape(num_lats, N)

        tau_cia = (k_cia_n2n2 * factor_n2n2 + k_cia_n2o2 * factor_n2o2 +
                   k_cia_o2o2 * factor_o2o2 + k_cia_co2co2 * factor_co2co2)

        if not include_CIA:
            tau_cia = tau_cia * 0.0

        # Planck emission
        v_c = planck_v_center_lw[band_index]
        w_m = planck_width_lw[band_index]
        B_atm_all = planck_fast(v_c, w_m, T_atm_all)  # (num_lats, N)
        B_surf_all = planck_fast(v_c, w_m, Ts_all)  # (num_lats,)

        w_g = w_g_lw[band_index]

        # Per-latitude flux propagation (must remain per-latitude for the sequential two-stream)
        for i in range(num_lats):
            tau_co2_i = k_co2[i] * u_co2_all[i, :, None]
            tau_h2o_i = k_h2o[i] * u_h2o_all[i, :, None]
            tau_ch4_i = k_ch4[i] * u_ch4_all[i, :, None]
            tau_o3_i = k_o3[i] * u_o3_precomp[i, :, None]

            tau_tensor = diffusivity_factor * (tau_ch4_i[:, :, None, None] + tau_co2_i[:, :, None, None] +
                                               tau_h2o_i[:, None, :, None] + tau_o3_i[:, None, None, :] +
                                               tau_cia[i, :, None, None, None] +
                                               tau_cloud_all[i, :, None, None, None])

            tau = tau_tensor.reshape(N, -1)
            transmission = np.exp(-tau)
            emissivity = 1 - transmission

            Fd_band, Fu_band = calc_lw_fluxes(N, transmission, emissivity, B_atm_all[i], B_surf_all[i], w_g)
            Fd_lw_total[i] += Fd_band
            Fu_lw_total[i] += Fu_band

    return Fd_lw_total, Fu_lw_total


def calc_sw_radiation_batch(T_atm_all, Ts_all, day_of_year, ppm_co2, ppm_ch4_surface,
                            insol_factors, path_factors, mu_vals, local_albedos,
                            rh_arr_local, layer_reflect_all, T_atm_h2o_all=None):
    """Compute SW radiation for ALL latitudes at once."""

    Fd_sw_total = np.zeros((num_lats, N + 1))
    Fu_sw_total = np.zeros((num_lats, N + 1))

    # Find which latitudes have sunlight
    lit_mask = insol_factors > 0
    if not np.any(lit_mask):
        return Fd_sw_total, Fu_sw_total

    lit_indices = np.where(lit_mask)[0]
    n_lit = len(lit_indices)

    # Gas profiles for lit latitudes only
    ppm_h2o_lit = np.zeros((n_lit, N))
    for ii, i in enumerate(lit_indices):
        T_water = T_atm_all[i, :] if T_atm_h2o_all is None else T_atm_h2o_all[i, :]
        ppm_h2o_lit[ii, :] = get_h2o_ppm(T_water, p_mid[i, :], rh_arr_local[i])

    u_total_lit = u_total_all[lit_indices]
    u_h2o_lit = u_total_lit * (ppm_h2o_lit * 1e-6)
    u_o3_lit = u_o3_precomp[lit_indices]

    log_p_lit = log_p_all[lit_indices]
    T_atm_lit = T_atm_all[lit_indices]

    # Batch coords
    log_p_flat = log_p_lit.reshape(-1)
    T_flat = T_atm_lit.reshape(-1)
    log_h2o_flat = np.log10(np.maximum(ppm_h2o_lit, 1e-10)).reshape(-1)

    h2o_coords_batch = np.column_stack((log_p_flat, T_flat, log_h2o_flat))
    o3_sw_coords_batch = T_flat.reshape(-1, 1)

    # CIA factors for lit latitudes
    n_air_lit = (p_mid[lit_indices] / (k_B * T_atm_lit)) * 1e-6
    f_co2_cia = ppm_co2 * 1e-6
    factor_n2n2 = cia_f_n2n2 * (n_air_lit * u_total_lit)
    factor_n2o2 = cia_f_n2o2 * (n_air_lit * u_total_lit)
    factor_o2o2 = cia_f_o2o2 * (n_air_lit * u_total_lit)
    factor_co2co2 = (f_co2_cia ** 2) * (n_air_lit * u_total_lit)

    dist_factor = get_sun_distance_factor(day_of_year)

    albedos_lit = local_albedos[lit_indices]
    if not include_albedo:
        albedos_lit = np.zeros(n_lit)

    for band_index in range(num_bands_sw):
        S_toa_base = solar_flux_toa_fast(planck_v_center_sw[band_index], planck_width_sw[band_index], dist_factor)

        # Batch interpolation
        k_h2o_flat = h2o_sw_interps[band_index](h2o_coords_batch)
        k_o3_flat = o3_sw_interps[band_index](o3_sw_coords_batch)

        k_h2o = k_h2o_flat.reshape(n_lit, N, -1)
        k_o3 = k_o3_flat.reshape(n_lit, N, -1)

        # CIA batch
        T_flat_1d = T_atm_lit.reshape(-1)
        k_cia_n2n2 = np.interp(T_flat_1d, cia_t_grid, cia_sw_arrays['N2-N2'][band_index]).reshape(n_lit, N)
        k_cia_n2o2 = np.interp(T_flat_1d, cia_t_grid, cia_sw_arrays['N2-O2'][band_index]).reshape(n_lit, N)
        k_cia_o2o2 = np.interp(T_flat_1d, cia_t_grid, cia_sw_arrays['O2-O2'][band_index]).reshape(n_lit, N)
        k_cia_co2co2 = np.interp(T_flat_1d, cia_t_grid, cia_sw_arrays['CO2-CO2'][band_index]).reshape(n_lit, N)

        tau_cia_vert = (k_cia_n2n2 * factor_n2n2 + k_cia_n2o2 * factor_n2o2 +
                        k_cia_o2o2 * factor_o2o2 + k_cia_co2co2 * factor_co2co2)
        if not include_CIA:
            tau_cia_vert *= 0.0

        w_g = w_g_sw[band_index]

        for ii, i in enumerate(lit_indices):
            S_toa = S_toa_base * insol_factors[i]
            pf = path_factors[i]

            tau_h2o_down = (k_h2o[ii] * u_h2o_lit[ii, :, None]) * pf
            tau_o3_down = (k_o3[ii] * u_o3_lit[ii, :, None]) * pf
            tau_h2o_up = (k_h2o[ii] * u_h2o_lit[ii, :, None]) * 1.66
            tau_o3_up = (k_o3[ii] * u_o3_lit[ii, :, None]) * 1.66

            tau_cia_down = tau_cia_vert[ii, :] * pf
            tau_cia_up = tau_cia_vert[ii, :] * 1.66

            tau_matrix_down = tau_h2o_down[:, :, None] + tau_o3_down[:, None, :] + tau_cia_down[:, None, None]
            tau_down = tau_matrix_down.reshape(N, -1)
            transmission_down = np.exp(-tau_down)

            tau_matrix_up = tau_h2o_up[:, :, None] + tau_o3_up[:, None, :] + tau_cia_up[:, None, None]
            tau_up = tau_matrix_up.reshape(N, -1)
            transmission_up = np.exp(-tau_up)

            Fd_band, Fu_band = calc_sw_fluxes_reflect(N, transmission_down, transmission_up,
                                                      layer_reflect_all[i], S_toa, albedos_lit[ii], w_g)
            Fd_sw_total[i] += Fd_band
            Fu_sw_total[i] += Fu_band

    return Fd_sw_total, Fu_sw_total
# </editor-fold>

# <editor-fold desc="fluxes">
@jit(nopython=True)
def calc_lw_fluxes(N, transmission, emissivity, B_atm, B_surf, w_g):
    Fd_g_possibility = np.zeros(transmission.shape[1])
    Fd_band = np.zeros(N + 1)

    for i in range(N):
        Fd_g_possibility = transmission[i, :] * Fd_g_possibility + emissivity[i, :] * B_atm[i]
        Fd_band[i + 1] = np.sum(Fd_g_possibility * w_g)

    Fu_g_possibility = np.full(transmission.shape[1], B_surf)
    Fu_band = np.zeros(N + 1)
    Fu_band[N] = B_surf

    for i in range(N - 1, -1, -1):
        Fu_g_possibility = transmission[i, :] * Fu_g_possibility + emissivity[i, :] * B_atm[i]
        Fu_band[i] = np.sum(Fu_g_possibility * w_g)

    return Fd_band, Fu_band

@jit(nopython=True)
def calc_sw_fluxes_reflect(N, transmission_down, transmission_up, reflectivity, S_toa, albedo_surface, w_g):
    n_g = transmission_down.shape[1]

    alpha_layer = reflectivity.reshape((N, 1))

    t_eff_down = transmission_down * (1.0 - alpha_layer)
    t_eff_up = transmission_up * (1.0 - alpha_layer)

    alpha_combined = np.zeros((N + 1, n_g))
    alpha_combined[N, :] = albedo_surface

    for i in range(N - 1, -1, -1):
        alpha_i = alpha_layer[i, 0]
        alpha_combined[i, :] = alpha_i + ((alpha_combined[i + 1, :] * t_eff_down[i, :] * t_eff_up[i, :])
                                          / (1.0 - (alpha_i * alpha_combined[i + 1, :])))

    Fd = np.zeros((N + 1, n_g))
    Fu = np.zeros((N + 1, n_g))

    Fd[0, :] = S_toa

    for i in range(N):
        alpha_i = alpha_layer[i, 0]
        Fd[i + 1, :] = Fd[i, :] * t_eff_down[i, :] / (1.0 - (alpha_i * alpha_combined[i + 1, :]))

    Fu = Fd * alpha_combined

    Fd_band = np.zeros(N + 1)
    Fu_band = np.zeros(N + 1)
    for i in range(N + 1):
        Fd_band[i] = np.sum(Fd[i, :] * w_g)
        Fu_band[i] = np.sum(Fu[i, :] * w_g)

    return Fd_band, Fu_band
# </editor-fold>

def run_model():
    dt = dt_days * 24 * 3600
    nsteps = int(years * 365.25 * 24 * 3600 / dt)
    steps_per_year = int(365.25 * 24 * 3600 / dt)

    switch_step = int(nsteps * 0.5) if perturb else nsteps + 1

    lats = lats_grid
    x_grid, dx_grid, x_bounds, bound_lens = spherical_grid(lats)

    T_atm_curr = np.zeros((num_lats, N))
    for i in range(num_lats):
        T_atm_curr[i, :] = T_initial.copy()
    Ts_curr = ts_init_arr.astype(float)

    date_history = []
    hist_Ts = np.zeros((nsteps, num_lats))
    hist_Tatm = np.zeros((nsteps, num_lats, N))
    hist_imbalance = np.zeros((nsteps, num_lats))
    hist_albedo = np.zeros((nsteps, num_lats))

    hist_Fd_sw = np.zeros((nsteps, num_lats, N + 1))
    hist_Fu_sw = np.zeros((nsteps, num_lats, N + 1))
    hist_Fd_lw = np.zeros((nsteps, num_lats, N + 1))
    hist_Fu_lw = np.zeros((nsteps, num_lats, N + 1))
    hist_heating_sw = np.zeros((nsteps, num_lats, N))
    hist_heating_lw = np.zeros((nsteps, num_lats, N))

    Ts_climatology = np.zeros((steps_per_year, num_lats))
    T_atm_climatology = np.zeros((steps_per_year, num_lats, N))

    start_date = datetime.datetime(start_year, start_month, start_day)
    initial_doy = start_date.timetuple().tm_yday - 1
    seconds_in_year = 365.25 * 86400

    # Pre-compute date offsets to avoid datetime construction in loop
    _dt_seconds = dt

    print(f"Integration: {dt_days} day steps")
    print(f"Running model from {start_date.strftime('%Y-%m-%d')} for {years} years...")

    # Pre-allocate reusable arrays
    insol_factors = np.zeros(num_lats)
    path_factors = np.zeros(num_lats)
    mu_vals = np.zeros(num_lats)
    layer_reflect_all = np.zeros((num_lats, N))
    tau_cloud_all = np.zeros((num_lats, N))

    for n in range(nsteps):
        time_elapsed = n * dt
        step_in_year = n % steps_per_year
        doy = (initial_doy + time_elapsed / 86400) % 365.25
        current_date = start_date + datetime.timedelta(seconds=time_elapsed)
        date_history.append(current_date)

        # --- GHOST YEAR RECORDING ---
        if perturb and (switch_step - steps_per_year <= n < switch_step):
            Ts_climatology[step_in_year, :] = Ts_curr.copy()
            if fixed_water_feedback:
                T_atm_climatology[step_in_year, :, :] = T_atm_curr.copy()

        if perturb and n >= switch_step:
            current_co2 = initial_co2_ppm * 2
            current_ch4_surface = initial_ch4_ppm
            lock_surface = lock_surface_after_switch
        else:
            current_co2 = initial_co2_ppm + (time_elapsed / seconds_in_year) * co2_ppm_rate
            current_ch4_surface = initial_ch4_ppm + (time_elapsed / seconds_in_year) * ch4_ppm_rate
            lock_surface = False

        effective_day = doy if seasonal_variation else initial_doy

        # --- Vectorized seasonal interpolation ---
        curr_albedos = np.array([np.interp(effective_day, days_padded, albedo_grid[:, i]) for i in range(num_lats)])
        curr_cld_albedos = np.array([np.interp(effective_day, days_padded, cld_albedo_grid[:, i]) for i in range(num_lats)])
        curr_cld_fracs = np.array([np.interp(effective_day, days_padded, cld_frac_grid[:, i]) for i in range(num_lats)])
        curr_cld_press = np.array([np.interp(effective_day, days_padded, cld_press_grid[:, i]) for i in range(num_lats)])

        # --- Vectorized dynamic albedo ---
        loc_albedos = get_dynamic_albedo_vec(Ts_curr, curr_albedos)

        # --- Solar parameters and cloud profiles per latitude ---
        for i in range(num_lats):
            insol_factors[i], path_factors[i], mu_vals[i] = calculate_solar_parameters(effective_day, lats[i])
            layer_reflect_all[i], tau_cloud_all[i] = cloud_profiles_batch(
                curr_cld_press[i], curr_cld_fracs[i], curr_cld_albedos[i], p_mid[i, :], dp[i, :], i)

        # --- Water vapor locking ---
        if fixed_water_feedback and (perturb and n >= switch_step):
            T_h2o_all = T_atm_climatology[step_in_year, :, :]
        else:
            T_h2o_all = None

        # --- BATCHED RADIATION across all latitudes ---
        Fd_lw_all, Fu_lw_all = calc_lw_radiation_batch(
            T_atm_curr, Ts_curr, current_co2, current_ch4_surface, rh_arr,
            tau_cloud_all, T_atm_h2o_all=T_h2o_all)

        Fd_sw_all, Fu_sw_all = calc_sw_radiation_batch(
            T_atm_curr, Ts_curr, doy, current_co2, current_ch4_surface,
            insol_factors, path_factors, mu_vals, loc_albedos,
            rh_arr, layer_reflect_all, T_atm_h2o_all=T_h2o_all)

        # --- Compute heating rates for all latitudes at once ---
        net_lw = Fu_lw_all - Fd_lw_all  # (num_lats, N+1)
        net_sw = Fu_sw_all - Fd_sw_all

        heating_lw = net_lw[:, 1:] - net_lw[:, :-1]  # (num_lats, N)
        heating_sw = net_sw[:, 1:] - net_sw[:, :-1]

        hist_Fd_sw[n] = Fd_sw_all
        hist_Fu_sw[n] = Fu_sw_all
        hist_Fd_lw[n] = Fd_lw_all
        hist_Fu_lw[n] = Fu_lw_all
        hist_heating_sw[n] = heating_sw
        hist_heating_lw[n] = heating_lw

        rate_atm_all = (heating_lw + heating_sw) / C_air  # (num_lats, N)

        T_atm_curr += rate_atm_all * dt

        if not lock_surface:
            net_s = (Fd_lw_all[:, -1] - Fu_lw_all[:, -1]) + (Fd_sw_all[:, -1] - Fu_sw_all[:, -1])
            rate_s = net_s / Cs_arr
            Ts_curr += rate_s * dt
        else:
            Ts_curr = Ts_climatology[step_in_year, :].copy()

        if diffusion and not lock_surface:
            Ts_curr = apply_diffusion(Ts_curr, Cs_arr, x_grid, dx_grid, bound_lens, dt, p_surface_arr)

        for i in range(num_lats):
            if convection:
                T_atm_curr[i, :], Ts_curr[i] = convective_adjustment_fast(
                    T_atm_curr[i, :], Ts_curr[i], p_mid[i, :], p_surface_arr[i], C_air[i, :], Cs_arr[i],
                    R_dry, R_vap, L_v, cp_dry_air, cp_h2o_vap)

            hist_Ts[n, i] = Ts_curr[i]
            hist_Tatm[n, i, :] = T_atm_curr[i, :]
            hist_imbalance[n, i] = (Fd_sw_all[i, 0] - Fu_sw_all[i, 0]) - (Fu_lw_all[i, 0] - Fd_lw_all[i, 0])
            hist_albedo[n, i] = Fu_sw_all[i, 0] / Fd_sw_all[i, 0] if Fd_sw_all[i, 0] > 0 else 0

        if n % 10 == 0:
            status = f" ({'LOCKED' if lock_surface else 'FREE'})" if perturb else ""
            instant_avg_T = np.average(Ts_curr, weights=dx_grid)

            start_idx = max(0, n + 1 - steps_per_year)
            recent_Ts_history = hist_Ts[start_idx:n + 1, :]
            time_averaged_lats = np.mean(recent_Ts_history, axis=0)
            rolling_avg_T = np.average(time_averaged_lats, weights=dx_grid)

            print(
                f"Step {n}/{nsteps}: Instant Ts = {instant_avg_T:.2f} K{status}, "
                f"CO2 = {current_co2:.1f} ppm, "
                f"1-Yr Rolling Ts = {rolling_avg_T:.2f} K"
            )

    return (date_history, lats, rh_arr, hist_Ts, hist_Tatm, hist_albedo, hist_imbalance,
            hist_Fd_sw, hist_Fu_sw, hist_Fd_lw, hist_Fu_lw, hist_heating_sw, hist_heating_lw)

(dates, lats_out, rh_out, Ts_history, Tatm_history, albedo_history,
 imbalance_history, Fd_sw_hist, Fu_sw_hist, Fd_lw_hist, Fu_lw_hist, heat_sw_hist, heat_lw_hist) = run_model()

# <editor-fold desc="text">
separator = "-" * 50
print("\n" + "=" * 50)
print("            MODEL SIMULATION PARAMETERS")

print(f" SIMULATION SETTINGS")
print(f"{'Total Duration:':<25} {years} years")
print(f"{'Time Step (dt):':<25} {dt_days} days")
print(f"{'Vertical Layers (N):':<25} {N}")

print(f"\n MODEL PHYSICS (FLAGS)")
physics_flags = {
    "Topography": include_topography, "Convection": convection, "CIA": include_CIA,
    "Albedo": include_albedo, "Clouds": include_clouds, "Blanket": include_blanket,
    "Seasonal": seasonal_variation, "Dyn Ice": dynamic_ice_albedo, "Diffusion": diffusion
}

for name, status in physics_flags.items():
    state = "ON" if status else "OFF"
    print(f"{name:<15}: {state}")

print("=" * 50 + "\n")
end_time = time.time()
print(f"run time: {end_time - start_time:.2f} seconds")
# </editor-fold>

# <editor-fold desc="data capture">
if save_data == True:
    print("\nCompressing and saving the database...")

    date_strings = [d.strftime('%Y-%m-%d') for d in dates]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, data_save_name)

    run_params = {
        'start_year': start_year,
        'start_month': start_month,
        'start_day': start_day,
        'years': years,
        'dt_days': dt_days,
        'N': N,
        'convection': convection,
        'linear_pressure': linear_pressure,
        'include_CIA': include_CIA,
        'seasonal_variation': seasonal_variation,
        'include_albedo': include_albedo,
        'include_clouds': include_clouds,
        'include_blanket': include_blanket,
        'perturb': perturb,
        'fixed_water_feedback': fixed_water_feedback,
        'dynamic_ice_albedo': dynamic_ice_albedo,
        'lock_surface_after_switch': lock_surface_after_switch,
        'diffusion': diffusion,
        'include_topography': include_topography,
        'heat_capacity_factor': heat_capacity_factor,
        'D_coeff': D_coeff,
        'initial_co2_ppm': initial_co2_ppm,
        'co2_ppm_rate': co2_ppm_rate,
        'initial_ch4_ppm': initial_ch4_ppm,
        'ch4_ppm_rate': ch4_ppm_rate,
        'ozone_peak_ppm': ozone_peak_ppm,
        'relative_humidity': relative_humidity,
        'diffusivity_factor': diffusivity_factor
    }

    np.savez_compressed(save_path,
                        dates=date_strings,
                        lats=np.asarray(lats_out),
                        p_mid=np.asarray(p_mid),
                        p_int=np.asarray(p_int),
                        Ts=np.asarray(Ts_history),
                        albedo=np.asarray(albedo_history),
                        imbalance=np.asarray(imbalance_history),
                        Tatm=np.asarray(Tatm_history),
                        Fd_sw=np.asarray(Fd_sw_hist),
                        Fu_sw=np.asarray(Fu_sw_hist),
                        Fd_lw=np.asarray(Fd_lw_hist),
                        Fu_lw=np.asarray(Fu_lw_hist),
                        heating_sw=np.asarray(heat_sw_hist),
                        heating_lw=np.asarray(heat_lw_hist),
                        params=run_params
                        )
    print("\n" + "=" * 50)
    print(f"data saved to:")
    print(f" -> {save_path}")
    print("=" * 50 + "\n")
# </editor-fold>

if show_plotting == True:
    # <editor-fold desc="plotting help">
    weights = np.cos(np.radians(lats_out))
    weights /= np.sum(weights)

    global_Ts = np.average(Ts_history, axis=1, weights=weights)
    global_imbalance = np.average(imbalance_history, axis=1, weights=weights)
    global_albedo = np.average(albedo_history, axis=1, weights=weights)

    plot_idx = np.argmin(np.abs(lats_out - plotting_latitude))
    plot_lat = lats_out[plot_idx]
    plot_rh = rh_out[plot_idx]

    Ts = global_Ts
    imbalance = global_imbalance
    albedo = global_albedo
    T_atm = Tatm_history[:, plot_idx, :]

    time_years = np.linspace(0, years, len(dates))
    final_doy = (start_day + years * 365.25) % 365.25

    plot_albedo_val = np.interp(final_doy, days_padded, albedo_grid[:, plot_idx])
    plot_cld_alb = np.interp(final_doy, days_padded, cld_albedo_grid[:, plot_idx])
    plot_cld_frac = np.interp(final_doy, days_padded, cld_frac_grid[:, plot_idx])
    plot_cld_press = np.interp(final_doy, days_padded, cld_press_grid[:, plot_idx])

    local_p_mid = p_mid[plot_idx, :]
    local_dp = dp[plot_idx, :]

    plot_layer_reflect, plot_tau_cloud = cloud_profiles_batch(plot_cld_press, plot_cld_frac, plot_cld_alb, local_p_mid,
                                                        local_dp, plot_idx)

    T_final = Tatm_history[-1, plot_idx, :]
    Ts_final = Ts_history[-1, plot_idx]

    water_profile = get_h2o_ppm(T_final, local_p_mid, plot_rh)
    ch4_final_profile = get_ch4_ppm(local_p_mid, initial_ch4_ppm)

    p_tropopause = 200
    p_stratopause = 1

    table_data = np.array([
        [-1000, 294.65, 113900], [0, 288.15, 101325], [1000, 281.65, 89880],
        [2000, 275.15, 79500], [3000, 268.65, 70120], [4000, 262.15, 61660],
        [5000, 255.65, 54050], [6000, 249.15, 47220], [7000, 242.65, 41110],
        [8000, 236.15, 35650], [9000, 229.65, 30800], [10000, 223.15, 26500],
        [11000, 216.65, 22700], [12000, 216.65, 19400], [15000, 216.65, 12110],
        [20000, 216.65, 5529], [25000, 221.55, 2549], [30000, 226.50, 1197],
        [32000, 228.65, 889], [35000, 236.51, 575], [40000, 250.35, 287],
        [45000, 264.16, 149], [47000, 270.65, 116], [50000, 270.65, 79.78],
        [60000, 247.02, 21.96], [70000, 219.58, 5.22], [80000, 198.64, 1.05]
    ])

    alt_table_m = table_data[:, 0]
    temp_table_k = table_data[:, 1]
    pres_table_pa = table_data[:, 2]

    interp_temp_from_alt = interp1d(alt_table_m, temp_table_k, kind='linear', fill_value="extrapolate")
    interp_logP_from_alt = interp1d(alt_table_m, np.log(pres_table_pa), kind='linear', fill_value="extrapolate")
    interp_alt_from_logP = interp1d(np.log(pres_table_pa), alt_table_m, kind='linear', fill_value="extrapolate")


    def get_pressure_at_alt(alt_m):
        return np.exp(interp_logP_from_alt(alt_m))


    def get_alt_at_pressure(pressure_pa):
        p_safe = np.maximum(pressure_pa, 1e-5)
        return interp_alt_from_logP(np.log(p_safe))

    smooth_alt_m = np.linspace(0, 80000, 500)
    smooth_temp = interp_temp_from_alt(smooth_alt_m)
    smooth_pres = get_pressure_at_alt(smooth_alt_m)
    # </editor-fold>

    # <editor-fold desc="final temperatures by latitude">
    fig, ax = plt.subplots(figsize=(8, 5))
    final_lats_Ts = Ts_history[-1, :]
    sort_idx = np.argsort(lats_out)
    sorted_lats = lats_out[sort_idx]
    sorted_Ts = final_lats_Ts[sort_idx]

    ax.plot(sorted_lats, sorted_Ts, color='blue', marker='o', linestyle='-',
            linewidth=2, markersize=5, label='Final $T_s$')

    ax.set_title("Surface Temperature by Latitude")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Surface Temperature (K)")
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlim([-90, 90])
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.legend()

    plt.tight_layout()

    if save_figures_to_desktop:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path = os.path.join(desktop_path, "temperature_by_latitude")
        plotting.save_for_paper(file_path)
    # </editor-fold>

    # <editor-fold desc="temperature profile">
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), sharey=True,
                                        gridspec_kw={'width_ratios': [6, 1, 1]})

    ax1.plot(smooth_temp, smooth_pres / 100, 'black', linewidth=0.8, label='US Standard Atmosphere')
    ax1.plot(T_final, local_p_mid / 100, 'r-o', linewidth=1, label=f'Final T ({plot_lat}°)')
    ax1.scatter(Ts_final, p_surface_arr[plot_idx] / 100, color='blue', marker='x', s=50, label='Surface', zorder=10)

    x_pos = ax1.get_xlim()[1] - 2

    ax1.axhline(y=p_tropopause, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.text(x_pos, p_tropopause, 'Tropopause', color='gray', fontweight='bold', fontsize=9, va='bottom', ha='right')

    ax1.axhline(y=p_stratopause, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.text(x_pos, p_stratopause, 'Stratopause', color='gray', fontweight='bold', fontsize=9, va='bottom', ha='right')

    if include_clouds:
        ax1.axhline(y=plot_cld_press / 100, color='gray', linestyle='-', linewidth=1, alpha=0.7)
        ax1.text(x_pos, plot_cld_press / 100, 'Cloud Layer', color='gray', fontweight='bold', fontsize=9, va='bottom',
                 ha='right')

    ax1.invert_yaxis()
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_ylabel("Pressure (hPa)")
    ax1.set_xlabel("Temperature (K)")
    ax1.legend(loc='lower left')
    ax1.set_ylim(1050, 0.1)

    ax2.plot(get_ozone_ppm(local_p_mid), local_p_mid / 100, '-', linewidth=1, label="O$_3$", color='red')
    ax2.plot(ch4_final_profile, local_p_mid / 100, '-', linewidth=1, label="CH$_4$", color='green')
    ax2.set_xlabel("")

    ax3.plot(water_profile, local_p_mid / 100, '-', linewidth=1, label='H$_2$O', color='blue')
    ax3.plot(np.full((len(local_p_mid),), initial_co2_ppm), local_p_mid / 100, '-', linewidth=1, label='CO$_2$',
             color='black')
    ax3.set_xscale('log')
    ax3.set_xlabel("")
    ax3.legend()

    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    center_x = (pos2.x0 + pos3.x1) / 2
    fig.text(center_x, 0.04, "         Concentration (ppm)", ha='center')
    ax2.legend()

    ax_height = ax3.twinx()
    p_bottom, p_top_plot = ax1.get_ylim()

    h_bottom_km = get_alt_at_pressure(p_bottom * 100) / 1000
    h_top_km = get_alt_at_pressure(p_top_plot * 100) / 1000.0

    ax_height.set_ylim(h_bottom_km, h_top_km)
    ax_height.set_ylabel("Altitude (km)")
    ax_height.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes = [ax1, ax2, ax3]
    labels = ['(a)', '(b)', '(c)']

    for ax, label in zip(axes, labels):
        ax.text(0.0, 1.02, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', ha='left')

    plt.tight_layout()
    if save_figures_to_desktop:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path = os.path.join(desktop_path, "temperature profile")
        plotting.save_for_paper(file_path)
    # </editor-fold>

    # <editor-fold desc="flux plotting">
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(dates, albedo, color='blue', label='Earth Albedo')
    ax1.xaxis.set_visible(False)
    ax1.set_ylabel("Albedo fraction")
    ax1.set_xlabel("Year")
    ax1.set_title("Albedo")

    i_fac, p_fac, m_val = calculate_solar_parameters(final_doy, plotting_latitude)

    # For final flux plot, use original single-latitude function
    def calc_sw_radiation_single(T_atm, Ts, day_of_year, ppm_co2, ppm_ch4, insol_factor, path_factor, mu, local_albedo,
                          local_rh, local_layer_reflectivity, p_mid_local, dp_local, T_atm_h2o=None):
        if insol_factor <= 0:
            return np.zeros(N + 1), np.zeros(N + 1)
        T_water = T_atm if T_atm_h2o is None else T_atm_h2o
        ppm_h2o = get_h2o_ppm(T_water, p_mid_local, local_rh)
        ppm_o3 = get_ozone_ppm(p_mid_local)
        u_total = (dp_local / (g * m_air)) * Na * 1e-4
        u_h2o_all = u_total * (ppm_h2o * 1e-6)
        u_o3_all = u_total * (ppm_o3 * 1e-6)
        n_air = (p_mid_local / (k_B * T_atm)) * 1e-6
        f_co2 = ppm_co2 * 1e-6
        factor_n2n2 = (f_n2 * f_n2) * (n_air * u_total)
        factor_n2o2 = (f_n2 * f_o2) * (n_air * u_total)
        factor_o2o2 = (f_o2 * f_o2) * (n_air * u_total)
        factor_co2co2 = (f_co2 * f_co2) * (n_air * u_total)
        log_p = np.log10(p_mid_local / 101325.0)
        log_h2o_ppm = np.log10(np.maximum(ppm_h2o, 1e-10))
        h2o_coords_sw = np.column_stack((log_p, T_atm, log_h2o_ppm))
        o3_sw_coords = np.column_stack([T_atm])
        Fd_sw_total = np.zeros(N + 1)
        Fu_sw_total = np.zeros(N + 1)
        dist_factor = get_sun_distance_factor(day_of_year)
        if not include_albedo: local_albedo = 0.0
        for band_index in range(num_bands_sw):
            band = ktable_h2o_sw['bands'][band_index]
            S_raw = solar_flux_toa_fast(planck_v_center_sw[band_index], planck_width_sw[band_index], dist_factor)
            S_toa = S_raw * insol_factor
            k_h2o = h2o_sw_interps[band_index](h2o_coords_sw)
            k_o3 = o3_sw_interps[band_index](o3_sw_coords)
            tau_h2o_down = (k_h2o * u_h2o_all[:, None]) * path_factor
            tau_o3_down = (k_o3 * u_o3_all[:, None]) * path_factor
            tau_h2o_up = (k_h2o * u_h2o_all[:, None]) * 1.66
            tau_o3_up = (k_o3 * u_o3_all[:, None]) * 1.66
            k_cia_n2n2 = np.interp(T_atm, cia_t_grid, cia_sw['N2-N2'][band_index])
            k_cia_n2o2 = np.interp(T_atm, cia_t_grid, cia_sw['N2-O2'][band_index])
            k_cia_o2o2 = np.interp(T_atm, cia_t_grid, cia_sw['O2-O2'][band_index])
            k_cia_co2co2 = np.interp(T_atm, cia_t_grid, cia_sw['CO2-CO2'][band_index])
            tau_cia_vertical = (k_cia_n2n2 * factor_n2n2 + k_cia_n2o2 * factor_n2o2 +
                                k_cia_o2o2 * factor_o2o2 + k_cia_co2co2 * factor_co2co2)
            if not include_CIA: tau_cia_vertical *= 0.0
            tau_cia_down = tau_cia_vertical * path_factor
            tau_cia_up = tau_cia_vertical * 1.66
            tau_matrix_down = tau_h2o_down[:, :, None] + tau_o3_down[:, None, :] + tau_cia_down[:, None, None]
            tau_down = tau_matrix_down.reshape(N, -1)
            transmission_down = np.exp(-tau_down)
            tau_matrix_up = tau_h2o_up[:, :, None] + tau_o3_up[:, None, :] + tau_cia_up[:, None, None]
            tau_up = tau_matrix_up.reshape(N, -1)
            transmission_up = np.exp(-tau_up)
            w_g = w_g_sw[band_index]
            Fd_band, Fu_band = calc_sw_fluxes_reflect(N, transmission_down, transmission_up, local_layer_reflectivity,
                                                      S_toa, local_albedo, w_g)
            Fd_sw_total += Fd_band
            Fu_sw_total += Fu_band
        return Fd_sw_total, Fu_sw_total

    Fd_final, Fu_final = calc_sw_radiation_single(T_final, Ts_final, final_doy, initial_co2_ppm, initial_ch4_ppm,
                                           i_fac, p_fac, m_val, plot_albedo_val, plot_rh, plot_layer_reflect,
                                           local_p_mid, local_dp)

    local_p_int = p_int[plot_idx, :]
    ax2.plot(Fd_final, local_p_int / 100, 'k--', label='Incoming Solar')
    ax2.plot(Fu_final, local_p_int / 100, 'r-', linewidth=2, label='Reflected Up')

    if include_clouds:
        ax2.axhline(y=plot_cld_press / 100, color='gray', linestyle='-', linewidth=1, alpha=0.7)
        ax2.text(x_pos, plot_cld_press / 100, 'Cloud Layer', color='gray', fontweight='bold', fontsize=9, va='bottom',
                 ha='right')

    ax2.invert_yaxis()
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.set_xlabel("Flux (W/m²)")
    ax2.set_ylabel("Pressure (hPa)")
    ax2.set_title(f"SW Flux Profile (Final, {plot_lat}°)")
    ax2.legend()

    plt.tight_layout()
    # </editor-fold>

    # <editor-fold desc="forcing">
    time_years = np.linspace(0, years, len(dates))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(time_years, imbalance, color='red', linewidth=1.5, label='Global TOA Imbalance')
    ax1.set_ylabel("Energy Imbalance (W/m²)")
    ax1.set_title("Radiative Forcing & Recovery")
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.legend()

    num_lats = len(lats_out)
    sort_idx = np.argsort(lats_out)
    colors = plt.cm.jet(np.linspace(0, 1, num_lats))

    for idx, i in enumerate(sort_idx):
        lat = lats_out[i]
        ax2.plot(time_years, Ts_history[:, i], color=colors[idx], alpha=0.5, linewidth=1)

    ax2.plot(time_years, Ts, color='black', linewidth=2.5, label='Global Average')

    ax2.set_ylabel("Surface Temperature (K)")
    ax2.set_xlabel("Time (Years)")
    ax2.legend(loc='best')

    plt.tight_layout()
    if save_figures_to_desktop:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path = os.path.join(desktop_path, "forcing and surface")
        plotting.save_for_paper(file_path)
    # </editor-fold>
    plt.show()