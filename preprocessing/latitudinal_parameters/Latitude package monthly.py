import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to import user's custom plotting style if available
try:
    from plotting import plotting_overview

    plotting_overview.set_publication_style()
except ImportError:
    print("Custom 'plotting' module not found. Proceeding with default matplotlib styles.")

df_ceres = pd.read_csv('Ceres data.csv')
df_surface = pd.read_csv('surface_fractions_with_ice.csv')
df_elev = pd.read_csv('elevation data.csv')
print("Loaded data sets")
output_filename = '../../data/latitude_parameters/latitude_params_monthly.csv'

df_ceres['Month'] = pd.to_datetime(df_ceres['time']).dt.month

df_mean = df_ceres.groupby(['Month', 'lat']).mean(numeric_only=True).reset_index()

df_mean['Surface_Albedo'] = df_mean['zsfc_sw_up_clr_c_mon'] / (df_mean['zsfc_sw_down_clr_c_mon'] + 1e-6)
df_mean['Surface_Albedo'] = df_mean['Surface_Albedo'].clip(0, 1)

df_mean['Cloud_Fraction'] = df_mean['zcldarea_total_daynight_mon'] / 100.0
df_mean['Cloud_Pressure_Pa'] = df_mean['zcldpress_total_daynight_mon'] * 100.0

f = df_mean['Cloud_Fraction']
sw_up_all = df_mean['ztoa_sw_all_mon']
sw_up_clr = df_mean['ztoa_sw_clr_c_mon']
zsolar = df_mean['zsolar_mon']

cloud_albedo_raw = (sw_up_all - sw_up_clr * (1 - f)) / (zsolar * f + 1e-6)
df_mean['Cloud_Albedo'] = cloud_albedo_raw.clip(0, 1)

model_lats = np.linspace(-85, 85, 18)

heat_cap = (df_surface['water_fraction'] * 1e8) + \
           (df_surface['land_fraction'] * 4e6) + \
           (df_surface['ice_fraction'] * 4e6)

climo_lats = [-90, -60, -30, 0, 30, 60, 90]
climo_rh = [0.85, 0.80, 0.65, 0.82, 0.65, 0.80, 0.85]
rh_interp = np.interp(model_lats, climo_lats, climo_rh)

zonal_elev = df_elev.groupby('lat')['elevation'].mean().reset_index()
zonal_elev = zonal_elev.sort_values('lat')
elev_interp = np.interp(model_lats, zonal_elev['lat'], zonal_elev['elevation'])

initial_temp_profile = 240.0 + 60.0 * np.cos(np.radians(model_lats))

all_months_data = []

for month in range(1, 13):
    df_month = df_mean[df_mean['Month'] == month]
    if df_month.empty: continue

    month_params = pd.DataFrame({'Month': month, 'Latitude': model_lats})

    #VARIABLE
    month_params['Albedo'] = np.interp(model_lats, df_month['lat'], df_month['Surface_Albedo'])
    month_params['Cloud_Albedo'] = np.interp(model_lats, df_month['lat'], df_month['Cloud_Albedo'])
    month_params['Cloud_Fraction'] = np.interp(model_lats, df_month['lat'], df_month['Cloud_Fraction'])
    month_params['Cloud_Pressure'] = np.interp(model_lats, df_month['lat'], df_month['Cloud_Pressure_Pa'])

    #STATIC
    month_params['Initial_Ts'] = initial_temp_profile
    month_params['Heat_Capacity'] = heat_cap.values
    month_params['Relative_Humidity'] = rh_interp
    month_params['Elevation'] = elev_interp

    all_months_data.append(month_params)

final_params = pd.concat(all_months_data, ignore_index=True)

weights = np.cos(np.radians(model_lats))
weights /= weights.sum()

global_rows = []
variables_to_average = ['Albedo', 'Cloud_Albedo', 'Cloud_Fraction', 'Cloud_Pressure', 'Initial_Ts', 'Heat_Capacity',
                        'Relative_Humidity', 'Elevation']

for month in range(1, 13):
    month_data = final_params[final_params['Month'] == month]
    if month_data.empty: continue

    global_row = {'Month': month, 'Latitude': 'Global'}
    for var in variables_to_average:
        global_row[var] = np.average(month_data[var], weights=weights)

    global_rows.append(global_row)

global_df = pd.DataFrame(global_rows)
final_params = pd.concat([final_params, global_df], ignore_index=True)


final_params.to_csv(output_filename, index=False)
print(f"file {output_filename} is saved")
print(f"Note: Your CSV now has {len(final_params)} rows")

# 2 rows x 4 columns layout; figsize removed to let `plotting` module handle it
fig, axes_2d = plt.subplots(2, 4, figsize=(6.5, 2), sharey=True)
axes = axes_2d.flatten()

plot_params = final_params[final_params['Latitude'] != 'Global'].copy()
plot_params['Latitude'] = plot_params['Latitude'].astype(float)

def plot_monthly_spread(ax, var_name, xlabel_text, color):
    for m in range(1, 13):
        m_data = plot_params[plot_params['Month'] == m]
        ax.plot(m_data[var_name], m_data['Latitude'], color=color, alpha=0.3)

    annual_mean = plot_params.groupby('Latitude')[var_name].mean().reset_index()
    ax.plot(annual_mean[var_name], annual_mean['Latitude'], marker='o', color=color, label='Annual Mean')

    ax.set_xlabel(xlabel_text)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)

# --- TOP ROW: Dynamic Variables ---
plot_monthly_spread(axes[0], 'Albedo', 'Surface albedo ', 'dodgerblue')
plot_monthly_spread(axes[1], 'Cloud_Albedo', 'Cloud albedo ', 'darkorange')
plot_monthly_spread(axes[2], 'Cloud_Fraction', 'Cloud Cover', 'slategray')

for m in range(1, 13):
    m_data = plot_params[plot_params['Month'] == m]
    axes[3].plot(m_data['Cloud_Pressure'] / 100, m_data['Latitude'], color='purple', alpha=0.3)
annual_mean_cp = plot_params.groupby('Latitude')['Cloud_Pressure'].mean().reset_index()
axes[3].plot(annual_mean_cp['Cloud_Pressure'] / 100, annual_mean_cp['Latitude'], marker='o', color='purple')
axes[3].set_xlabel('Cloud Pressure (hPa)')
axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[3].invert_xaxis()

# --- BOTTOM ROW: Static Variables ---
static_data = plot_params[plot_params['Month'] == 1]

axes[4].plot(static_data['Initial_Ts'], static_data['Latitude'], marker='o', color='crimson')
axes[4].set_xlabel('Initial T$_s$ (K)')
axes[4].axhline(0, color='black', linestyle='--', alpha=0.5)

axes[5].plot(static_data['Relative_Humidity'], static_data['Latitude'], marker='o', color='teal')
axes[5].set_xlabel('Relative humidity')
axes[5].axhline(0, color='black', linestyle='--', alpha=0.5)

axes[6].plot(static_data['Elevation'], static_data['Latitude'], marker='o', color='saddlebrown')
axes[6].set_xlabel('Surface elevation (m)')
axes[6].axhline(0, color='black', linestyle='--', alpha=0.5)

axes[7].plot(static_data['Heat_Capacity'], static_data['Latitude'], marker='o', color='mediumseagreen')
axes[7].set_xlabel('C$_s$ (J m$^{-2}$ K$^{-1}$)')
axes[7].axhline(0, color='black', linestyle='--', alpha=0.5)


# --- FORMATTING & LABELS ---
axes[0].set_ylim(-90, 90)

# Format the y-axis for scientific publication on the left-most plots of BOTH rows
for ax in [axes[0], axes[4]]:
    ax.set_ylabel('Latitude')
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_yticklabels(['90°S', '60°S', '30°S', '0°', '30°N', '60°N', '90°N'])

labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

for ax, label in zip(axes, labels):
    ax.text(0.0, 1.02, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')

axes[0].legend(loc='lower left')

# Compress the horizontal and vertical space between subplots
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.3)

import os
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "paper latitude parameters")
plotting_overview.save_for_paper(file_path)

plt.show()