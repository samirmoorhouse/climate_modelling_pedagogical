import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading CERES dataset...")
df = pd.read_csv('Ceres data.csv')

df_mean = df.groupby('lat').mean(numeric_only=True).reset_index()

df_mean['Albedo'] = df_mean['zsfc_sw_up_clr_c_mon'] / (df_mean['zsfc_sw_down_clr_c_mon'] + 1e-6)
df_mean['Albedo'] = df_mean['Albedo'].clip(0, 1)

df_mean['Initial_Ts'] = 280

df_mean['Cloud_Fraction'] = df_mean['zcldarea_total_daynight_mon'] / 100.0
df_mean['Cloud_Pressure_Pa'] = df_mean['zcldpress_total_daynight_mon'] * 100.0

model_lats = np.linspace(-85, 85, 18)
model_params = pd.DataFrame({'Latitude': model_lats})

model_params['Albedo'] = np.interp(model_lats, df_mean['lat'], df_mean['Albedo'])
model_params['Initial_Ts'] = np.interp(model_lats, df_mean['lat'], df_mean['Initial_Ts'])
model_params['Cloud_Fraction'] = np.interp(model_lats, df_mean['lat'], df_mean['Cloud_Fraction'])
model_params['Cloud_Pressure'] = np.interp(model_lats, df_mean['lat'], df_mean['Cloud_Pressure_Pa'])

print("Calculating Heat Capacity from surface fractions...")

df_surface = pd.read_csv('surface_fractions_with_ice.csv')

model_params['Heat_Capacity'] = (df_surface['water_fraction'] * 2.0e8) + \
                                (df_surface['land_fraction'] * 1.0e7) + \
                                (df_surface['ice_fraction'] * 1.0e7)

print("Calculating Climatological Relative Humidity...")

climo_lats = [-90, -60, -30,  0,  30,  60,  90]
climo_rh   = [0.85, 0.80, 0.65, 0.82, 0.65, 0.80, 0.85]

model_params['Relative_Humidity'] = np.interp(model_lats, climo_lats, climo_rh)
output_filename = '../../data/latitude_parameters/latitude_params.csv'
model_params.to_csv(output_filename, index=False)
print(f"Success! {output_filename} is ready for your main script.")

fig, axes = plt.subplots(1, 6, figsize=(18, 7), sharey=True)

# Define a single plotting function to keep the code clean
def plot_param(ax, x_data, y_data, title, xlabel, color):
    ax.plot(x_data, y_data, marker='o', linestyle='-', color=color, markersize=5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)

lats = model_params['Latitude']

plot_param(axes[0], model_params['Albedo'], lats, 'Surface Albedo', 'Fraction', 'dodgerblue')
axes[0].set_ylabel('Latitude', fontsize=12)
axes[0].set_ylim(-90, 90)

plot_param(axes[1], model_params['Initial_Ts'], lats, 'Initial Temp (Ts)', 'K', 'crimson')

plot_param(axes[2], model_params['Heat_Capacity'], lats, 'Heat Capacity (Cs)', '', 'sienna')
axes[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plot_param(axes[3], model_params['Relative_Humidity'], lats, 'Relative Humidity', 'Fraction', 'teal')

plot_param(axes[4], model_params['Cloud_Fraction'], lats, 'Cloud Cover', 'Fraction', 'slategray')

plot_param(axes[5], model_params['Cloud_Pressure'] / 100, lats, 'Cloud Pressure', 'hPa', 'purple')
axes[5].invert_xaxis()

plt.tight_layout()
plt.show()