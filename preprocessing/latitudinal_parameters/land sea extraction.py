import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the binned data
print("Loading binned land/water data...")
df = pd.read_csv('binned_10deg_land_water.csv')

# Ensure we have an ice fraction column starting at 0
df['ice_fraction'] = 0.0

# Earth radius in km for calculating latitude band areas
R_earth = 6371.0


def get_band_area_km2(lat_center):
    # Calculate the surface area of a 10-degree latitude band
    lat_top = np.radians(lat_center + 5)
    lat_bottom = np.radians(lat_center - 5)
    # Surface area formula for a spherical segment: 2 * pi * R^2 * |sin(top) - sin(bottom)|
    area = 2 * np.pi * (R_earth ** 2) * abs(np.sin(lat_top) - np.sin(lat_bottom))
    return area


# 2. Apply Ice Logic row by row
for index, row in df.iterrows():
    lat = row['Lat_Bin_Center']
    land = row['land_fraction']
    water = row['water_fraction']
    ice = 0.0

    # --- A. ANTARCTICA (Land Ice) ---
    if lat <= -65:
        # 98% of the land mass below 60S is ice
        antarctica_ice = land * 0.98
        ice += antarctica_ice
        land -= antarctica_ice

    # --- B. GREENLAND (Land Ice) ---
    if lat == 65:
        band_area = get_band_area_km2(lat)
        greenland_ice_area = 633700 * 0.80
        greenland_frac = greenland_ice_area / band_area
        # Shift this fraction from land to ice (safeguard against going below 0)
        actual_shift = min(land, greenland_frac)
        ice += actual_shift
        land -= actual_shift

    elif lat == 75:
        band_area = get_band_area_km2(lat)
        greenland_ice_area = 1175121 * 0.80
        greenland_frac = greenland_ice_area / band_area
        actual_shift = min(land, greenland_frac)
        ice += actual_shift
        land -= actual_shift

    elif lat == 85:
        band_area = get_band_area_km2(lat)
        greenland_ice_area = 250000 * 0.80
        greenland_frac = greenland_ice_area / band_area
        actual_shift = min(land, greenland_frac)
        ice += actual_shift
        land -= actual_shift

    # --- C. SEA ICE (Ocean Ice) ---
    sea_ice_shift = 0.0
    if lat == 85:
        sea_ice_shift = water * 0.90
    elif lat == 75:
        sea_ice_shift = water * 0.60
    elif lat == 65:
        sea_ice_shift = water * 0.15
    elif lat == -85:
        sea_ice_shift = water * 1.00
    elif lat == -75:
        sea_ice_shift = water * 0.75
    elif lat == -65:
        sea_ice_shift = water * 0.30

    ice += sea_ice_shift
    water -= sea_ice_shift

    # Update the dataframe
    df.at[index, 'land_fraction'] = land
    df.at[index, 'water_fraction'] = water
    df.at[index, 'ice_fraction'] = ice

# 3. Save the updated data
output_csv = 'surface_fractions_with_ice.csv'
df.to_csv(output_csv, index=False)
print(f"Calculations complete! Saved to {output_csv}")

# --- PLOTTING SECTION ---
print("Generating final surface distribution plot...")
plt.figure(figsize=(10, 6))

x_labels = df['Lat_Bin_Center'].astype(str)
land_frac = df['land_fraction']
water_frac = df['water_fraction']
ice_frac = df['ice_fraction']

# Stack the bars: Land (bottom), Water (middle), Ice (top)
plt.bar(x_labels, land_frac, label='Bare Land', color='sienna', edgecolor='black')
plt.bar(x_labels, water_frac, bottom=land_frac, label='Open Water', color='steelblue', edgecolor='black')
plt.bar(x_labels, ice_frac, bottom=land_frac + water_frac, label='Ice (Land & Sea)', color='lightcyan',
        edgecolor='black')

plt.title('Global Surface Composition by Latitude (Including Ice)', fontsize=14)
plt.xlabel('Latitude (Negative = Southern Hemisphere)', fontsize=12)
plt.ylabel('Fractional Coverage', fontsize=12)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()