"""
@author: Jams
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# =====================================================
# 1. INPUT FILE
# =====================================================
input_file = r"New Volume D/River System/SW277.xlsx"

# =====================================================
# 2. READ EXCEL
# =====================================================
df = pd.read_excel(input_file)

# First column = Date
date_col = df.columns[0]

# Convert to datetime
df[date_col] = pd.to_datetime(
    df[date_col],
    dayfirst=True,
    errors='coerce'
)

# Sort by date
df = df.sort_values(date_col).reset_index(drop=True)

# =====================================================
# 3. FIND MISSING CALENDAR DATES
# =====================================================
full_dates = pd.date_range(
    start=df[date_col].min(),
    end=df[date_col].max(),
    freq='D'
)

missing_dates = full_dates.difference(df[date_col])

print("Missing dates found:", len(missing_dates))

# Save missing dates
missing_dates_df = pd.DataFrame({
    "Missing_Date": missing_dates
})

missing_dates_df.to_excel(
    "Missing_Calendar_Dates_SW277.xlsx",
    index=False
)

# =====================================================
# 4. INSERT MISSING DATES
# =====================================================
df_full = (
    df.set_index(date_col)
      .reindex(full_dates)
      .reset_index()
)

df_full.rename(
    columns={"index": date_col},
    inplace=True
)

# =====================================================
# 5. CREATE TEMPORAL FEATURES
# =====================================================
df_full['Year'] = df_full[date_col].dt.year
df_full['Month'] = df_full[date_col].dt.month
df_full['Day'] = df_full[date_col].dt.day
df_full['DOY'] = df_full[date_col].dt.dayofyear

# =====================================================
# 6. IDENTIFY DATA COLUMNS
# =====================================================
# Assuming:
# Column 1 = Date
# Column 2,3,4 = Water level stations

water_cols = list(df.columns[1:4])

print("\nWater Level Columns:")
print(water_cols)

# =====================================================
# 7. KNN IMPUTATION
# =====================================================
features_for_knn = (
    ['Year', 'Month', 'Day', 'DOY']
    + water_cols
)

knn_data = df_full[features_for_knn]

imputer = KNNImputer(
    n_neighbors=15,
    weights='distance'
)

imputed = imputer.fit_transform(knn_data)

imputed_df = pd.DataFrame(
    imputed,
    columns=features_for_knn
)

# Replace only water-level columns
for col in water_cols:
    df_full[col] = imputed_df[col]

# =====================================================
# 8. REMOVE TEMPORAL FEATURES
# =====================================================
df_final = df_full[
    [date_col] + water_cols
]

# =====================================================
# 9. SAVE OUTPUT
# =====================================================
output_file = "SW277_KNN_Imputed.xlsx"

df_final.to_excel(
    output_file,
    index=False
)

print("\nFinished Successfully!")
print("Missing dates file:")
print("   Missing_Calendar_Dates_SW266.xlsx")

print("\nImputed dataset:")
print("   River_Level_KNN_Imputed.xlsx")