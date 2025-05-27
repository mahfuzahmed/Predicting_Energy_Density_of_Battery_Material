import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

file_path = "Data/Merged_Cleaned/Distinct_Name_All_Properties_Merged_V3.csv"
df = pd.read_csv(file_path)

columns_to_scale = [
    'Molecular_weight',
    'Capacity_per_gram_in_mAh',
    'Voltage_in_V',
    'Efficiency_in_percent'
    ,'Energy_in_Watt_hour_per_kg'
]

# Standardization
scaler_std = StandardScaler()
standard_scaled_df = df.copy()
standard_scaled_df[columns_to_scale] = scaler_std.fit_transform(df[columns_to_scale])
standard_scaled_df.to_csv("Data/Scaled_dataset/Scaled_dataset_V3.csv", index=False)

# Normalization
scaler_norm = MinMaxScaler()
normalized_df = df.copy()
normalized_df[columns_to_scale] = scaler_norm.fit_transform(df[columns_to_scale])
normalized_df.to_csv("Data/Scaled_dataset/Normalized_dataset_V3.csv", index=False)


print(f"Scaled and normalized dataset saved")
