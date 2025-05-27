import pandas as pd
from sklearn import preprocessing

file_path = "Data/Merged_Cleaned/Distinct_Name_All_Properties_Merged.csv"
dataframe = pd.read_csv(file_path)

columns_to_scale = [
    'Molecular_weight',
    'Capacity_per_gram_in_mAh',
    'Voltage_in_V',
    'Efficiency_in_percent'
]

min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(dataframe[columns_to_scale])

scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)

non_scaled_df = dataframe.drop(columns=columns_to_scale).reset_index(drop=True)

final_df = pd.concat([non_scaled_df, scaled_df], axis=1)

output_path = "Data/Scaled_dataset/Scaled_dataset.csv"
final_df.to_csv(output_path, index=False)

print(f"Scaled dataset saved to {output_path}")
