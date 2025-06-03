import pandas as pd

# Load data
name_dataset = pd.read_csv("Data/Merged/Distinct_Name_All_Properties_Merged_V3.csv")
print("--------------------------\nInitial info\n--------------------------\n")
print(name_dataset.info())
print(name_dataset.head())

name_dataset.drop(columns=['Conductivity_in_Siemens_per_cm'], inplace=True)

df_cleaned = name_dataset.dropna(subset=["Compound_name"])
df_cleaned = df_cleaned.dropna(subset=["Extracted_name"])

df_cleaned = df_cleaned.drop_duplicates(subset=["Compound_name"])
df_cleaned = df_cleaned.drop_duplicates(subset=["Molecular_weight"])

df_cleaned["Efficiency_in_percent"].fillna(df_cleaned["Efficiency_in_percent"].median(), inplace=True)
df_cleaned["Voltage_in_V"].fillna(df_cleaned["Voltage_in_V"].median(), inplace=True)

df_cleaned["Energy_in_Watt_hour_per_kg"] = df_cleaned["Energy_in_Watt_hour_per_kg"]
df_cleaned["Energy_in_Watt_hour_per_kg"].fillna((df_cleaned['Capacity_per_gram_in_mAh'] * df_cleaned['Voltage_in_V']), inplace=True)

df_cleaned["Type"] = df_cleaned["Type"].str.strip().str.capitalize()
df_cleaned['Type'] = df_cleaned['Type'].replace({'Cath': 'Cathode', 'Anod': 'Anode'})

print(df_cleaned.info())
print(df_cleaned.head())

df_cleaned.to_csv("Data/Merged_Cleaned/Distinct_Name_All_Properties_Merged_V3.csv", index=False)
