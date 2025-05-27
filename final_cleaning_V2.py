import pandas as pd

# Load data
name_dataset = pd.read_csv("Data/Merged/Distinct_Name_All_Properties_Merged.csv")
print("--------------------------\nInitial info\n--------------------------\n")
print(name_dataset.info())
print(name_dataset.head())

# 1. Drop rows where key features are missing (e.g., Compound_name or Capacity)
df_cleaned = name_dataset.dropna(subset=["Compound_name", "Capacity_per_gram_in_mAh"])

# 2. Fill numeric columns with mean or median
df_cleaned["Voltage_in_V"].fillna(df_cleaned["Voltage_in_V"].median(), inplace=True)
df_cleaned["Efficiency_in_percent"].fillna(df_cleaned["Efficiency_in_percent"].median(), inplace=True)
df_cleaned["Conductivity_in_Siemens_per_cm"].fillna(df_cleaned["Conductivity_in_Siemens_per_cm"].median(), inplace=True)

# 3. Standardize categorical values
df_cleaned["Type"] = df_cleaned["Type"].str.strip().str.capitalize()
df_cleaned['Type'] = df_cleaned['Type'].replace({'Cath': 'Cathode', 'Anod': 'Anode'})

# 4. Drop duplicate rows
df_cleaned.drop_duplicates(inplace=True)
df_cleaned = df_cleaned.drop_duplicates(subset=["Compound_name"])

# Step 5: Drop 'Conductivity_in_Siemens_per_cm' due to excessive missing values
df_cleaned.drop(columns=['Conductivity_in_Siemens_per_cm'], inplace=True)

df_cleaned.to_csv("Data/Merged_Cleaned/Distinct_Name_All_Properties_Merged_V2.csv", index=False)
