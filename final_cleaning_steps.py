import pandas as pd

name_dataset = pd.read_csv("Data/Merged/Distinct_Name_All_Properties_Merged.csv")
print("--------------------------\nInitial info\n--------------------------\n")
print(name_dataset.info())
print(name_dataset.head())

# Step 1: Drop rows with missing values in critical columns
critical_columns = ['Compound_name', 'Extracted_name', 'Type', 'Capacity_per_gram_in_mAh']
name_dataset_cleaned = name_dataset.dropna(subset=critical_columns)

# Step 2: Impute missing values
# Impute 'Voltage_in_V' and 'Efficiency_in_percent' with median
name_dataset_cleaned['Voltage_in_V'].fillna(name_dataset_cleaned['Voltage_in_V'].median(), inplace=True)
name_dataset_cleaned['Efficiency_in_percent'].fillna(name_dataset_cleaned['Efficiency_in_percent'].median(), inplace=True)

# Step 3: Drop 'Conductivity_in_Siemens_per_cm' due to excessive missing values
name_dataset_cleaned.drop(columns=['Conductivity_in_Siemens_per_cm'], inplace=True)

# Optional: Reset index
name_dataset_cleaned.reset_index(drop=True, inplace=True)

name_dataset_cleaned = name_dataset_cleaned.drop_duplicates(subset=["Compound_name"])
name_dataset_cleaned = name_dataset_cleaned.drop_duplicates(subset=["Molecular_weight"])


# Save cleaned dataset
name_dataset_cleaned.to_csv("Data/Merged_Cleaned/Distinct_Name_All_Properties_Merged.csv", index=False)

# Preview cleaned data
print(name_dataset_cleaned.info())
print(name_dataset_cleaned.head())



