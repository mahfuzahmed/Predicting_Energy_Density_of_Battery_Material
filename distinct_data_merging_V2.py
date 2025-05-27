import pandas as pd
from pandas import DataFrame


def load_dataset_weight(filename: str) -> DataFrame:
    print(f"\n--------------------------\nReading the table {filename}\n--------------------------")
    data_set = pd.read_csv(f"Data/Distinct_Cleaned_MW/{filename}_Distinct_by_weight.csv")
    print("--------------------------\nInitial info\n--------------------------\n")
    print(data_set.info())
    print(data_set.head())
    return data_set


def load_dataset_name(filename: str) -> DataFrame:
    print(f"\n--------------------------\nReading the table {filename}\n--------------------------")
    data_set = pd.read_csv(f"Data/Distinct_Cleaned_MW/{filename}_Distinct_by_name.csv")
    print("--------------------------\nInitial info\n--------------------------\n")
    print(data_set.info())
    print(data_set.head())
    return data_set


properties = ["CAPACITY", "VOLTAGE", "COULOMBIC_EFFICIENCY", "CONDUCTIVITY", "ENERGY"]

capacity = load_dataset_weight(properties[0])
capacity = capacity[["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Capacity_per_gram_in_mAh"]]

voltage = load_dataset_weight(properties[1])
voltage = voltage[["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Voltage_in_V"]]

coulombs = load_dataset_weight(properties[2])
coulombs = coulombs[["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Efficiency_in_percent"]]

conductivity = load_dataset_weight(properties[3])
conductivity = conductivity[
    ["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Conductivity_in_Siemens_per_cm"]]

energy = load_dataset_weight(properties[4])
energy = energy[
    ["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Energy_in_Watt_hour_per_kg"]]

merged = capacity.copy()
merged = merged.merge(voltage[["Molecular_weight", "Voltage_in_V"]], on="Molecular_weight", how="outer")
merged = merged.merge(coulombs[["Molecular_weight", "Efficiency_in_percent"]], on="Molecular_weight", how="outer")
merged = merged.merge(conductivity[["Molecular_weight", "Conductivity_in_Siemens_per_cm"]], on="Molecular_weight",
                      how="outer")
merged = merged.merge(energy[["Molecular_weight", "Energy_in_Watt_hour_per_kg"]], on="Molecular_weight", how="outer")

# merged = merged.dropna()
merged.to_csv("Data/Merged/Distinct_Weight_All_Properties_Merged_V3.csv", index=False)

#########################################################################################################################################


capacity = load_dataset_name(properties[0])
capacity = capacity[["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Capacity_per_gram_in_mAh"]]

voltage = load_dataset_name(properties[1])
voltage = voltage[["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Voltage_in_V"]]

coulombs = load_dataset_name(properties[2])
coulombs = coulombs[["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Efficiency_in_percent"]]

conductivity = load_dataset_name(properties[3])
conductivity = conductivity[
    ["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Conductivity_in_Siemens_per_cm"]]

energy = load_dataset_name(properties[4])
energy = energy[
    ["Compound_name", "Extracted_name", "Molecular_weight", "Type", "Energy_in_Watt_hour_per_kg"]]

merged = capacity.copy()
merged = merged.merge(voltage[["Molecular_weight", "Voltage_in_V"]], on="Molecular_weight", how="outer")
merged = merged.merge(coulombs[["Molecular_weight", "Efficiency_in_percent"]], on="Molecular_weight", how="outer")
merged = merged.merge(conductivity[["Molecular_weight", "Conductivity_in_Siemens_per_cm"]], on="Molecular_weight", how="outer")
merged = merged.merge(energy[["Molecular_weight", "Energy_in_Watt_hour_per_kg"]], on="Molecular_weight", how="outer")

# merged = merged.dropna(how='all')
# merged = merged.dropna()
# merged = merged.drop_duplicates(subset=["Compound_name"])
merged.to_csv("Data/Merged/Distinct_Name_All_Properties_Merged_V3.csv", index=False)
