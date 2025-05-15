import pandas as pd
from pandas import DataFrame


def load_dataset(filename: str) -> DataFrame:
    print(f"\n--------------------------\nReading the table {filename}\n--------------------------")
    data_set = pd.read_csv(f"Data/Cleaned_Molecular_Weight/{filename}_with_molecular_weight_clean.csv")
    print("--------------------------\nInitial info\n--------------------------\n")
    print(data_set.info())
    print(data_set.head())
    return data_set


properties = ["CAPACITY", "VOLTAGE", "COULOMBIC_EFFICIENCY", "CONDUCTIVITY"]

capacity = load_dataset(properties[0])
capacity = capacity[["Name", "Extracted_name", "Molecular_weight", "Type", "Value"]]
capacity = capacity.rename(columns={"Name": "Compound_Name", "Value": "Capacity_per_gram_in_mAh"})

voltage = load_dataset(properties[1])
voltage = voltage[["Name", "Extracted_name", "Molecular_weight", "Type", "Value"]]
voltage = voltage.rename(columns={"Name": "Compound_Name", "Value": "Voltage_in_V"})

coulombs = load_dataset(properties[2])
coulombs = coulombs[["Name", "Extracted_name", "Molecular_weight", "Type", "Value"]]
coulombs = coulombs.rename(columns={"Name": "Compound_Name", "Value": "Efficiency_in_percent"})

conductivity = load_dataset(properties[3])
conductivity = conductivity[["Name", "Extracted_name", "Molecular_weight", "Type", "Value"]]
conductivity = conductivity.rename(columns={"Name": "Compound_Name", "Value": "Conductivity_in_Siemens_per_cm"})

merged = capacity.copy()
merged = merged.merge(voltage[["Molecular_weight", "Voltage_in_V"]], on="Molecular_weight", how="inner")
merged = merged.merge(coulombs[["Molecular_weight", "Efficiency_in_percent"]], on="Molecular_weight", how="inner")
merged = merged.merge(conductivity[["Molecular_weight", "Conductivity_in_Siemens_per_cm"]], on="Molecular_weight", how="right")

merged.to_csv("Data/Merged/Combined_all_properties_with_molecular_weight.csv", index=False)