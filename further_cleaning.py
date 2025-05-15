import pandas as pd
from pandas import DataFrame


def load_dataset(filename: str) -> DataFrame:
    print(f"\n--------------------------\nReading the table {filename}\n--------------------------")
    data_set = pd.read_csv(f"Data/Cleaned_Molecular_Weight/{filename}_with_molecular_weight_clean.csv")
    print("--------------------------\nInitial info\n--------------------------\n")
    print(data_set.info())
    print(data_set.head())
    return data_set


properties = ["CAPACITY", "VOLTAGE", "COULOMBIC_EFFICIENCY", "CONDUCTIVITY", "ENERGY"]

capacity = load_dataset(properties[0])
capacity = capacity[["Name", "Extracted_name", "Molecular_weight", "Type", "Value"]]
capacity = capacity.rename(columns={"Name": "Compound_name", "Value": "Capacity_per_gram_in_mAh"})
capacity.to_csv(f"Data/Cleaned_Renamed_MW/{properties[0]}_Cleaned_Renamed_MW.csv", index=False)

voltage = load_dataset(properties[1])
voltage = voltage[["Name", "Extracted_name", "Molecular_weight", "Type", "Value"]]
voltage = voltage.rename(columns={"Name": "Compound_name", "Value": "Voltage_in_V"})
voltage.to_csv(f"Data/Cleaned_Renamed_MW/{properties[1]}_Cleaned_Renamed_MW.csv", index=False)

coulombs = load_dataset(properties[2])
coulombs = coulombs[["Name", "Extracted_name", "Molecular_weight", "Type", "Value"]]
coulombs = coulombs.rename(columns={"Name": "Compound_name", "Value": "Efficiency_in_percent"})
coulombs.to_csv(f"Data/Cleaned_Renamed_MW/{properties[2]}_Cleaned_Renamed_MW.csv", index=False)

conductivity = load_dataset(properties[3])
conductivity = conductivity[["Name", "Extracted_name", "Molecular_weight", "Type", "Value"]]
conductivity = conductivity.rename(columns={"Name": "Compound_name", "Value": "Conductivity_in_Siemens_per_cm"})
conductivity.to_csv(f"Data/Cleaned_Renamed_MW/{properties[3]}_Cleaned_Renamed_MW.csv", index=False)

energy = load_dataset(properties[4])
energy = energy[["Name", "Extracted_name", "Molecular_weight", "Type", "Value"]]
energy = energy.rename(columns={"Name": "Compound_name", "Value": "Energy_in_Watt_hour_per_kg"})
energy.to_csv(f"Data/Cleaned_Renamed_MW/{properties[4]}_Cleaned_Renamed_MW.csv", index=False)
