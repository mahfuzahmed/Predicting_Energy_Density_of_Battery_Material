import pandas as pd
from pandas import DataFrame


def load_dataset(filename: str) -> DataFrame:
    print(f"\n--------------------------\nReading the table {filename}\n--------------------------")
    data_set = pd.read_csv(f"Data/Cleaned/{filename}_clean.csv")
    print("--------------------------\nInitial info\n--------------------------\n")
    print(data_set.info())
    print(data_set.head())
    return data_set


properties = ["CAPACITY", "VOLTAGE", "COULOMBIC_EFFICIENCY", "ENERGY", "CONDUCTIVITY"]

dfs = [pd.read_csv(f"Data/Cleaned_Molecular_Weight/{prop}_with_molecular_weight_clean.csv") for prop in properties]
combined_vertical = pd.concat(dfs, ignore_index=True)

combined_vertical.to_csv("Data/Merged/Combined_all_data_with_molecular_weight_clean.csv", index=False)

