import pandas as pd
from pandas import DataFrame


def load_dataset(filename: str) -> DataFrame:
    print(f"\n--------------------------\nReading the table {filename}\n--------------------------")
    data_set = pd.read_csv(f"Data/Cleaned_Renamed_MW/{filename}_Cleaned_Renamed_MW.csv")
    print("--------------------------\nInitial info\n--------------------------\n")
    print(data_set.info())
    print(data_set.head())
    return data_set


properties = ["CAPACITY", "VOLTAGE", "COULOMBIC_EFFICIENCY", "CONDUCTIVITY", "ENERGY"]

for prop in properties:
    dataset = load_dataset(prop)
    distinct_by_name = dataset.drop_duplicates(subset=["Compound_name"])
    distinct_by_name.to_csv(f"Data/Distinct_Cleaned_MW/{prop}_Distinct_by_name.csv", index=False)

    distinct_by_weight = dataset.drop_duplicates(subset=["Molecular_weight"])
    distinct_by_weight.to_csv(f"Data/Distinct_Cleaned_MW/{prop}_Distinct_by_weight.csv", index=False)



