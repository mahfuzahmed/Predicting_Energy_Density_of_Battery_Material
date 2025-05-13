import pandas as pd
from pandas import DataFrame


def load_dataset(filename: str) -> DataFrame:
    print(f"\n--------------------------\nReading the table {filename}\n--------------------------")
    data_set = pd.read_csv(f"Data/Cleaned/{filename}_clean.csv")
    print("--------------------------\nInitial info\n--------------------------\n")
    print(data_set.info())
    print(data_set.head())
    return data_set



