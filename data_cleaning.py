import pandas as pd

properties = ["CAPACITY", "VOLTAGE", "COULOMBIC_EFFICIENCY", "ENERGY", "CONDUCTIVITY"]
total_missing = 0
total_clean_rows = 0

for name in properties:
    print(f"\n-------------------------------------------\nReading the table {name}\n-------------------------------------------\n")
    data_set = pd.read_csv(f"Data/Main/{name}.csv", encoding='latin1')

    print("\n------------------\nInitial info\n------------------\n")
    data_set.info()
    data_set.head()

    print("\nMissing Values per Column:\n---------------------------------\n")
    print(data_set.isnull().sum())

    # Store rows with missing values in a separate dataset
    na_rows = data_set[data_set.isnull().any(axis=1)]
    na_rows.to_csv(f"Data/{name}_missing.csv", index=False)
    total_missing = total_missing + len(na_rows)

    # Save cleaned dataset (drop missing values)
    data_set_clean = data_set.dropna()
    data_set_clean.to_csv(f"Data/{name}_clean.csv", index=False)
    total_clean_rows = total_clean_rows + len(data_set_clean)

    print(f"\nSaved {len(na_rows)} rows with missing values to {name}_missing.csv")
    print(f"Saved {len(data_set_clean)} cleaned rows to {name}_clean.csv")
    print("\nRemaining Missing Values in Cleaned Dataset:\n---------------------------------\n")
    print(data_set_clean.isnull().sum())


print(total_missing, total_clean_rows)
