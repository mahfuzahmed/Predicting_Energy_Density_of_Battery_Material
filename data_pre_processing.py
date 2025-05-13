import json
import pandas as pd
from pandas import DataFrame

# Atomic weights dictionary (g/mol)
ATOMIC_WEIGHTS = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011, 'N': 14.007,
    'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085,
    'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867,
    'V': 50.9415, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546,
    'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
    'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.95, 'Tc': 98,
    'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71,
    'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33, 'La': 138.91,
    'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Pm': 145, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25,
    'Tb': 158.93, 'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97,
    'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08,
    'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Th': 232.04, 'Pa': 231.04,
    'U': 238.03
}


def load_dataset(filename: str) -> DataFrame:
    print(f"\n--------------------------\nReading the table {filename}\n--------------------------")
    data_set = pd.read_csv(f"Data/Cleaned/{filename}_clean.csv")
    print("--------------------------\nInitial info\n--------------------------\n")
    print(data_set.info())
    print(data_set.head())
    return data_set


def calculate_molecular_weight(extracted_name):
    try:
        compounds = json.loads(extracted_name.replace("'", '"'))
        molecular_weight = 0.0
        for compound in compounds:
            for element, count in compound.items():
                count = float(count)
                if element in ATOMIC_WEIGHTS:
                    molecular_weight += ATOMIC_WEIGHTS[element] * count
                else:
                    raise ValueError(f"Unknown element: {element}")
        return molecular_weight
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing input: {e}")
        return None


# Main execution
properties = ["CAPACITY", "VOLTAGE", "COULOMBIC_EFFICIENCY", "ENERGY", "CONDUCTIVITY"]
weights = []

for prop in properties:
    df = load_dataset(prop)
    df["Molecular_weight"] = df["Extracted_name"].apply(calculate_molecular_weight)
    data_set_with_molecular_weight = df.dropna()
    output_path = f"Data/Cleaned_Molecular_Weight/{prop}_with_molecular_weight_clean.csv"
    data_set_with_molecular_weight.to_csv(output_path, index=False)
    print(f"Saved updated dataset to {output_path}")
