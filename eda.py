import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("Data/Merged_Cleaned/Distinct_Name_All_Properties_Merged_V3.csv")

output_folder = "Plots/Main_dataset"

# ----------------------
# Basic Overview
# ----------------------
print("\n--- Dataset Info ---\n")
print(df.info())

print("\n--- Summary Statistics ---\n")
print(df.describe())

print("\n--- Missing Values ---\n")
print(df.isna().sum())

print("\n--- Unique Types ---\n")
print(df['Type'].value_counts())

# ----------------------
# Histograms of Key Numeric Columns
# ----------------------
numeric_cols = [
    "Capacity_per_gram_in_mAh",
    "Voltage_in_V",
    "Energy_in_Watt_hour_per_kg",
    "Efficiency_in_percent",
    "Molecular_weight"
]

df[numeric_cols].hist(bins=30, figsize=(12, 8), layout=(2, 3))
plt.tight_layout()
plt.suptitle("Histograms of Key Features", y=1.02)
plt.savefig(f"{output_folder}/histograms_key_features.png")
plt.show()

# ----------------------
# Correlation Heatmap
# ----------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Correlation Heatmap")
plt.savefig(f"{output_folder}/correlation_heatmap.png")
plt.show()

# ----------------------
# Boxplot of Capacity, Voltage, Efficiency, Energy, Molecular Weight by Type
# ----------------------
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Type", y="Molecular_weight")
plt.title("Molecular Weight Distribution by Type")
plt.savefig(f"{output_folder}/Boxplot_Type_Molecular_weight.png")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Type", y="Capacity_per_gram_in_mAh")
plt.title("Capacity Distribution by Type")
plt.savefig(f"{output_folder}/Boxplot_Type_Capacity.png")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Type", y="Efficiency_in_percent")
plt.title("Efficiency Distribution by Type")
plt.savefig(f"{output_folder}/Boxplot_Type_Efficiency.png")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Type", y="Energy_in_Watt_hour_per_kg")
plt.title("Energy Distribution by Type")
plt.savefig(f"{output_folder}/Boxplot_Type_Energy.png")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Type", y="Voltage_in_V")
plt.title("Voltage Distribution by Type")
plt.savefig(f"{output_folder}/Boxplot_Type_Voltage.png")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------
# Pairplot for Top Features
# ----------------------
top_features = ["Capacity_per_gram_in_mAh", "Voltage_in_V", "Energy_in_Watt_hour_per_kg", "Molecular_weight"]
sns.pairplot(df[top_features].dropna())
plt.suptitle("Pairwise Relationships", y=1.02)
plt.savefig(f"{output_folder}/Pairwise_relations.png")
plt.show()
