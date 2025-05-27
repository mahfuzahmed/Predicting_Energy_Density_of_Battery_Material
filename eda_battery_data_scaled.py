import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Data/Scaled_dataset/Scaled_dataset_V2.csv")

# Select numeric columns
numeric_cols = ['Molecular_weight', 'Capacity_per_gram_in_mAh', 'Voltage_in_V', 'Efficiency_in_percent']

# 1. Histograms
for col in numeric_cols:
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# 2. Box plots
for col in numeric_cols:
    sns.boxplot(y=df[col], palette='Set2')
    plt.title(f'Boxplot of {col}')
    plt.grid(True)
    plt.show()

# 3. Count plot for 'Type'
sns.countplot(data=df, x='Type', palette='pastel')
plt.title('Count of Compounds by Type')
plt.ylabel('Count')
plt.grid(axis='y')
plt.show()

# 4. Grouped Box plots
for col in numeric_cols:
    sns.boxplot(data=df, x='Type', y=col, palette='muted')
    plt.title(f'{col} by Type')
    plt.grid(True)
    plt.show()

# 5. Correlation Heatmap
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# 6. Pairplot
sns.pairplot(df[numeric_cols], diag_kind='kde', plot_kws={'alpha':0.6})
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

# 7. Unique values for Efficiency
print("Unique values in Efficiency_in_percent:")
print(df['Efficiency_in_percent'].value_counts())

# 8. Top performers
print("\nTop 10 Compounds by Capacity:")
print(df[['Compound_name', 'Capacity_per_gram_in_mAh']].sort_values(by='Capacity_per_gram_in_mAh', ascending=False).head(10))

print("\nTop 10 Compounds by Voltage:")
print(df[['Compound_name', 'Voltage_in_V']].sort_values(by='Voltage_in_V', ascending=False).head(10))

print("\nTop 10 Compounds by Efficiency:")
print(df[['Compound_name', 'Efficiency_in_percent']].sort_values(by='Efficiency_in_percent', ascending=False).head(10))

# 9. PCA
X = df[numeric_cols]
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

sns.scatterplot(x='PCA1', y='PCA2', hue='Type', data=df, palette='tab10', s=70, alpha=0.8)
plt.title('PCA Projection of Battery Compounds')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend(title='Type')
plt.show()

# 10. Clustering using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1', s=70)
plt.title('KMeans Clusters in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend(title='Cluster')
plt.show()
