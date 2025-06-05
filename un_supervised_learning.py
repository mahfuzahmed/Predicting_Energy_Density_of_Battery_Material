import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
df = pd.read_csv("Data/Merged_Cleaned/Distinct_Name_All_Properties_Merged_V3.csv")

output_folder = "Plots/Unsupervised_learning"

# Numeric columns for analysis
numeric_cols = [
    'Molecular_weight',
    'Capacity_per_gram_in_mAh',
    'Voltage_in_V',
    'Efficiency_in_percent',
    'Energy_in_Watt_hour_per_kg'
]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])

# ----------------- PCA ----------------- #
pca = PCA()
pca_components = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_

pca_loadings = pd.DataFrame(pca.components_.T, index=numeric_cols, columns=[f'PC{i+1}' for i in range(len(numeric_cols))])
print(pca_loadings)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
sns.lineplot(x=range(1, len(explained_variance)+1), y=explained_variance.cumsum(), marker="o")
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_folder}/Plot cumulative explained variance.png")
plt.show()

# 2D PCA Plot
pca_df = pd.DataFrame(pca_components[:, :2], columns=['PC1', 'PC2'])
pca_df['Type'] = df['Type']
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Type', palette='Set2', s=60, alpha=0.8)
plt.title('2D PCA Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Type')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_folder}/PCA_plot.png")
plt.show()

print("\nCumulative explained variance (2 components):", explained_variance[:2].sum(), "\n")

# ----------------- Factor Analysis ----------------- #
fa = FactorAnalysis(n_components=2, random_state=42)
fa_components = fa.fit_transform(scaled_data)
factor_loadings = pd.DataFrame(fa.components_.T, index=numeric_cols, columns=['Factor1', 'Factor2'])

# Plot heatmap of factor loadings
plt.figure(figsize=(6, 4))
sns.heatmap(factor_loadings, annot=True, cmap="coolwarm", center=0)
plt.title("Factor Loadings Heatmap")
plt.tight_layout()
plt.savefig(f"{output_folder}/Factor loading heatmap.png")
plt.show()

# ----------------- KMeans Clustering ----------------- #
inertia = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(scaled_data, kmeans.labels_))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.lineplot(x=list(K_range), y=inertia, marker='o')
plt.title('KMeans Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
sns.lineplot(x=list(K_range), y=sil_scores, marker='o', color='green')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig(f"{output_folder}/K_means.png")
plt.show()

best_k = sil_scores.index(max(sil_scores)) + 2
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)
print(f"Best KMeans Clusters: {best_k}, Silhouette Score: {max(sil_scores):.3f}")

# ----------------- Hierarchical Clustering ----------------- #
plt.figure(figsize=(10, 5))
linked = linkage(scaled_data, method='ward')
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig(f"{output_folder}/Dendrogram.png")
plt.show()

agglo = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
agglo_labels = agglo.fit_predict(scaled_data)
print(f"Agglomerative Clustering Silhouette Score: {silhouette_score(scaled_data, agglo_labels):.3f}")

# ----------------- DBSCAN ----------------- #
dbscan = DBSCAN(eps=2, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_data)
mask = dbscan_labels != -1
if len(set(dbscan_labels)) > 1 and np.sum(mask) > 0:
    score = silhouette_score(scaled_data[mask], dbscan_labels[mask])
    print(f"DBSCAN Silhouette Score (excluding noise): {score:.3f}")
else:
    print("DBSCAN could not form valid clusters. Try different eps/min_samples.")

# DBSCAN k-distance plot
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(scaled_data)
distances, indices = neighbors_fit.kneighbors(scaled_data)

plt.figure(figsize=(6, 4))
plt.plot(sorted(distances[:, 4]), marker='o')
plt.title("DBSCAN: k-distance Graph (k=5)")
plt.xlabel("Points sorted by distance")
plt.ylabel("5th Nearest Neighbor Distance")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_folder}/K-distance plot.png")
plt.show()

# Count unique cluster labels excluding noise (-1)
unique_labels = set(dbscan_labels)
n_clusters = len(unique_labels - {-1})

print(f"\nNumber of clusters formed by DBSCAN (excluding noise): {n_clusters}")
