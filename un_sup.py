import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Data/Merged_Cleaned/Distinct_Name_All_Properties_Merged_V3.csv")

# Drop missing values
df_cleaned = df.dropna()

# Use only numeric columns
df_features = df_cleaned.select_dtypes(include=['float64', 'int64'])

# Dimensionality reduction for visualization (PCA)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_features)

# --- KMeans Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(df_features)
kmeans_silhouette = silhouette_score(df_features, kmeans_labels)

# --- DBSCAN Clustering ---
dbscan = DBSCAN(eps=1.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_features)
# Handle case where all points might be noise
if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(df_features, dbscan_labels)
else:
    dbscan_silhouette = np.nan

# --- Agglomerative Clustering ---
agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(df_features)
agg_silhouette = silhouette_score(df_features, agg_labels)

# --- t-SNE for Visualization ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(df_features)


# --- Plotting Function ---
def plot_clusters(embedding, labels, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette='viridis', s=60)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.show()


# --- Visualizations ---
plot_clusters(pca_result, kmeans_labels, "KMeans Clustering (PCA View)")
plot_clusters(pca_result, dbscan_labels, "DBSCAN Clustering (PCA View)")
plot_clusters(pca_result, agg_labels, "Agglomerative Clustering (PCA View)")
plot_clusters(tsne_result, kmeans_labels, "KMeans Clustering (t-SNE View)")

# --- Silhouette Score Comparison ---
print("Silhouette Scores:")
print(f"KMeans: {kmeans_silhouette:.3f}")
print(f"DBSCAN: {dbscan_silhouette if not np.isnan(dbscan_silhouette) else 'N/A (too few clusters)'}")
print(f"Agglomerative: {agg_silhouette:.3f}")
