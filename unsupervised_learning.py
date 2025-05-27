import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# Load your dataset
df = pd.read_csv("Data/Scaled_dataset/Scaled_dataset.csv")

# Numeric features to use
features = ['Molecular_weight', 'Capacity_per_gram_in_mAh', 'Voltage_in_V', 'Efficiency_in_percent']
X = df[features]

# 1. Determine best k for KMeans
silhouette_scores = {}
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores[k] = score

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"Best number of clusters based on silhouette score: {best_k}")

# 2. Apply KMeans with best_k
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X)

# 3. Apply Agglomerative Clustering with same k
agglo = AgglomerativeClustering(n_clusters=best_k)
df['Agglo_Cluster'] = agglo.fit_predict(X)

# 4. Apply DBSCAN (you can adjust eps and min_samples as needed)
dbscan = DBSCAN(eps=0.8, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X)

# 5. PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# 6. Plot clusters from each algorithm
def plot_clusters(cluster_col, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue=cluster_col, palette='tab10', s=70)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.legend(title=cluster_col)
    plt.show()


plot_clusters('KMeans_Cluster', f'KMeans Clustering (k={best_k})')
plot_clusters('Agglo_Cluster', 'Agglomerative Clustering')
plot_clusters('DBSCAN_Cluster', 'DBSCAN Clustering')

# 7. Optional: Print DBSCAN cluster counts
print("DBSCAN cluster label counts:")
print(df['DBSCAN_Cluster'].value_counts())
