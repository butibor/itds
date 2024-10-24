
# Clustering Numerical Data with Multiple Methods

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats

# 1. Loading and Exploring the Data
data = pd.read_csv('data.csv')  # Load your own dataset here

# Explore the dataset
print(data.head())  # Display the first few rows
print(data.info())  # Data types and missing values
print(data.describe())  # Statistical summary of numerical columns

# 2. Data Cleaning
# Handling missing values
data = data.dropna()  # Alternatively, data.fillna(method='ffill') can be used

# Removing duplicates
data = data.drop_duplicates()

# Outlier detection (optional)
# Example: using Z-score to filter out outliers (assuming `data` has numerical values)
z_scores = np.abs(stats.zscore(data.select_dtypes(include=np.number)))
data = data[(z_scores < 3).all(axis=1)]  # Retaining data within 3 standard deviations

# 3. Feature Scaling
# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Optional: Normalization if required
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data)

# 4. Dimensionality Reduction (Optional)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Visualize PCA components
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title('PCA-reduced Data')
plt.show()

# 5. Clustering Methods

# A. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Plot K-Means result
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.show()

# B. Agglomerative Hierarchical Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(scaled_data)

# Plot Agglomerative Clustering result
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=agg_labels, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.show()

# C. DBSCAN (Density-Based Spatial Clustering)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_data)

# Plot DBSCAN result
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()

# D. Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(scaled_data)

# Plot GMM result
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=gmm_labels, cmap='viridis')
plt.title('Gaussian Mixture Model Clustering')
plt.show()

# 6. Cluster Evaluation

# Silhouette Scores
kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
agg_silhouette = silhouette_score(scaled_data, agg_labels)
dbscan_silhouette = silhouette_score(scaled_data, dbscan_labels)
gmm_silhouette = silhouette_score(scaled_data, gmm_labels)

print(f'Silhouette Score - KMeans: {kmeans_silhouette}')
print(f'Silhouette Score - Agglomerative: {agg_silhouette}')
print(f'Silhouette Score - DBSCAN: {dbscan_silhouette}')
print(f'Silhouette Score - GMM: {gmm_silhouette}')

# Davies-Bouldin Scores
kmeans_db = davies_bouldin_score(scaled_data, kmeans_labels)
agg_db = davies_bouldin_score(scaled_data, agg_labels)
dbscan_db = davies_bouldin_score(scaled_data, dbscan_labels)
gmm_db = davies_bouldin_score(scaled_data, gmm_labels)

print(f'Davies-Bouldin Score - KMeans: {kmeans_db}')
print(f'Davies-Bouldin Score - Agglomerative: {agg_db}')
print(f'Davies-Bouldin Score - DBSCAN: {dbscan_db}')
print(f'Davies-Bouldin Score - GMM: {gmm_db}')

# 7. Model Comparison
best_model = min([(kmeans_silhouette, 'KMeans'), 
                  (agg_silhouette, 'Agglomerative'),
                  (dbscan_silhouette, 'DBSCAN'),
                  (gmm_silhouette, 'GMM')], key=lambda x: -x[0])

print(f'The best model based on Silhouette Score is: {best_model[1]}')

# 8. Final Visualization
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('Final Clustering Result (KMeans)')
plt.show()
