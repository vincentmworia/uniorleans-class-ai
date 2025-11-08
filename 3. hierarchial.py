# ============================================
# Lab: Understanding Hierarchical Clustering
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# ============================================
# Part 1: Synthetic Data and Clustering
# ============================================

# Generate synthetic 2D data
X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Generated 2D data")
plt.grid(True)
plt.show()

# ============================================
# Part 2: Agglomerative Clustering with Dendrogram
# ============================================

# Compute linkage matrix (use 'ward', 'single', 'complete', or 'average')
linked = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# ============================================
# Part 3: Cluster Extraction
# ============================================

# Fit agglomerative clustering
cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = cluster.fit_predict(X)

# Plot clustering result
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=50)
plt.title("Agglomerative Clustering (Ward)")
plt.grid(True)
plt.show()

# ============================================
# Part 4: Comparing Linkage Methods
# ============================================

methods = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for i, method in enumerate(methods):
    cluster = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = cluster.fit_predict(X)

    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=50)
    axes[i].set_title(f"Agglomerative Clustering ({method.capitalize()} linkage)")
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Compare dendrograms for different linkage methods
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for i, method in enumerate(methods):
    linked = linkage(X, method=method)

    axes[i].set_title(f"Dendrogram ({method.capitalize()} linkage)")
    dendrogram(linked, ax=axes[i])
    axes[i].set_xlabel("Sample index")
    axes[i].set_ylabel("Distance")
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# ============================================
# Part 5: Iris Dataset (2D Projection with PCA)
# ============================================

# Load and scale data
iris = load_iris()
X_iris = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply clustering
cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = cluster.fit_predict(X_scaled)

# Plot clustering in 2D
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', s=50)
plt.title("Hierarchical Clustering on Iris (PCA projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Dendrogram for Iris
linked_iris = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked_iris)
plt.title("Hierarchical Clustering Dendrogram - Iris Dataset")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# ============================================
# Part 7: Going Further - Using fcluster
# ============================================

# Cut dendrogram at specific distance
distance_threshold = 10
clusters_at_distance = fcluster(linked, distance_threshold, criterion='distance')

plt.scatter(X[:, 0], X[:, 1], c=clusters_at_distance, cmap='rainbow', s=50)
plt.title(f"Clusters by cutting dendrogram at distance={distance_threshold}")
plt.grid(True)
plt.show()

print(f"Number of clusters at distance {distance_threshold}: {len(set(clusters_at_distance))}")

# Try different distance thresholds
distances = [5, 10, 15, 20]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for i, dist in enumerate(distances):
    clusters = fcluster(linked, dist, criterion='distance')
    n_clusters = len(set(clusters))

    axes[i].scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow', s=50)
    axes[i].set_title(f"Distance threshold={dist} ({n_clusters} clusters)")
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# ============================================
# Comparison: K-Means vs DBSCAN vs Hierarchical
# ============================================

from sklearn.cluster import KMeans, DBSCAN

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
labels_kmeans = kmeans.fit_predict(X)
axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=50)
axes[0].set_title("K-Means")
axes[0].grid(True)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='plasma', s=50)
axes[1].set_title("DBSCAN")
axes[1].grid(True)

# Hierarchical
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hierarchical = hierarchical.fit_predict(X)
axes[2].scatter(X[:, 0], X[:, 1], c=labels_hierarchical, cmap='rainbow', s=50)
axes[2].set_title("Hierarchical (Ward)")
axes[2].grid(True)

plt.tight_layout()
plt.show()

print("Lab completed!")