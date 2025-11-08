# ============================================
# Lab: Discovering Density-Based Clustering with DBSCAN
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_iris
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================
# Part 2: DBSCAN on Synthetic 2D Data (Non-Spherical)
# ============================================

# Generate data
X, y = make_moons(n_samples=300, noise=0.05, random_state=0)

# Visualize raw data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Generated 2D data (non-spherical)")
plt.grid(True)
plt.show()

# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# Visualize clustering result
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=50)
plt.title("DBSCAN Clustering Result")
plt.grid(True)
plt.show()

# ============================================
# Part 3: Playing with Parameters
# ============================================

# Try different eps values
eps_values = [0.1, 0.2, 0.3, 0.5]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, eps in enumerate(eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X)

    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=50)
    axes[i].set_title(f"DBSCAN with eps={eps}")
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Try different min_samples values
min_samples_values = [3, 5, 10, 20]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, min_samp in enumerate(min_samples_values):
    dbscan = DBSCAN(eps=0.2, min_samples=min_samp)
    labels = dbscan.fit_predict(X)

    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=50)
    axes[i].set_title(f"DBSCAN with min_samples={min_samp}")
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# ============================================
# Part 4: Comparison with K-Means
# ============================================

# K-Means on same data
kmeans = KMeans(n_clusters=2, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
ax1.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='plasma', s=50)
ax1.set_title("DBSCAN on Non-Spherical Data")
ax1.grid(True)

# K-Means
ax2.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
centers = kmeans.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Centroids')
ax2.set_title("K-Means on Non-Spherical Data")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# Part 5: DBSCAN on the Iris Dataset (with PCA)
# ============================================

# Load and normalize data
iris = load_iris()
X_iris = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# Project to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply DBSCAN
dbscan_iris = DBSCAN(eps=0.6, min_samples=5)
labels_iris = dbscan_iris.fit_predict(X_scaled)

# Plot clustering result
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_iris, cmap='rainbow', s=50)
plt.title("DBSCAN on Iris Dataset (PCA projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Count noise points
n_noise = list(labels_iris).count(-1)
n_clusters = len(set(labels_iris)) - (1 if -1 in labels_iris else 0)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# ============================================
# Part 7: Going Further - Varying Density
# ============================================

from sklearn.datasets import make_blobs

# Create data with varying density
X_var, y_var = make_blobs(n_samples=[100, 50, 200],
                          centers=[[0, 0], [5, 5], [10, 2]],
                          cluster_std=[0.5, 1.5, 0.3],
                          random_state=42)

# Apply DBSCAN
dbscan_var = DBSCAN(eps=0.8, min_samples=5)
labels_var = dbscan_var.fit_predict(X_var)

plt.scatter(X_var[:, 0], X_var[:, 1], c=labels_var, cmap='plasma', s=50)
plt.title("DBSCAN on Data with Varying Density")
plt.grid(True)
plt.show()

print("Lab completed!")