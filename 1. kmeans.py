# ============================================
# K-Means Clustering Algorithm
# ============================================

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================
# Part 2: K-Means on Synthetic 2D Data
# ============================================

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Generated 2D data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# Apply KMeans
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot with cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.7, marker='X', label='Centroids')
plt.title("K-Means Clustering Result")
plt.legend()
plt.grid(True)
plt.show()

# ============================================
# Part 3: Choosing the Right Number of Clusters (Elbow Method)
# ============================================
inertias = []
K_values = range(1, 10)

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(K_values, inertias, marker='o')
plt.title("Elbow Method to Determine Optimal K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.grid(True)
plt.show()

# ============================================
# Part 4: K-Means on the Iris Dataset
# ============================================

# Load data
iris = load_iris()
X = iris.data
y_true = iris.target

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(X_scaled)

# Plot the clusters in 2D PCA space
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', s=50)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.title("K-Means Clustering on Iris Dataset (PCA view)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()

# ============================================
# Part 6: Going Further - Non-spherical data
# ============================================
from sklearn.datasets import make_moons, make_circles

# Test on moon-shaped data
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

kmeans_moons = KMeans(n_clusters=2, random_state=0)
y_moons = kmeans_moons.fit_predict(X_moons)

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, s=50, cmap='viridis')
centers_moons = kmeans_moons.cluster_centers_
plt.scatter(centers_moons[:, 0], centers_moons[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.title("K-Means on Non-Spherical Data (Moons)")
plt.legend()
plt.grid(True)
plt.show()

# Test on circular data
X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

kmeans_circles = KMeans(n_clusters=2, random_state=0)
y_circles = kmeans_circles.fit_predict(X_circles)

plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, s=50, cmap='viridis')
centers_circles = kmeans_circles.cluster_centers_
plt.scatter(centers_circles[:, 0], centers_circles[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.title("K-Means on Non-Spherical Data (Circles)")
plt.legend()
plt.grid(True)
plt.show()

# ============================================
# Silhouette Score for cluster evaluation
# ============================================

from sklearn.metrics import silhouette_score

# Calculate silhouette scores for different k values
silhouette_scores = []
K_values = range(2, 10)

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    y_pred = kmeans.fit_predict(X)
    score = silhouette_score(X, y_pred)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.plot(K_values, silhouette_scores, marker='o', color='orange')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
