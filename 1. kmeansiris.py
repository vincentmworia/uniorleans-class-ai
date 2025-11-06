# ===== Elbow on Iris + auto-pick best K, then fit/plot =====
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1) Load + scale
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) (Optional) PCA for plotting later
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)

# 3) Compute inertia for a range of K
K_values = range(1, 11)
inertias = []
for k in K_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# 4) Auto-detect the elbow (max distance from line between first and last points)
#    This is a common, robust heuristic (no external deps).
x1, y1 = K_values[0], inertias[0]
x2, y2 = K_values[-1], inertias[-1]
# line vector
line_vec = np.array([x2 - x1, y2 - y1], dtype=float)
line_vec_norm = line_vec / np.linalg.norm(line_vec)

distances = []
for k, sse in zip(K_values, inertias):
    p = np.array([k - x1, sse - y1], dtype=float)
    # perpendicular distance magnitude to the baseline
    proj_len = np.dot(p, line_vec_norm)
    proj = proj_len * line_vec_norm
    perp = p - proj
    distances.append(np.linalg.norm(perp))

best_k = K_values[int(np.argmax(distances))]
print(f"[Elbow] Chosen K = {best_k}")

# 5) Plot elbow curve with chosen K highlighted
plt.plot(K_values, inertias, marker='o')
plt.scatter([best_k], [inertias[best_k-1]], s=120, marker='X', color='red', label=f'Chosen K = {best_k}')
plt.title("Elbow Method on Iris (Inertia vs K)")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.legend()
plt.grid(True)
plt.show()

# 6) Fit KMeans with best_k and visualize in 2D PCA space
kmeans_best = KMeans(n_clusters=best_k, n_init=10, random_state=0)
y_best = kmeans_best.fit_predict(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_best, cmap='viridis', s=50)
centers_2d = pca.transform(kmeans_best.cluster_centers_)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.title(f"K-Means on Iris (PCA view) — K = {best_k}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()