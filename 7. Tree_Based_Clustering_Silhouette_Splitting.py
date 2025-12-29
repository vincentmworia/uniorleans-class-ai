# -----------------------------------------
# Lab: Tree-Based Clustering — Building Unsupervised Trees Using Silhouette-Based Splitting
# -----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


# ----------------------------
# 1) Data generation + plot
# ----------------------------
def make_data(n_samples=300, centers=4, n_features=2, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
    )
    return X


def plot_points(X, title, c=None, cmap=None):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=c, cmap=cmap, s=20)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# -----------------------------------------
# 2) Best split using Silhouette score
# -----------------------------------------
def best_split(X, min_leaf_size=5):
    """
    Search all features and candidate thresholds.
    For each candidate (feature, threshold), create binary labels:
        left: X[:, feature] <= threshold
        right: X[:, feature] > threshold
    Score split by silhouette_score(X, labels) and return the best.
    """
    n_samples, n_features = X.shape
    best_score = -1.0
    best_feature = None
    best_threshold = None

    for f in range(n_features):
        thresholds = np.unique(X[:, f])
        for t in thresholds:
            left_mask = X[:, f] <= t
            right_mask = ~left_mask

            if left_mask.sum() < min_leaf_size or right_mask.sum() < min_leaf_size:
                continue

            labels = np.zeros(n_samples, dtype=int)
            labels[right_mask] = 1

            # Valid because we ensured both sides have at least min_leaf_size points
            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_feature = f
                best_threshold = t

    return best_feature, best_threshold, best_score


def visualize_best_split(X, feature, threshold, score):
    colors = (X[:, feature] > threshold).astype(int)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=colors, cmap="coolwarm", s=20)

    if feature == 0:
        plt.axvline(threshold, color="black", linestyle="--")
    else:
        plt.axhline(threshold, color="black", linestyle="--")

    plt.title(f"Best Split (Feature {feature}, Threshold {threshold:.2f}) | Silhouette={score:.3f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# -----------------------------------------
# 3) Build the tree recursively
# -----------------------------------------
class UnsupervisedTree:
    """
    A simple binary unsupervised tree:
    - Each node chooses (feature, threshold) that maximizes Silhouette score
      for the 2-way split, subject to min_leaf_size.
    - Stop if:
        * depth >= max_depth
        * no valid split
        * best silhouette score < min_gain
    """

    def __init__(self, min_leaf_size=10, min_gain=0.01, depth=0, max_depth=3):
        self.min_leaf_size = int(min_leaf_size)
        self.min_gain = float(min_gain)
        self.depth = int(depth)
        self.max_depth = int(max_depth)

        self.is_leaf = False
        self.feature = None
        self.threshold = None
        self.score = None

        self.left = None
        self.right = None

    def fit(self, X):
        # Stop if max depth reached
        if self.depth >= self.max_depth:
            self.is_leaf = True
            return

        # Find best split at this node
        feature, threshold, score = best_split(X, self.min_leaf_size)

        # Stop if no valid split or not enough gain
        if feature is None or score < self.min_gain:
            self.is_leaf = True
            return

        self.feature = feature
        self.threshold = threshold
        self.score = score

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Create children
        self.left = UnsupervisedTree(
            min_leaf_size=self.min_leaf_size,
            min_gain=self.min_gain,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )
        self.right = UnsupervisedTree(
            min_leaf_size=self.min_leaf_size,
            min_gain=self.min_gain,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )

        # Fit children on their subsets
        self.left.fit(X[left_mask])
        self.right.fit(X[right_mask])

    def predict(self, X):
        """
        Predict cluster labels for X by traversing the trained tree.

        IMPORTANT: We assign UNIQUE labels using the binary PATH to the leaf:
        left edge = 0, right edge = 1. The label is the integer formed by this path.
        This avoids label collisions that can happen if labels are based only on depth.
        """
        labels = np.zeros(X.shape[0], dtype=int)

        def recurse(node, idx_sub, code):
            # If leaf (or somehow no split), assign this leaf's code
            if node.is_leaf or node.feature is None:
                labels[idx_sub] = code
                return

            go_right = X[idx_sub, node.feature] > node.threshold

            left_idx = idx_sub[~go_right]
            right_idx = idx_sub[go_right]

            if left_idx.size:
                recurse(node.left, left_idx, (code << 1) | 0)
            if right_idx.size:
                recurse(node.right, right_idx, (code << 1) | 1)

        # Start code at 1 so the first shift operations keep labels > 0 (optional)
        recurse(self, np.arange(X.shape[0]), code=1)

        return labels


# ----------------------------
# Main run
# ----------------------------
def main():
    # Step 1: Create and plot synthetic data
    X = make_data(n_samples=300, centers=4, n_features=2, random_state=42)
    plot_points(X, "Synthetic Data for Unsupervised Tree")

    # Step 2: Find and visualize the single best split
    feature, threshold, score = best_split(X, min_leaf_size=5)
    print(f"Best split on feature {feature}, threshold {threshold:.3f}, silhouette score = {score:.3f}")
    visualize_best_split(X, feature, threshold, score)

    # Step 3: Train full tree and visualize final clustering
    tree = UnsupervisedTree(max_depth=3, min_leaf_size=10, min_gain=0.01)
    tree.fit(X)
    labels = tree.predict(X)

    plot_points(X, "Tree-Based Clustering (Silhouette Splitting)", c=labels, cmap="viridis")


if __name__ == "__main__":
    main()
