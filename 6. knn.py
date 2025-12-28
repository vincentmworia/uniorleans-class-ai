# =========================
# K-Nearest Neighbors (KNN)
# =========================

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1) Data Generation

# Number of classes
n_classes = 3

# Generate synthetic dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=n_classes,
    n_clusters_per_class=1,
    random_state=42
)

# Visualize generated data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)
plt.title(f"Generated Data for Classification with {n_classes} Classes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Split data into training and test sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2) Applying KNN

# Create KNN model with k = 5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of KNN model with {n_classes} classes: {accuracy:.2f}")

# Visualize predictions
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', s=50)
plt.title(f"KNN Predictions on Test Set ({n_classes} Classes)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 3) Tuning k

k_values = [1, 3, 5, 7, 9]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"k = {k}, Accuracy = {accuracy:.2f}")

# Plot accuracy vs k
plt.plot(k_values, accuracies, marker='o')
plt.title(f"Impact of Number of Neighbors on Accuracy ({n_classes} Classes)")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
