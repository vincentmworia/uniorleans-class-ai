# ===== 0) Imports =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

from matplotlib.colors import ListedColormap

# For cleaner plots
sns.set_theme(style="whitegrid", context="notebook")

# ===== 1) Load & inspect data =====
# Ensure the CSV is in the same directory as this script/notebook
df = pd.read_csv("Data/Social_Network_Ads.csv")

print("\n=== Head ===")
print(df.head())
print("\n=== Info ===")
print(df.info())
print("\n=== Missing values per column ===")
print(df.isna().sum())

# ===== 2) Select features X and target y =====
# Typical columns in this dataset: ['User ID','Gender','Age','EstimatedSalary','Purchased']
# We'll use Age & EstimatedSalary to predict Purchased.
if {"Age", "EstimatedSalary", "Purchased"}.issubset(df.columns):
    X = df[["Age", "EstimatedSalary"]].values
    y = df["Purchased"].values
else:
    # Fallback to positional indexing if column names differ:
    # Use columns 2 & 3 for X and column 4 for y (0-based indexing).
    X = df.iloc[:, [2, 3]].values
    y = df.iloc[:, 4].values

# ===== 3) Train/test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

# ===== 4) (Optional but recommended) Standardize features =====
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ===== 5) Train logistic regression =====
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X_train_std, y_train)

# ===== 6) Predict on test data =====
y_pred = clf.predict(X_test_std)
y_proba = clf.predict_proba(X_test_std)[:, 1]

print("\n=== Accuracy ===")
print(f"{accuracy_score(y_test, y_pred):.4f}")

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred))

# ===== 7) Confusion matrix (numbers + heatmap) =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4.8, 4.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ===== 8) Decision boundary visualization =====
# Helper to plot boundary for a given (unscaled) X_set and y_set
def plot_decision_boundary(X_set, y_set, title):
    # Build a grid in ORIGINAL feature space
    x1_min, x1_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
    x2_min, x2_max = X_set[:, 1].min() - 1000, X_set[:, 1].max() + 1000  # salary ranges are big
    X1, X2 = np.meshgrid(
        np.arange(start=x1_min, stop=x1_max, step=0.1),
        np.arange(start=x2_min, stop=x2_max, step=0.1 * 1000),  # keep step ~0.1 but scaled for salary
    )

    # Prepare grid for prediction: scale using the same scaler as training
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    grid_points_std = scaler.transform(grid_points)
    Z = clf.predict(grid_points_std).reshape(X1.shape)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(
        X1, X2, Z,
        alpha=0.25,
        cmap=ListedColormap(("red", "green"))
    )
    # Scatter the actual points
    plt.scatter(
        X_set[:, 0], X_set[:, 1],
        c=y_set, edgecolor="k",
        cmap=ListedColormap(("red", "green"))
    )
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Training set boundary
plot_decision_boundary(X_train, y_train, "Logistic Regression — Decision Boundary (Training Set)")

# Test set boundary
plot_decision_boundary(X_test, y_test, "Logistic Regression — Decision Boundary (Test Set)")

# ===== 9) ROC curve =====
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(5.6, 4.6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# (Optional) If you want the AUC computed from hard labels (not recommended):
auc_from_labels = roc_auc_score(y_test, y_pred)
print(f"\nAUC (using predicted probabilities): {auc_score:.4f}")
print(f"AUC (using hard labels only):        {auc_from_labels:.4f}")
