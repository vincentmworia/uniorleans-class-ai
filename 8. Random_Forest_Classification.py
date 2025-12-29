from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="class")

print("X shape:", X.shape)
print("Classes:", data.target_names)

# 2) Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 3) Train Random Forest model
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    oob_score=True,
    max_depth=None
)
rf.fit(X_train, y_train)

# 4) Predictions
y_pred = rf.predict(X_test)

# 5) Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("OOB Score:", rf.oob_score_)
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 6) Confusion matrix (heatmap)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.title("Confusion Matrix - Random Forest (Wine)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# 7) Feature importance (Gini-based)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 4))
importances.plot(kind="bar", title="Feature Importance (Gini-based)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
