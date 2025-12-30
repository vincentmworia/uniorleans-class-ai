# -----------------------------------------
# Lab 12 – Neural Networks
# -----------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# For repeatability
np.random.seed(0)
tf.random.set_seed(0)

# --------------- PART 1: CHURN MODELLING (Binary classification) ---------------

dataset = pd.read_csv("Data/Churn_Modelling.csv")

# X = columns 3..12, y = column 13 (same as manual)
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13].values

# Encode categorical + scale continuous
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

preprocess = make_column_transformer(
    (OneHotEncoder(), ["Geography", "Gender"]),
    (StandardScaler(), ["CreditScore", "Age", "Tenure", "Balance",
                        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"])
)

X = preprocess.fit_transform(X)

# Delete dummy columns (manual)
X = np.delete(X, [0, 3], 1)

# Train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# --------------- PART 2: BASELINE ANN ---------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Dense(units=6, activation="relu", input_dim=11))
classifier.add(Dense(units=6, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

classifier.compile(optimizer="adam",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])

history = classifier.fit(X_train, y_train,
                          batch_size=10,
                          epochs=100,
                          verbose=0)

plt.figure()
plt.plot(history.history["loss"])
plt.title("Churn - Loss (Baseline ANN)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# --------------- PART 3: PREDICTIONS + CONFUSION MATRIX ---------------

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

y_pred = (classifier.predict(X_test, verbose=0) > 0.5)

print("\nChurn (Baseline) Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Single new prediction (manual example)
Xnew = pd.DataFrame({
    "CreditScore": [600],
    "Geography": ["France"],
    "Gender": ["Male"],
    "Age": [40],
    "Tenure": [3],
    "Balance": [60000],
    "NumOfProducts": [2],
    "HasCrCard": [1],
    "IsActiveMember": [1],
    "EstimatedSalary": [50000],
})

Xnew = preprocess.transform(Xnew)
Xnew = np.delete(Xnew, [0, 3], 1)

print("Single customer prediction:",
      bool((classifier.predict(Xnew, verbose=0) > 0.5)[0][0]))

# --------------- TASK 2: OTHER ARCHITECTURES ---------------

# Small ANN
classifier_small = Sequential()
classifier_small.add(Dense(4, activation="relu", input_dim=11))
classifier_small.add(Dense(1, activation="sigmoid"))
classifier_small.compile(optimizer="adam",
                          loss="binary_crossentropy",
                          metrics=["accuracy"])
classifier_small.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)

print("\nSmall ANN Accuracy:",
      accuracy_score(y_test, classifier_small.predict(X_test, verbose=0) > 0.5))

# Big ANN
classifier_big = Sequential()
classifier_big.add(Dense(12, activation="relu", input_dim=11))
classifier_big.add(Dense(6, activation="relu"))
classifier_big.add(Dense(1, activation="sigmoid"))
classifier_big.compile(optimizer="adam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"])
classifier_big.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)

print("Big ANN Accuracy:",
      accuracy_score(y_test, classifier_big.predict(X_test, verbose=0) > 0.5))

# --------------- TASK 3: DROPOUT (manual example) ---------------

classifier_dropout = Sequential()
classifier_dropout.add(Dense(128, activation="relu", input_dim=11))
classifier_dropout.add(Dropout(0.2))
classifier_dropout.add(Dense(1, activation="sigmoid"))

classifier_dropout.compile(optimizer="adam",
                            loss="binary_crossentropy",
                            metrics=["accuracy"])
classifier_dropout.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

print("\nDropout ANN Accuracy:",
      accuracy_score(y_test, classifier_dropout.predict(X_test, verbose=0) > 0.5))

# --------------- (MANUAL EXTRA) CROSS-VALIDATION + GRID SEARCH ---------------

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

def build_classifier(optimizer="adam"):
    model = Sequential()
    model.add(Dense(6, activation="relu", input_dim=11))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

classifier_cv = KerasClassifier(model=build_classifier,
                                epochs=100,
                                batch_size=10,
                                verbose=0)

accuracies = cross_val_score(classifier_cv, X_train, y_train, cv=10, n_jobs=1)
print("\nCross-val mean accuracy:", accuracies.mean())
print("Cross-val std:", accuracies.std())

classifier_gs = KerasClassifier(model=build_classifier, verbose=0)

parameters = {
    "batch_size": [25, 32],
    "epochs": [100, 500],
    "optimizer": ["adam", "rmsprop"]
}

grid_search = GridSearchCV(classifier_gs, parameters, cv=10, scoring="accuracy")
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

# --------------- TASK 4: IRIS NEURAL NETWORK (Multiclass) ---------------

from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_iris = StandardScaler().fit_transform(X_iris)

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=41, stratify=y_iris
)

iris_model = Sequential()
iris_model.add(Dense(8, activation="relu", input_dim=4))
iris_model.add(Dense(8, activation="relu"))
iris_model.add(Dense(3, activation="softmax"))

iris_model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

history_iris = iris_model.fit(X_train_i, y_train_i,
                              epochs=150,
                              batch_size=8,
                              verbose=0)

plt.figure()
plt.plot(history_iris.history["loss"])
plt.title("Iris - Loss (Neural Network)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

y_pred_i = np.argmax(iris_model.predict(X_test_i, verbose=0), axis=1)

print("\nIris Accuracy:", accuracy_score(y_test_i, y_pred_i))
print("Iris Confusion Matrix:\n", confusion_matrix(y_test_i, y_pred_i))
print("Iris Classification Report:\n",
      classification_report(y_test_i, y_pred_i, target_names=iris.target_names))
