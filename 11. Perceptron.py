# -----------------------------------------
# Lab 11: Gradient Descent on a Perceptron
# -----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ----------------------------
# 1) Generate dataset + split
# ----------------------------

# Generate a simple 2D binary classification dataset
x, y = make_classification(
    n_samples=100,
    n_features=2,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=41
)

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=41
)

# Reshape target vectors into column vectors for matrix operations
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# Plot the training data points
def plot_data(x_data, y_data, title):
    plt.figure()
    plt.scatter(
        x_data[:, 0],          # first feature
        x_data[:, 1],          # second feature
        c=y_data.ravel(),      # class labels as colors
        cmap="bwr",
        edgecolor="k",
        s=100
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Visualize the training dataset
plot_data(x_train, y_train, "Initial Training Data")


# ----------------------------
# 2) Perceptron (Pure Python)
# ----------------------------

# Sigmoid activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Derivative of the sigmoid function
def sigmoid_derivative(s):
    return s * (1 - s)


# Mean Squared Error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Derivative of MSE with respect to predictions
def mean_squared_error_derivative(y_true, y_pred):
    return y_pred - y_true


# Initialize weights and bias randomly
weights = np.random.randn(2, 1)
bias = np.random.randn(1)


# Forward propagation
def forward(x_data):
    z = np.dot(x_data, weights) + bias
    return sigmoid(z)


# Backpropagation using gradient descent
def backward(x_data, y_true, y_pred, learning_rate):
    global weights, bias

    error = mean_squared_error_derivative(y_true, y_pred)
    grad = error * sigmoid_derivative(y_pred)

    d_w = np.dot(x_data.T, grad) / x_data.shape[0]
    d_b = np.mean(grad)

    weights -= learning_rate * d_w
    bias -= learning_rate * float(d_b)


# Training parameters
epochs = 1000
learning_rate = 0.1
loss_history = []

# Training loop
for epoch in range(epochs):
    y_pred = forward(x_train)
    loss = mean_squared_error(y_train, y_pred)
    loss_history.append(loss)

    backward(x_train, y_train, y_pred, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Evaluate model on test data
y_test_pred = forward(x_test)
test_loss = mean_squared_error(y_test, y_test_pred)
print(f"Test loss: {test_loss:.4f}")


# Plot loss evolution during training
plt.figure()
plt.plot(loss_history)
plt.title("Loss During Training (Python)")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.show()


# Plot decision boundary
def plot_decision_boundary(x_data, y_data, w, b, title):
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    z = sigmoid(np.dot(grid, w) + b)
    z = z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, z, levels=[0, 0.5, 1], cmap="bwr", alpha=0.2)
    plt.scatter(
        x_data[:, 0],
        x_data[:, 1],
        c=y_data.ravel(),
        cmap="bwr",
        edgecolor="k",
        s=100
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_decision_boundary(
    x_train,
    y_train,
    weights,
    bias,
    "Decision Boundary (Python Perceptron)"
)


# ----------------------------
# 3) Keras Perceptron
# ----------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

'''
Old way (still valid, but produces a warning):
model = Sequential()
model.add(Dense(1, input_dim=2, activation="sigmoid"))
'''

# New preferred way (no warning)
model = Sequential([
    Input(shape=(2,)),          # explicitly define input layer
    Dense(1, activation="sigmoid")
])

# Compile the model with SGD and MSE loss
model.compile(
    optimizer=SGD(learning_rate=0.1),
    loss="mean_squared_error"
)

# Display model architecture
model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=1000, verbose=0)

# Evaluate model on test data
test_loss_keras = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss (Keras): {test_loss_keras:.4f}")


# Plot training loss for Keras model
plt.figure()
plt.plot(history.history["loss"])
plt.title("Loss During Training (Keras)")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.show()


# Plot decision boundary for Keras model
def plot_decision_boundary_keras(x_data, y_data, model, title):
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    z = model.predict(grid, verbose=0)
    z = z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, z, levels=[0, 0.5, 1], cmap="bwr", alpha=0.2)
    plt.scatter(
        x_data[:, 0],
        x_data[:, 1],
        c=y_data.ravel(),
        cmap="bwr",
        edgecolor="k",
        s=100
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_decision_boundary_keras(
    x_train,
    y_train,
    model,
    "Decision Boundary (Keras Perceptron)"
)
