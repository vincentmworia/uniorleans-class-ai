# TP 14 — MNIST Classification with Autoencoders (Keras)

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical
from sklearn.manifold import TSNE

# ----------------------------
# Part 1: Load and preprocess
# ----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize [0,1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Flatten 28x28 -> 784
x_train = x_train.reshape((x_train.shape[0], -1))
x_test  = x_test.reshape((x_test.shape[0], -1))

# Use a subset to speed up training
subset_size = 10000
x_train_subset = x_train[:subset_size]
y_train_subset = y_train[:subset_size]

# --------------------------------
# Part 2: Build and train autoencoder
# --------------------------------
input_img = Input(shape=(784,))
encoded = Dense(128, activation="relu")(input_img)
encoded = Dense(64, activation="relu")(encoded)
latent  = Dense(32, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(latent)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dense(784, activation="sigmoid")(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, latent)

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(
    x_train_subset, x_train_subset,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# --------------------------------
# Part 3: Classify using latent space
# --------------------------------
y_train_cat = to_categorical(y_train_subset, 10)
y_test_cat  = to_categorical(y_test, 10)

input_latent = Input(shape=(32,))
x = Dense(64, activation="relu")(input_latent)
output = Dense(10, activation="softmax")(x)

classifier_model = Model(input_latent, output)
classifier_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

latent_train = encoder.predict(x_train_subset, verbose=0)
latent_test  = encoder.predict(x_test, verbose=0)

classifier_model.fit(
    latent_train, y_train_cat,
    epochs=10,
    batch_size=256,
    validation_data=(latent_test, y_test_cat)
)

# Evaluate classifier
test_loss, test_acc = classifier_model.evaluate(latent_test, y_test_cat, verbose=0)
print("Classifier Test loss:", test_loss)
print("Classifier Test accuracy:", test_acc)

# ----------------------------
# Part 4: t-SNE visualization
# ----------------------------
latent_2d = TSNE(n_components=2).fit_transform(latent_test)

plt.figure(figsize=(8, 6))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_test, cmap="jet", s=5)
plt.colorbar()
plt.title("t-SNE projection of latent space")
plt.show()

# ----------------------------
# Part 5: Random predictions
# ----------------------------
n = 10
indices = np.random.choice(len(x_test), n, replace=False)

plt.figure(figsize=(12, 3))
for i, idx in enumerate(indices):
    # Show original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.axis("off")

    # Predict and show probabilities
    pred = classifier_model.predict(latent_test[idx:idx+1], verbose=0)
    ax = plt.subplot(2, n, i + 1 + n)
    plt.bar(np.arange(10), pred[0])
    plt.title(f"Pred: {np.argmax(pred)}")
    plt.xticks(np.arange(10))

plt.tight_layout()
plt.show()
