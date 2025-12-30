# -----------------------------------------
# Lab 10: Introductory Gradient Descent Lab
# Function: f(x) = (x - 3)^2
# Gradient: f'(x) = 2(x - 3)
# -----------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Function to minimize
def f(x):
    return (x - 3) ** 2

# Gradient of the function
def gradient(x):
    return 2 * (x - 3)

# Gradient descent algorithm
def gradient_descent(starting_point, learning_rate, num_iterations):
    x = float(starting_point)
    history = [x]
    for _ in range(num_iterations):
        x = x - learning_rate * gradient(x)
        history.append(x)
    return x, history


# --------------------------------------------------
# Common parameters
# --------------------------------------------------
starting_point = 10
num_iterations = 25
x_values = np.linspace(0, 10, 400)
y_values = f(x_values)

# --------------------------------------------------
# Learning rate experiments
# --------------------------------------------------
learning_rates = [0.1, 0.01, 0.8]

for learning_rate in learning_rates:
    final_x, history = gradient_descent(starting_point, learning_rate, num_iterations)

    # Plot function + iteration trajectory
    plt.plot(x_values, y_values, label='f(x) = (x - 3)^2')
    plt.scatter(history, [f(x) for x in history], color='red', label='Iterations')
    plt.title(f'Gradient Descent (learning rate = {learning_rate})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print results
    print(f"[LR = {learning_rate}] Final x: {final_x}")
    print(f"[LR = {learning_rate}] Final f(x): {f(final_x)}")

    # Plot error decrease
    errors = [f(x) for x in history]
    plt.plot(errors)
    plt.title(f'Decrease of f(x) per iteration (LR = {learning_rate})')
    plt.xlabel('Iterations')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()
