# ============================================
# TP REGRESSION
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

# ============================================
# PART 1: SIMPLE LINEAR REGRESSION
# ============================================

print("=" * 50)
print("SIMPLE LINEAR REGRESSION")
print("=" * 50)

# 1. Load the dataset
# Note: You need to upload salary-data.csv
df_salary = pd.read_csv('salary-data.csv')

# Display basic characteristics
print(f"\nNumber of rows: {df_salary.shape[0]}")
print(f"Number of columns: {df_salary.shape[1]}")
print("\nFirst few rows:")
print(df_salary.head())
print("\nDataset info:")
print(df_salary.info())

# 2. Basic statistics
print("\nBasic statistics:")
print(df_salary.describe())

# Correlation coefficient
correlation = df_salary.corr()
print(f"\nCorrelation matrix:\n{correlation}")

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()

# 3. Build predictive model
X = df_salary.iloc[:, :-1].values  # Experience (all columns except last)
y = df_salary.iloc[:, -1].values  # Salary (last column)

# Split dataset: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictions
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# Calculate MSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"\nMSE (Training): {mse_train:.2f}")
print(f"MSE (Test): {mse_test:.2f}")

# Calculate R² and Adjusted R²
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

n_train = X_train.shape[0]
n_test = X_test.shape[0]
p = X_train.shape[1]  # number of features

adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)

print(f"\nR² (Training): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"Adjusted R² (Training): {adj_r2_train:.4f}")
print(f"Adjusted R² (Test): {adj_r2_test:.4f}")

# Visualize predictions vs actual
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Training set
ax1.scatter(X_train, y_train, color='blue', label='Actual')
ax1.plot(X_train, y_train_pred, color='red', linewidth=2, label='Predicted')
ax1.set_title('Training Set: Predictions vs Actual')
ax1.set_xlabel('Experience (years)')
ax1.set_ylabel('Salary')
ax1.legend()
ax1.grid(True)

# Test set
ax2.scatter(X_test, y_test, color='blue', label='Actual')
ax2.plot(X_test, y_test_pred, color='red', linewidth=2, label='Predicted')
ax2.set_title('Test Set: Predictions vs Actual')
ax2.set_xlabel('Experience (years)')
ax2.set_ylabel('Salary')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 4. Analyze normality of errors
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

print(f"\nMean of residuals (Training): {residuals_train.mean():.4f}")
print(f"Mean of residuals (Test): {residuals_test.mean():.4f}")

# Plot residual distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1.hist(residuals_test, bins=20, edgecolor='black', alpha=0.7)
ax1.set_title('Distribution of Residuals (Test Set)')
ax1.set_xlabel('Residuals')
ax1.set_ylabel('Frequency')
ax1.grid(True)

# Q-Q plot (to check normality)
stats.probplot(residuals_test, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot (Test Set)')
ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# PART 2: MULTIPLE LINEAR REGRESSION
# ============================================

print("\n" + "=" * 50)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 50)

# Load dataset
df_startups = pd.read_csv('50_Startups.csv')

print(f"\nNumber of rows: {df_startups.shape[0]}")
print(f"Number of columns: {df_startups.shape[1]}")
print("\nFirst few rows:")
print(df_startups.head())
print("\nDataset info:")
print(df_startups.info())

# Visualize variables
print("\nBasic statistics:")
print(df_startups.describe())

# Prepare data
X = df_startups.iloc[:, :-1].values  # All columns except last
y = df_startups.iloc[:, -1].values  # Profit (last column)

# Handle categorical variable (State column - index 3)
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train model
regressor_multi = LinearRegression()
regressor_multi.fit(X_train, y_train)

# Predictions
y_train_pred = regressor_multi.predict(X_train)
y_test_pred = regressor_multi.predict(X_test)

# Calculate metrics
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

n_train = X_train.shape[0]
n_test = X_test.shape[0]
p = X_train.shape[1]

adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)

print(f"\nMSE (Training): {mse_train:.2f}")
print(f"MSE (Test): {mse_test:.2f}")
print(f"R² (Training): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"Adjusted R² (Training): {adj_r2_train:.4f}")
print(f"Adjusted R² (Test): {adj_r2_test:.4f}")

# Predict for a specific input
sample_input = [[1, 0, 0, 130000, 140000, 300000]]  # Note: 3 columns for one-hot encoded State
prediction = regressor_multi.predict(sample_input)
print(f"\nPrediction for input {sample_input}: ${prediction[0]:,.2f}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
plt.scatter(range(len(y_test_pred)), y_test_pred, color='red', label='Predicted', alpha=0.6)
plt.title('Multiple Regression: Predictions vs Actual (Test Set)')
plt.xlabel('Sample Index')
plt.ylabel('Profit')
plt.legend()
plt.grid(True)
plt.show()

# Residual analysis
residuals = y_test - y_test_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(residuals, bins=15, edgecolor='black', alpha=0.7)
ax1.set_title('Distribution of Residuals')
ax1.set_xlabel('Residuals')
ax1.set_ylabel('Frequency')
ax1.grid(True)

stats.probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot')
ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# PART 3: POLYNOMIAL REGRESSION - Part 1
# ============================================

print("\n" + "=" * 50)
print("POLYNOMIAL REGRESSION - Position Salaries")
print("=" * 50)

# Load dataset
df_position = pd.read_csv('Position_Salaries.csv')

print("\nFirst few rows:")
print(df_position.head())

# Prepare data
X = df_position.iloc[:, 1:2].values  # Position level
y = df_position.iloc[:, 2].values  # Salary

# Visualize data
plt.scatter(X, y, color='blue')
plt.title('Position Level vs Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

# Try linear regression first
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

r2_lin = r2_score(y, y_pred_lin)
print(f"\nLinear Regression R²: {r2_lin:.4f}")

# Plot linear regression
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred_lin, color='red', label='Linear Regression')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()

# Try polynomial regression with different degrees
degrees = [2, 3, 4, 5, 6]
results = []

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for i, degree in enumerate(degrees):
    # Create polynomial features
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)

    # Train model
    lin_reg_poly = LinearRegression()
    lin_reg_poly.fit(X_poly, y)

    # Predictions
    y_pred_poly = lin_reg_poly.predict(X_poly)

    # Calculate metrics
    r2 = r2_score(y, y_pred_poly)
    n = X.shape[0]
    p = degree
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    results.append({'Degree': degree, 'R²': r2, 'Adjusted R²': adj_r2})

    print(f"\nDegree {degree}:")
    print(f"  R²: {r2:.4f}")
    print(f"  Adjusted R²: {adj_r2:.4f}")

    # Plot
    X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_smooth_poly = poly_reg.transform(X_smooth)
    y_smooth_pred = lin_reg_poly.predict(X_smooth_poly)

    axes[i].scatter(X, y, color='blue', label='Actual')
    axes[i].plot(X_smooth, y_smooth_pred, color='red', label=f'Degree {degree}')
    axes[i].set_title(f'Polynomial Regression (Degree {degree})\nR²={r2:.4f}')
    axes[i].set_xlabel('Position Level')
    axes[i].set_ylabel('Salary')
    axes[i].legend()
    axes[i].grid(True)

# Hide last subplot
axes[-1].axis('off')

plt.tight_layout()
plt.show()

# Summary table
results_df = pd.DataFrame(results)
print("\nSummary of Polynomial Regression Results:")
print(results_df)

# ============================================
# PART 3: POLYNOMIAL REGRESSION - Part 2 (Simulated)
# ============================================

print("\n" + "=" * 50)
print("POLYNOMIAL REGRESSION - Simulated Data")
print("=" * 50)

# Generate simulated data
np.random.seed(2)
loading_time = np.random.normal(3.0, 1.0, 1000)
purchase_amount = np.random.normal(50.0, 10.0, 1000) / (loading_time * loading_time)

# 1. Display distributions
print(f"\nLoading time - Mean: {loading_time.mean():.2f}, Std: {loading_time.std():.2f}")
print(f"Purchase amount - Mean: {purchase_amount.mean():.2f}, Std: {purchase_amount.std():.2f}")

# 2. Plot data in 2D
plt.figure(figsize=(10, 6))
plt.scatter(loading_time, purchase_amount, alpha=0.5)
plt.title('Page Loading Time vs Purchase Amount')
plt.xlabel('Loading Time (seconds)')
plt.ylabel('Purchase Amount (€)')
plt.grid(True)
plt.show()

# 3. Apply polynomial regression
X_sim = loading_time.reshape(-1, 1)
y_sim = purchase_amount

# Try different polynomial degrees
degrees_sim = [2, 3, 4, 5, 6]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for i, degree in enumerate(degrees_sim):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X_sim)

    lin_reg_poly = LinearRegression()
    lin_reg_poly.fit(X_poly, y_sim)

    y_pred_poly = lin_reg_poly.predict(X_poly)

    r2 = r2_score(y_sim, y_pred_poly)

    print(f"\nDegree {degree}: R² = {r2:.4f}")

    # Plot
    X_sorted = np.sort(X_sim, axis=0)
    X_sorted_poly = poly_reg.transform(X_sorted)
    y_sorted_pred = lin_reg_poly.predict(X_sorted_poly)

    axes[i].scatter(X_sim, y_sim, alpha=0.3, s=10)
    axes[i].plot(X_sorted, y_sorted_pred, color='red', linewidth=2)
    axes[i].set_title(f'Polynomial Degree {degree}\nR²={r2:.4f}')
    axes[i].set_xlabel('Loading Time (s)')
    axes[i].set_ylabel('Purchase Amount (€)')
    axes[i].grid(True)

axes[-1].axis('off')

plt.tight_layout()
plt.show()

# 4. Discussion
print("\n" + "=" * 50)
print("DISCUSSION")
print("=" * 50)
print("""
The relationship is inverse quadratic: y = k / x²

Observations:
- Polynomial regression tries to fit this curve but may not be ideal
- The curve has an asymptotic behavior (approaches infinity as x→0)
- Higher degree polynomials may overfit

Better alternatives:
1. Transform the data: Use 1/x² as a feature
2. Non-linear regression models
3. Exponential or power law models
""")

# Try transformation approach
X_transformed = 1 / (loading_time ** 2)
X_transformed = X_transformed.reshape(-1, 1)

lin_reg_transformed = LinearRegression()
lin_reg_transformed.fit(X_transformed, y_sim)
y_pred_transformed = lin_reg_transformed.predict(X_transformed)

r2_transformed = r2_score(y_sim, y_pred_transformed)
print(f"\nUsing 1/x² transformation: R² = {r2_transformed:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(loading_time, purchase_amount, alpha=0.3, label='Actual')
plt.scatter(loading_time, y_pred_transformed, alpha=0.3, color='red', label='Predicted (1/x² model)')
plt.title(f'Linear Regression with 1/x² Transformation\nR²={r2_transformed:.4f}')
plt.xlabel('Loading Time (s)')
plt.ylabel('Purchase Amount (€)')
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "=" * 50)
print("LAB COMPLETED!")
print("=" * 50)