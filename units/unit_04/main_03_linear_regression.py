# ============================================================================
#   Linear Regression
# ============================================================================
#
#   * Linear regression models the relationship between input features and
#     a continuous target variable using a linear equation.
#
#   * Model performance is commonly evaluated using RMSE, which measures
#     prediction error, and R^2, which measures the proportion of variance
#     explained by the model.
#
#   * Comparing training and test performance helps identify underfitting
#     and overfitting, two common challenges in machine learning.
#
# ============================================================================

import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import dataset
df = pd.read_csv(os.path.join("datasets", "dataset_protein_yield.csv"))

# Extract feature columns (i.e., all but last column)
features = df.columns[:-1]

# Extract target column (i.e., the last column)
targets = df.columns[-1]

# Convert to numpy
x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create an instance of StandardScaler
scaler = StandardScaler()

# Create an instance of LinearRegression
linreg = LinearRegression()

# Learn the mean and SD of each feature using x_train
scaler.fit(x_train)

# Standardize x_train and x_test using the mean and SD from x_train
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

# Train the Linear Regression model on x_train_std
linreg.fit(x_train_std, y_train)

# Make a prediction using the trained Linear Regression model on x_train_std and x_test_std
y_train_pred = linreg.predict(x_train_std)
y_test_pred = linreg.predict(x_test_std)

# Calculate performance metrics for the true values of y vs the predicted values of y
m_train_rmse = root_mean_squared_error(y_train, y_train_pred)
m_train_r2 = r2_score(y_train, y_train_pred)
m_test_rmse = root_mean_squared_error(y_test, y_test_pred)
m_test_r2 = r2_score(y_test, y_test_pred)

# Print the results
print()
print("RMSE: The best score is 0, the worst score is positive infinity.  Lower is better.")
print("R^2:  The best score is 1, the worst score is negative infinity. Higher is better")
print("      An R^2 < 0 implies that the model is worse than just predicting np.mean(y_test) for every sample.")
print()
print("The metrics for the training set tell us how well the model captured the patterns in the training data.")
print("If the RMSE is high or R^2 is low for the training set,")
print("then our model may be underfitting the data.")
print()
print(f"  Train | RMSE | {m_train_rmse:6.3} mg/L")
print(f"  Train | R^2  | {m_train_r2:6.3}")
print()
print("The metrics for the test set tell us if the model is capable of generalizing to new data.")
print("If the RMSE is much higher or R^2 is much lower for the test set than for the training set,")
print("then our model may be overfitting the data.")
print()
print(f"  Test  | RMSE | {m_test_rmse:6.3} mg/L")
print(f"  Test  | R^2  | {m_test_r2:6.3}")
