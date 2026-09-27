# ============================================================================
#   Pipelines
# ============================================================================
#
#   * A pipeline combines preprocessing and model training into a single
#     workflow.
#
#   * Pipelines automatically apply the same data transformations learned
#     from the training set to new data.
#
#   * Using pipelines simplifies code, improves reproducibility, and helps
#     reduce common machine learning workflow errors.
#
# ============================================================================

import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
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

# Instead of creating an instance for each class, we can simply make a pipeline.
# Now we can just use the model object we created instead of managing scaler, linreg, and intermediate arrays.

# This:
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Replaces:
# scaler = StandardScaler()
# linreg = LinearRegression()
# scaler.fit(x_train)
# x_train_std = scaler.transform(x_train)
# linreg.fit(x_train_std, y_train)
# y_train_pred = linreg.predict(x_train_std)
# x_test_std = scaler.transform(x_test)
# y_test_pred = linreg.predict(x_test_std)

# Calculate performance metrics for the true values of y vs the predicted values of y
m_train_rmse = root_mean_squared_error(y_train, y_train_pred)
m_train_r2 = r2_score(y_train, y_train_pred)
m_test_rmse = root_mean_squared_error(y_test, y_test_pred)
m_test_r2 = r2_score(y_test, y_test_pred)

# Print the results
print()
print(f"Train | RMSE | {m_train_rmse:6.3} mg/L")
print(f"Test  | RMSE | {m_test_rmse:6.3} mg/L")

print()
print(f"Train | R^2  | {m_train_r2:6.3}")
print(f"Test  | R^2  | {m_test_r2:6.3}")
