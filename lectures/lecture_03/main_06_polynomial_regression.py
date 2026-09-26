# ============================================================================
#   Polynomial Regression
# ============================================================================
#
#   * Polynomial regression extends linear regression using higher-order
#     terms of the input features.
#
#   * These additional terms allow the model to capture nonlinear
#     relationships in the data.
#
#   * More complex polynomial models can improve fit but may increase the
#     risk of overfitting.
#
# ============================================================================

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

df = pd.read_csv(os.path.join("datasets", "dataset_protein_yield.csv"))

features = df.columns[:-1]
targets = df.columns[-1]

x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

# We can add the PolynomialFeatures class to our pipeline.
# With degree=2, this becomes a 2nd-order polynomial.
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

cv_results = cross_validate(model, x, y, cv=5, scoring=("neg_root_mean_squared_error", "r2"), return_train_score=True)

ms_train_rmse = -cv_results["train_neg_root_mean_squared_error"]
ms_train_r2 = cv_results["train_r2"]
ms_test_rmse = -cv_results["test_neg_root_mean_squared_error"]
ms_test_r2 = cv_results["test_r2"]

print()
print(f"  Train | RMSE | {np.mean(ms_train_rmse):6.1f} +/- {np.std(ms_train_rmse):6.1f} mg/L")
print(f"  Test  | RMSE | {np.mean(ms_test_rmse):6.1f} +/- {np.std(ms_test_rmse):6.1f} mg/L")
print(f"  Train | R^2  | {np.mean(ms_train_r2):6.3f} +/- {np.std(ms_train_r2):6.3f}")
print(f"  Test  | R^2  | {np.mean(ms_test_r2):6.3f} +/- {np.std(ms_test_r2):6.3f}")
print()
