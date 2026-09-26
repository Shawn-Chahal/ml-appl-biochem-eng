# ============================================================================
#   Bias-Variance Tradeoff
# ============================================================================
#
#   * Simple models may underfit the data because they have high bias and
#     cannot capture important underlying patterns.
#
#   * Complex models may overfit the data because they have high variance
#     and become overly sensitive to noise in the training data.
#
#   * The goal is to find a model complexity that balances bias and
#     variance to achieve the best performance on new data.
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

print()
print("As we increase the degree of the polynomial, we can see that: ")
print("  * The model performs better on the training set when going from degree=1 to degree=10.")
print("  * The model performs better on the test set when going from degree=1 to degree=2.")
print("    - The model went from underfitting to optimal fit.")
print("  * The model performs worse on the test set when going from degree=2 to degree=10.")
print("    - The model went from optimal fit to overfitting.")
print()
print("+------------------------------------------------------------------+")
print("| degree | RMSE (Train) |  RMSE (Test) | R^2 (Train) |  R^2 (Test) |")
print("|------------------------------------------------------------------|")

for degree in range(1, 11):
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    cv_results = cross_validate(model, x, y, cv=5,
                                scoring=("neg_root_mean_squared_error", "r2"),
                                return_train_score=True)

    ms_train_rmse = -cv_results["train_neg_root_mean_squared_error"]
    ms_train_r2 = cv_results["train_r2"]
    ms_test_rmse = -cv_results["test_neg_root_mean_squared_error"]
    ms_test_r2 = cv_results["test_r2"]

    print(
        f"| {degree:6d} |"
        f" {np.mean(ms_train_rmse):{7}.1f} mg/L |"  # +/- {np.std(ms_train_rmse):{4}.1f} |"
        f" {np.mean(ms_test_rmse):{7}.1f} mg/L |"  # +/- {np.std(ms_test_rmse):{7}.1f} |"
        f" {np.mean(ms_train_r2):{11}.3f} |"  # +/- {np.std(ms_train_r2):{6}.3f} |"
        f" {np.mean(ms_test_r2):{11}.3f} |"  # +/- {np.std(ms_test_r2):{11}.3f} |"
    )

print("+------------------------------------------------------------------+")
