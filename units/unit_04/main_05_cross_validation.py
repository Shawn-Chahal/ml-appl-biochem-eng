# ============================================================================
#   Cross-Validation
# ============================================================================
#
#   * Cross-validation evaluates a model by repeatedly training and testing
#     it on different subsets of the data.
#
#   * Performance metrics from multiple folds provide a more reliable
#     estimate of model performance than a single train-test split.
#
#   * The mean and standard deviation of the cross-validation scores help
#     quantify both model performance and variability, respectively.
#
# ============================================================================

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(os.path.join("datasets", "dataset_protein_yield.csv"))

features = df.columns[:-1]
targets = df.columns[-1]

x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

model = make_pipeline(StandardScaler(), LinearRegression())

# We can perform cross-validation quite easily using the cross_validate() function.
# This replaces all the steps of splitting, fitting, predicting, and evaluating.
# This function is convenient if the main objective is to evaluate performance.

cv_results = cross_validate(model, x, y, cv=5, scoring=("neg_root_mean_squared_error", "r2"), return_train_score=True)

ms_train_rmse = -cv_results["train_neg_root_mean_squared_error"]
ms_train_r2 = cv_results["train_r2"]
ms_test_rmse = -cv_results["test_neg_root_mean_squared_error"]
ms_test_r2 = cv_results["test_r2"]

print("We now have 5 values for each metric since we had 5 CV folds:")
print(f"  Train | RMSE | {ms_train_rmse.round(1)} mg/L")
print(f"  Test  | RMSE | {ms_test_rmse.round(1)} mg/L")
print(f"  Train | R^2  | {ms_train_r2.round(3)}")
print(f"  Test  | R^2  | {ms_test_r2.round(3)}")
print()

print("An easier way to interpret these results is to calculate the mean and SD of each metric:")
print(f"  Train | RMSE | {np.mean(ms_train_rmse):6.1f} +/- {np.std(ms_train_rmse):6.1f} mg/L")
print(f"  Test  | RMSE | {np.mean(ms_test_rmse):6.1f} +/- {np.std(ms_test_rmse):6.1f} mg/L")
print(f"  Train | R^2  | {np.mean(ms_train_r2):6.3f} +/- {np.std(ms_train_r2):6.3f}")
print(f"  Test  | R^2  | {np.mean(ms_test_r2):6.3f} +/- {np.std(ms_test_r2):6.3f}")
print()
