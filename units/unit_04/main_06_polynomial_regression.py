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
degree = 2
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=degree, include_bias=False),
    LinearRegression()
)

cv_results = cross_validate(model, x, y, cv=5, scoring=("neg_root_mean_squared_error", "r2"), return_train_score=True)

ms_train_rmse = -cv_results["train_neg_root_mean_squared_error"]
ms_train_r2 = cv_results["train_r2"]
ms_test_rmse = -cv_results["test_neg_root_mean_squared_error"]
ms_test_r2 = cv_results["test_r2"]

print()
print(f"Polynomial model with {degree = }:")
print(f"  Train | RMSE | {np.mean(ms_train_rmse):6.1f} +/- {np.std(ms_train_rmse):6.1f} mg/L")
print(f"  Test  | RMSE | {np.mean(ms_test_rmse):6.1f} +/- {np.std(ms_test_rmse):6.1f} mg/L")
print(f"  Train | R^2  | {np.mean(ms_train_r2):6.3f} +/- {np.std(ms_train_r2):6.3f}")
print(f"  Test  | R^2  | {np.mean(ms_test_r2):6.3f} +/- {np.std(ms_test_r2):6.3f}")
print()

print("Let's examine the weights of our polynomial model.")
print()
# We need to fit the model since it was previously fit in the cross_validate() function
model.fit(x, y)

# We can get the power of each original feature used to create the new polynomial features.
powers = model['polynomialfeatures'].powers_

# We can get the coefficients of our model.
coefs = model['linearregression'].coef_

# We can get the intercept of our model.
intercept = model['linearregression'].intercept_

print("Let's look at the shape of the powers and coefs array:")
print(f"  {powers.shape = }")
print(f"  {coefs.shape = }")
print()

print("We can see from the shape that:")
print("  * The number of rows represents the number of new polynomial features")
print("    and is equal to the number of coefficients.")
print("  * The number of columns represents the number of original features.")
print()

print("Let's get a quick look at what our polynomial model looks like:")
print(list(features))
# When multiple arrays have the same number of rows,
# you can iterate over them simultaneously using the zip() function.
for c, p in zip(coefs, powers):
    print(f"{c:+8.3f} {p}")
print(f"{intercept:+8.3f}")
print()

print("The polynomial model can be defined more clearly as:\n")
print(f"{targets} = ")
print(f"{intercept:8.3f}")
for i in range(powers.shape[0]):
    partial_terms = []
    for j in range(powers.shape[1]):

        # Since we are using standardized features, we should remove the units.
        feature_unit_splitter = " ["
        if feature_unit_splitter in features[j]:
            feature_name = features[j].split(feature_unit_splitter)[0]
        else:
            feature_name = features[j]

        if powers[i, j] == 0:
            pass  # Do nothing because the j-th original feature is not used in the i-th new polynomial feature
        elif powers[i, j] == 1:
            partial_terms.append(f"[{feature_name}]")  # We don't need to say "[feature_name] to the power of 1"
        else:
            partial_terms.append(f"[{feature_name}]^{powers[i, j]}")

    coef = coefs[i]
    term = " * ".join(partial_terms)

    print(f"{coef:+8.3f} * {term}")

print("""
A few notes to to keep in mind:
  * It is important to remember that these coefficients are 
    based on the standardized features.
  * The use of standardize features is what allows us to 
    compare these coefficients on the same scale.
  * The intercept can be interpreted as the baseline and 
    each coefficient and term is a modifier to that baseline.
""")
