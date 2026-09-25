# ============================================================================
#   Standardization
# ============================================================================
#
#   * Standardization transforms features so they have a mean of 0 and a
#     standard deviation of 1.
#
#   * The scaling parameters are learned from a training dataset and then
#     applied to new data using the same transformation.
#
#   * Standardization helps ensure that features with different scales
#     contribute more equally to machine learning models.
#
# ============================================================================

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import dataset
df = pd.read_csv(os.path.join("datasets", "dataset_protein_yield.csv"))

# Extract feature columns (i.e., all but last column)
features = df.columns[:-1]

# Convert to numpy
x = df.loc[:, features].to_numpy()

# For demonstration purposes we will split the dataset
n_split = 400  # samples
x_1 = x[:n_split, :]  # Contains the first 400 samples
x_2 = x[n_split:, :]  # Contains the remaining 100 samples

# Create a standard scaler instance
scaler = StandardScaler()

# Learn the mean and SD of each feature using x_1
scaler.fit(x_1)

# Standardize x_1
x_1_std = scaler.transform(x_1)

# Standardize x_2 using the mean and SD learned from x_1
x_2_std = scaler.transform(x_2)

# Print the first row of the arrays to see how they have changed
print("You can check the first row of each array to see how they changed:")
print(f"  {x_1[0, :]     = }")
print(f"  {x_1_std[0, :] = }")
print(f"  {x_2[0, :]     = }")
print(f"  {x_2_std[0, :] = }")
print()

print("You can calculate the mean and SD of each feature for each array:")
print(f"  {np.mean(x_1, axis=0).round(2)     = }")
print(f"  {np.mean(x_1_std, axis=0).round(2) = }")
print(f"  {np.mean(x_2, axis=0).round(2)     = }")
print(f"  {np.mean(x_2_std, axis=0).round(2) = }")
print()

print(f"  {np.std(x_1, axis=0).round(2)     = }")
print(f"  {np.std(x_1_std, axis=0).round(2) = }")
print(f"  {np.std(x_2, axis=0).round(2)     = }")
print(f"  {np.std(x_2_std, axis=0).round(2) = }")
print()

print("Note how the mean and SD are (0, 1) for x_1, but not for x_2.")
