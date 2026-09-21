# ============================================================================
#   Pandas DataFrames
# ============================================================================
#
#   * Pandas DataFrames provide a convenient way to store, manipulate, and
#     analyze tabular datasets in Python.
#
#   * Rows typically represent samples, while columns represent features,
#     labels, or process variables.
#
#   * DataFrames are commonly used to import datasets, create new features,
#     filter data, and export results for further analysis.
#
# ============================================================================

import os

import numpy as np
import pandas as pd

LINE_BREAK = "\n" + 500 * "=" + "\n"
# The pandas library can be used to import data, modify it, and export it.

# These settings allow you to see the full dataframe in these examples.
# If you aren't printing your dataframes then these lines won't do anything.
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(LINE_BREAK)

print("You can make your filepath compatible across operating systems by using os.path.join():")
path_import = os.path.join("datasets", "dataset_wine.csv")
print(f"  * Dataset path: {path_import}")
print(LINE_BREAK)

print("You can import your data from a csv file into a pandas DataFrame:\n")
df = pd.read_csv(path_import)
print(df)
print(LINE_BREAK)

print("You can access an element in the DataFrame using the at[] property:")
sample_id = 4
feature_id = "alcohol"
print(f"  E.g., Sample {sample_id} contains {df.at[sample_id, feature_id]} {feature_id}.")
print(LINE_BREAK)

print("You can access a column in the DataFrame using the loc[] property:\n")
print(df.loc[:, feature_id])  # Access all (i.e., :) the rows for the given feature_id (i.e., alcohol)
print(LINE_BREAK)

print("You can access a row in the DataFrame using the loc[] property:\n")
print(df.loc[sample_id, :])  # For a given sample_id (i.e.,4), access all the features
print(LINE_BREAK)

print("You can also convert DataFrames and their slices into numpy arrays:\n")
columns_numpy = df.columns[1:]  # Skips the 1st column (i.e., 'wine') since it is text
print(df.loc[:, columns_numpy].to_numpy())
print(LINE_BREAK)

print("You can use masks to select parts of a DataFrame:\n")
alcohol_threshold = 14.0
wine_target = "wine_0"
mask_alcohol_high = df.loc[:, "alcohol"] > alcohol_threshold
mask_wine_target = df.loc[:, "wine"] == wine_target
mask = mask_alcohol_high * mask_wine_target
print(f"  {mask.to_numpy() = }")
print()

print(f"This DataFrame shows wines belonging to {wine_target} with alcohol > {alcohol_threshold}:\n")
print(df.loc[mask, :])
print(LINE_BREAK)

print("You can make a DataFrame from within your code as well:\n")
n_samples = 101

# Monod equation
mu_max = 0.5  # h^-1
k_s = 100  # mg/L
s = np.linspace(0, 1000, n_samples)  # mg/L
mu = mu_max * (s / (k_s + s))  # h^-1

records = []
for i in range(n_samples):
    # Create a record for each row in your DataFrame
    record = (s[i], k_s, mu[i], mu_max)
    records.append(record)

# Create the column names. Make sure they are in the same order as the record values.
column_names = ["S [mg/L]", "K_S [mg/L]", "mu [h^-1]", "mu_max [h^-1]"]
df_export = pd.DataFrame(records, columns=column_names)

# We can export the DataFrame to a CSV file:
filepath_export = os.path.join("tables", "table_monod_kinetics.csv")

# Assign index=False to prevent the index column from being copied to the csv file.
df_export.to_csv(filepath_export, index=False)

print(df_export)
print(LINE_BREAK)
