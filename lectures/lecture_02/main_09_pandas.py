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

# ============================================================================
#   PROBLEM
# ============================================================================
#
#   Consider the wine dataset contained in dataset_wine.csv
#    * 178 samples
#    * 13 features (+ 1 label)
#
#   You are interested in performing data analysis on the
#   proline/malic_acid ratio in wines, alongside a few other
#   features.
#
#   Export a DataFrame that contains columns for:
#    * wine
#    * alcohol
#    * color_intensity
#    * flavanoids
#    * od280_od315_ratio
#    * proline_malic_acid_ratio
#
#   Note: We are just exporting the DataFrame here.
#   We will perform the analysis in the next module.
#
# ============================================================================

# ============================================================================
#   SOLUTION
# ============================================================================
#
#   1) Import the dataset into a pandas DataFrame.
#   2) Create the proline_malic_acid_ratio feature.
#   3) Select the columns required for analysis.
#   4) Export the resulting DataFrame to a csv file.
#
# ============================================================================

print("============\n  SOLUTION  \n============\n")

# Note: We could technically use the same 'df' from above, but for
# completeness we will show the full solution here.


# Import the dataset into a pandas DataFrame:
path_import = os.path.join("datasets", "dataset_wine.csv")
df = pd.read_csv(path_import)

# Create a new column for proline_malic_acid_ratio:
df.loc[:, "proline_malic_acid_ratio"] = df.loc[:, "proline"] / df.loc[:, "malic_acid"]

# Create a list of the columns to be exported:
columns_export = [
    "wine",
    "alcohol",
    "color_intensity",
    "flavanoids",
    "od280_od315_ratio",
    "proline_malic_acid_ratio"
]

# Create a DataFrame that contains just the columns we want to export:
df_export = df.loc[:, columns_export]

# Export the DataFrame to a csv file:
path_export = os.path.join("tables", "table_wine_analysis.csv")
df_export.to_csv(path_export, index=False)  # index=False, prevents the index column from being copied to the csv file.

print(df_export)
