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

import os

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print("============\n  SOLUTION  \n============\n")

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
df_export.to_csv(path_export, index=False)

print(df_export)
