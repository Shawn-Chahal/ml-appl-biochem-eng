# ============================================================================
#   Data Visualization with Matplotlib
# ============================================================================
#
#   * Data visualization is an important step in exploratory data analysis
#     and machine learning workflows.
#
#   * Matplotlib can be used to create figures that reveal patterns,
#     relationships, and trends within datasets.
#
#   * In this example, we will create scatter plots and multi-panel figures
#     to visualize relationships between wine features.
#
# ============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Previously on matplotlib...

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red",
          "tab:purple", "tab:brown", "tab:gray", "tab:olive"]

MARKERS = ["o", "D", "s", "^", "X", "P", "*", "v"]

df = pd.read_csv(os.path.join("tables", "table_wine_analysis.csv"))

x_feature = "flavanoids"
y_feature = "proline_malic_acid_ratio"
labels = np.unique(df.loc[:, 'wine'])

# ============================================================================
#   Figure 3: Pairwise Feature Visualization
# ============================================================================
#
#   * The dataset we are using consists of 5 features.
#
#   * Pairwise visualizations can help identify trends, clusters, and
#     potential feature relationships.
#
#   * Create a grid of subplots showing every possible pairing of features.
#
# ============================================================================

# Define the features you will plot:
features = df.columns[1:]  # Column 1 to end (Remember we start at 0)

# print features so you can confirm you got the columns you wanted:
print(f"{features = }")

# The number of rows and columns will each be equal to the number of features:
nrows = len(features)
ncols = len(features)

# Create the blank figure and axes. We are now calling it axes since
# there are multiple subplots. We also need to make the figsize larger
# to accomodate all the subplots
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(6.5 * ncols, 4.0 * nrows),
    dpi=200, layout="constrained"
)

# Iterate over the features so that each feature has a chance to be
# plotted on the x-axis and y-axis with every other feature:
for x_idx, x_feature in enumerate(features):
    for y_idx, y_feature in enumerate(features):

        for idx_style, label in enumerate(labels):
            mask = (df.loc[:, "wine"] == label)

            # We now have to indicate which axes we want to draw the scatterplot
            # on. y represents the rows and x represents the columns, therefore
            # we want to plot on axes[y_idx, x_idx]. This will become more clear
            # once you see the resulting figure
            axes[y_idx, x_idx].scatter(
                df.loc[mask, x_feature], df.loc[mask, y_feature], label=label,
                color=COLORS[idx_style], marker=MARKERS[idx_style], s=32,
                alpha=0.7, edgecolors="black", linewidth=1.0
            )

        axes[y_idx, x_idx].set_xlabel(x_feature, fontsize=10)
        axes[y_idx, x_idx].set_ylabel(y_feature, fontsize=10)
        axes[y_idx, x_idx].tick_params(axis='both', which='major', labelsize=10)
        axes[y_idx, x_idx].legend(fontsize=10)

fig.savefig(os.path.join("figures", "figure_03.png"))
plt.close(fig)
