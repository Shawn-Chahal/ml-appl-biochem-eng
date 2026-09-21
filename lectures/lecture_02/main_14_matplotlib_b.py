import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Previously on matplotlib...

df = pd.read_csv(os.path.join("tables", "table_wine_analysis.csv"))
x_feature = "flavanoids"
y_feature = "proline_malic_acid_ratio"

# ============================================================================
#   Figure 2: Grouped Scatter Plot
# ============================================================================
#
#   * The dataset we are using consists of three types of wines.
#
#   * Color and marker styles can be used to distinguish different
#     classes within a dataset.
#
#   * Plot each wine type using a unique color and marker style
#     and include a legend to identify the groups.
#
# ============================================================================

# Define the colors and markers you want to use for each wine.
# For example, if there are only 3 groups, then the first 3
# colors and markers will be used.
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red",
          "tab:purple", "tab:brown", "tab:gray", "tab:olive"]

MARKERS = ["o", "D", "s", "^", "X", "P", "*", "v"]

# More colors at: https://matplotlib.org/stable/gallery/color/named_colors.html
# More markers at: https://matplotlib.org/stable/api/markers_api.html


# Get all the unique values in the 'wine' column in order (i.e., wine_0, wine_1, etc.):
labels = np.unique(df.loc[:, 'wine'])

# print so you can see what labels looks like:
print(f"{labels = }")

# Create the blank figure and axes:
fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=200, layout="constrained")

# We will iterate over each label and plot one group of wine at a time:
for idx_style, label in enumerate(labels):
    # Create a mask to identify:
    #    * which rows belong to the current label (True)
    #    * which rows do not belong to the current label (False)
    mask = (df.loc[:, "wine"] == label)

    ax.scatter(
        # The mask will make sure that only the rows belonging to the
        # current label will be plotted:
        df.loc[mask, x_feature], df.loc[mask, y_feature],

        # We can include the label here so it appears in the legend:
        label=label,

        # Use the color and marker that was defined earlier:
        color=COLORS[idx_style], marker=MARKERS[idx_style],

        # Modify settings until it looks nice:
        s=32, alpha=0.7, edgecolors="black", linewidth=1.0
    )

ax.set_xlabel(x_feature, fontsize=10)
ax.set_ylabel(y_feature, fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)

# Create a legend
ax.legend(fontsize=10)

fig.savefig(os.path.join("figures", "figure_02.png"))
plt.close(fig)
