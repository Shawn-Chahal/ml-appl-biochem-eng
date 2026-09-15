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

# Let's visualize the data in 'table_wine_analysis.csv

df = pd.read_csv(os.path.join("tables", "table_wine_analysis.csv"))

# Let's start with a basic figure plotting
# 'proline_malic_acid_ratio' vs 'flavanoids':
x_feature = "flavanoids"
y_feature = "proline_malic_acid_ratio"

# ============================================================================
#   Figure 1: Basic Scatter Plot
# ============================================================================
#
#   * Create and customize a scatter plot using matplotlib.
#
#   * Visualize the relationship between the proline_malic_acid_ratio
#     and flavanoids features.
#
# ============================================================================

# Create the blank figure canvas and axes
fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=200, layout="constrained")
# fig: represents the entire figure
# ax: represents an axes (you can think of it as a subplot)
# figsize: represents the width and height of the figure in inches
# dpi: pixels per inch. As a guideline, use:
#      * dpi=200 when prototyping
#      * dpi=400 when publishing online
#      * dpi=600 when publishing in print (e.g., Word, PDF, etc.)
# layout: represents the layout, in this case constrained, which will
#         limit whitespace


# Make a scatter plot with x_feature on the x-axis and y_feature on the y-axis:
ax.scatter(
    df.loc[:, x_feature], df.loc[:, y_feature],  # x and y data
    color="tab:blue",  # The color of the markers
    marker="o",  # The shape of the markers
    s=16,  # The size of the markers
    alpha=0.7,  # The opacity of the markers. Helps see dense regions.
    edgecolors="black",  # The color of the markers' edge
    linewidth=0.5  # The width of the markers' edge
)

# Name the x and y axes according to x_feature and y_feature, respectively.
# You can also adjust the fontsize:
ax.set_xlabel(x_feature, fontsize=10)
ax.set_ylabel(y_feature, fontsize=10)

# You can also adjust the labelsize for the tick marks:
ax.tick_params(axis='both', which='major', labelsize=10)

# Save the figure as a png file to the 'figures' directory:
fig.savefig(os.path.join("figures", "figure_01.png"))

# Close the figure to free up memory:
plt.close(fig)

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

# Get all the unique values in the 'wine' column:
labels = df.loc[:, 'wine'].unique()

# Sort the labels so they appear in order (i.e., wine_0, wine_1, etc.):
labels = np.sort(labels)

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

    # print so you can see what mask looks like:
    print(f"When {label = }, then {mask = }")

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
