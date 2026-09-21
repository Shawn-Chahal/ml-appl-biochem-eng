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
