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
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


# Define functions
def generate_figure(x_data, tag, standardized_data=False):
    markers = ["o", "D", "s", "^", "X", "P", "*", "v"]
    fig_width = 14.33 / 2.54
    fig_height = 10.24 / 2.54
    fontsize_xy_label = 14
    fontsize_ticks = 12
    fontsize_legend = 12
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), dpi=200, layout="constrained")
    for i, feature in enumerate(features):
        x_plot = x_data[:, i]
        y_plot = i * np.ones_like(x_plot)
        label = feature

        if standardized_data:
            idx_end = label.find('[') - 1
            if idx_end > 0:
                label = label[:label.find('[')]

        ax.scatter(x_plot, y_plot, label=label, marker=markers[i], edgecolors="black", alpha=0.1)

    ax.set_xlabel("Feature value", fontsize=fontsize_xy_label)
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.set_ylim(bottom=-0.5, top=len(features) + 0.5)
    ax.legend(ncols=2, loc="upper center", fontsize=fontsize_legend)

    fig.savefig(os.path.join("figures", f"figure_02_{tag}.png"))
    plt.close(fig)


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

# Create an instance of StandardScaler
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

# ========================
#   PLOT x_1 and x_1_std
# ========================

generate_figure(x_1, "RawData")
generate_figure(x_1_std, "Standardized", standardized_data=True)
