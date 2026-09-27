# ============================================================================
#   Normalization
# ============================================================================
#
#   * Normalization scales data while preserving its overall shape.
#
#   * L1 and L2 normalization can transform spectra to a common scale using
#     different mathematical definitions of vector length.
#
#   * Normalization helps compare samples based on spectral patterns rather
#     than signal magnitude, which may vary over time.
#
# ============================================================================

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler


# Define functions
def norm_l1(x):
    # Calculate the sum along columns (i.e., axis=1)
    return np.sum(np.abs(x), axis=1)


def norm_l2(x):
    # Calculate the sum along columns (i.e., axis=1)
    return np.sqrt(np.sum(x ** 2, axis=1))


def generate_figure(spectra_type, tag, plot_legend=True):
    linestyles = ["-", ":", "--", "-."]
    fig_width = 14.33 / 2.54
    fig_height = 10.24 / 2.54
    fontsize_xy_label = 14
    fontsize_ticks = 12
    fontsize_legend = 12
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), dpi=200, layout="constrained")
    for i in range(spectra.shape[0]):
        ax.plot(wavelengths, spectra_type[i, :], label=f"Sample {i}", linestyle=linestyles[i])

    ax.set_xlabel("Wavelength [nm]", fontsize=fontsize_xy_label)
    ax.set_ylabel("Fluorescence [a.u.]", fontsize=fontsize_xy_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    if plot_legend:
        ax.legend(fontsize=fontsize_legend)

    fig.savefig(os.path.join("figures", f"figure_01_{tag}.png"))
    plt.close(fig)


def wavelength_string_to_float(wavelength_string):
    # Create a list of strings. E.g., "410 nm" becomes ["410", "nm"]
    wavelength_string_list = wavelength_string.split(" ")

    # Get the first item in the list. E.g., "410"
    wavelength_string_value = wavelength_string_list[0]

    # Convert the number from a string to a float. E.g., "410" becomes 410.0
    wavelength_float = float(wavelength_string_value)

    return wavelength_float


# Import the dataset
df = pd.read_csv(os.path.join("datasets", "dataset_fluo_spectra.csv"))

# Remove the "nm" from column labels and convert from string to float
wavelengths = [wavelength_string_to_float(col) for col in df.columns]

# Convert the entire DataFrame into a numpy array
spectra = df.to_numpy()

# Perform L1 Normalization
spectra_l1 = normalize(spectra, norm="l1")

# Perform L2 Normalization
spectra_l2 = normalize(spectra, norm="l2")

# Recall the shape of a numpy array...
print(f"{spectra.shape = }")

# Generate figures
generate_figure(spectra, "RawData")
generate_figure(spectra_l1, "L1")
generate_figure(spectra_l2, "L2")

# See impact of Normalization
print("\nCalculating L1 norm...")
print(f"  {norm_l1(spectra)    = }")
print(f"  {norm_l1(spectra_l1) = }")
print(f"  {norm_l1(spectra_l2) = }")

print("\nCalculating L2 norm...")
print(f"  {norm_l2(spectra)    = }")
print(f"  {norm_l2(spectra_l1) = }")
print(f"  {norm_l2(spectra_l2) = }")

# ============================================================================
#   An example of what not to do...
# ============================================================================
#
#   * You can try this code once you've learned about standardization.
#
# ============================================================================

learned_about_standardization = True

if learned_about_standardization:
    scaler = StandardScaler()
    spectra_std = scaler.fit_transform(spectra)
    generate_figure(spectra_std, "Standardized", plot_legend=False)
