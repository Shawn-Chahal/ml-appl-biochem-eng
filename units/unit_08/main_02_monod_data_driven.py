# ============================================================================
#   Data-Driven Models
# ============================================================================
#
#   * This example trains a fully connected neural network to predict
#     cell growth rates directly from substrate concentration,
#     pH, and temperature measurements.
#
#   * Unlike the mechanistic Monod model, no biological equations or
#     process assumptions are imposed on the model structure.
#
# ============================================================================

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from units.unit_08.src.classes import MonodDataDrivenModel
from units.unit_08.src.functions import plot_monod

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Load the dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_hybrid_monod.csv"))

# Extract individual variables from the dataset.
ph = df.loc[:, "pH"].to_numpy()
temp = df.loc[:, "Temperature [C]"].to_numpy()
s = df.loc[:, "Glucose [mM]"].to_numpy()
mu = df.loc[:, "Specific Growth Rate [d^-1]"].to_numpy()

# Construct the feature matrix:
#   Column 0 -> substrate concentration
#   Column 1 -> pH
#   Column 2 -> temperature
x = np.stack((s, ph, temp), axis=1)

# Convert the target variable into a column vector.
y = np.expand_dims(mu, axis=1)

# Split data into training and testing subsets.
# Additional arrays are split simultaneously so that substrate, pH, and temperature values
# remain aligned with the corresponding feature and target samples.
tts_output = train_test_split(x, y, s, ph, temp, test_size=0.2, random_state=0)
x_train, x_test, y_train, y_test, s_train, s_test, ph_train, ph_test, temp_train, temp_test = tts_output

# Train the model.
model = MonodDataDrivenModel(random_state=0, verbose=True)
model.fit(x_train, y_train)

# Generate predictions for the test dataset.
y_pred = model.predict(x_test)

# Calculate overall test set prediction error.
rmse_test_overall = root_mean_squared_error(y_test, y_pred)

print(f"Data-Driven RMSE: {rmse_test_overall:.3f}")

# Determine all pH and temperature values represented in the test set.
u_ph_test = np.unique(ph_test)
u_temp_test = np.unique(temp_test)

# Generate a separate Monod plot for every observed pH-temperature combination.
for ph_val in u_ph_test:
    for temp_val in u_temp_test:
        # Create a dense substrate grid for visualizing the fitted Monod growth curve.
        s_array = np.linspace(0, np.max(s), 1000)

        # Hold pH and temperature constant while varying substrate concentration.
        ph_array = ph_val * np.ones_like(s_array)
        temp_array = temp_val * np.ones_like(s_array)

        # Construct model inputs for model visualization.
        x_input = np.stack((s_array, ph_array, temp_array), axis=1)

        # Evaluate the fitted Monod equation.
        x_plot = s_array
        y_plot = model.predict(x_input)[:, 0]

        # Select test samples corresponding to the current pH and temperature combination.
        mask_test_iter = (ph_test == ph_val) * (temp_test == temp_val)

        # Some pH-temperature combinations may not contain any samples after the train-test split.
        # In which case, skip this iteration of the for loop.
        if np.sum(mask_test_iter) == 0:
            print(f"Skipping pH: {ph_val}, T: {temp_val} C")
            continue

        # Extract the matching experimental observations.
        s_test_iter = s_test[mask_test_iter]
        x_test_iter = x_test[mask_test_iter]
        y_test_iter = y_test[mask_test_iter]

        # Compute predictions and RMSE for the current operating condition.
        y_pred_iter = model.predict(x_test_iter)
        rmse_test_iter = root_mean_squared_error(y_test_iter, y_pred_iter)

        # Generate information to be passed to the plotting function.
        ph_string = f"pH_{str(ph_val).replace('.', '_')}"
        temp_string = f"T_{str(temp_val).replace('.', '_')}"

        message = f"RMSE (overall):   {rmse_test_overall:.3f} " + r"$d^{-1}$" + "\n"
        message += f"RMSE (iteration): {rmse_test_iter:.3f} " + r"$d^{-1}$" + "\n"
        message += f"pH: {ph_val} | T: {temp_val} C"

        # Plot experimental data together with the fitted Monod curve.
        plot_monod(
            filepath=os.path.join("figures", f"figure_02_MonodDataDriven_{ph_string}_{temp_string}.png"),
            x_scatter=s_test_iter,
            y_scatter=y_test_iter,
            x_plot=x_plot,
            y_plot=y_plot,
            x_max=np.max(s),
            y_max=1.01 * np.max(y),
            model_name="Data-Driven",
            message=message,
        )
