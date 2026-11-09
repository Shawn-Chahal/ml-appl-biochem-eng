import gc
import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import root_mean_squared_error

from units.unit_07.src.classes import LSTMNeuralNetwork
from units.unit_07.src.functions import prepare_data, progress_bar

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Define training parameters
n_simulations = 1000
seq_length = 22

# Fraction of simulated forecasts used to calculate prediction intervals
prediction_interval = 0.95
alpha = 1 - prediction_interval

# Create a random number generator
rng = np.random.default_rng(0)

# Import the time series dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series.csv"))

# Import information describing the initial conditions of each batch
df_ic = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series_ic.csv"))

# Extract the unique batch IDs
batch_ids = np.unique(df.loc[:, "Batch ID"])

# Define batch_id_test
batch_id_test = 13

# Select all process variables to include in the NVAR model
features = df.columns[2:]

# Select the variable that will be plotted and evaluated
feature_plot = "Biomass [g/L]"
idx_fp = np.argmax(features == feature_plot)

# Extract units for plotting and reporting
units = feature_plot.split(" [")[-1][:-1]

n_iters = n_simulations
c_iter = 0

# Create a figure containing one subplot for each set of experimental initial conditions.
fig, ax = plt.subplots(1, 1, figsize=((14.4 / 2.54), (10.24 / 2.54)), dpi=300, layout="constrained")

y_min = 0
y_max = np.max(df.loc[:, feature_plot])

# Prepare the dataset to train LSTM
x_data = df.loc[:, features].to_numpy()
group_ids = df.loc[:, "Batch ID"].to_numpy()
x_sequences, y_sequences, g_sequences = prepare_data(x_data, group_ids, seq_length)

mask_test = (g_sequences == batch_id_test)
mask_train = ~mask_test

x_train = x_sequences[mask_train]
y_train = y_sequences[mask_train]
g_train = g_sequences[mask_train]

x_test = x_sequences[mask_test]
y_test = y_sequences[mask_test]
g_test = g_sequences[mask_test]

# Create the model
gc.collect()
torch.cuda.empty_cache()
model = LSTMNeuralNetwork(
    nodes_per_layer=len(features) + 1,
    n_layers=5,
    learning_rate=0.001,
    epochs=3000,
    batch_size=1024,
    weight_decay=0.0001,
    random_state=0,
    verbose=True
)

# Fit the model
model.fit(x_train, y_train, g_train)

# Calculate residual errors on the training data.
# These residuals will later be sampled during forecast simulations to estimate white noise.
y_train_pred, _ = model.predict(x_train)
y_train_res = y_train_pred - y_train

a_min = np.min(y_train)
a_max = np.max(y_train)

# Initial state containing all process variables
idx_ic = (df_ic.loc[:, "Batch ID"] == batch_id_test).idxmax()
x_initial = df_ic.loc[idx_ic, features].to_numpy()
initial_glucose = df_ic.at[idx_ic, "Glucose [g/L]"]
initial_biomass = df_ic.at[idx_ic, "Biomass [g/L]"]

# Extract the data for the test batch
mask = df.loc[:, "Batch ID"] == batch_id_test
t_exp_test = df.loc[mask, "Time [h]"].to_numpy()
x_exp_test = df.loc[mask, features].to_numpy()

# Perform multiple forecast simulations to estimate uncertainty
x_forecasts = []
for _ in range(n_simulations):
    # Initialize the forecast using the known initial conditions.
    x_forecast = [x_initial]
    hidden_state = None
    # Generate forecasts recursively.
    while len(x_forecast) < t_exp_test.shape[0]:

        # Sample residual errors from the training data.
        # This assumes the residuals behave like white noise.
        white_noise = []
        for j in range(y_train_res.shape[2]):
            white_noise.append(rng.choice(y_train_res[:, :, j].flatten()))
        white_noise = np.array([white_noise])

        # Predict the next process state baseline
        x_next, hidden_state = model.predict(np.expand_dims(x_forecast[-1], axis=0), hidden_state)

        # Don't let the model predict values outside the range it has been trained on.
        x_next = np.clip(x_next, a_min=a_min, a_max=a_max)

        # Add white noise
        x_next = x_next + white_noise

        # Prevent negative concentration predictions after white noise is added
        x_next = np.clip(x_next, a_min=0, a_max=None)

        x_forecast.append(x_next[0])

    x_forecast = np.array(x_forecast)

    x_forecasts.append(x_forecast)

    c_iter += 1
    progress_bar(c_iter, n_iters)

x_forecasts = np.array(x_forecasts)

# Calculate the median forecast and prediction intervals at each time point.
x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
x_forecast_pi_higher = np.quantile(x_forecasts, q=1 - alpha / 2, axis=0)

# Calculate RMSE for the selected plotting variable
rmse_forecast = root_mean_squared_error(x_exp_test[:, idx_fp], x_forecast_median[:, idx_fp])

message = f"{n_simulations} simulations\n"
message += f"RMSE: {rmse_forecast:.2f} {units}\n"
message += f"Initial:\n"
message += f"* GLC: {initial_glucose:3.0f} g/L\n"
message += f"* BIO: {initial_biomass:3.1f} g/L\n"

# Plot the test values, forecast median, and prediction interval
ax.scatter(t_exp_test, x_exp_test[:, idx_fp], label="Test", s=25, color="tab:orange", edgecolors="black", zorder=3)
ax.plot(t_exp_test, x_forecast_median[:, idx_fp], label=f"LSTM", color="black", zorder=4)
ax.fill_between(t_exp_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_higher[:, idx_fp],
                label=f"{prediction_interval:.0%} PI", color="tab:blue", edgecolors="none", alpha=0.3, zorder=2)

# Display forecast error and initial conditions
ax.text(x=0.02, y=0.97, s=message, va='top', ha='left', transform=ax.transAxes, fontname="monospace", fontsize=12)

# Track the maximum value needed for consistent subplot limits
y_max = max(y_max, np.max(x_test[:, idx_fp]), np.max(x_forecast_pi_higher[:, idx_fp]))

# Apply consistent axis limits and formatting to all subplots
y_range = y_max - y_min
y_buffer = 0.05 * y_range

ax.set_ylim(bottom=y_min - y_buffer, top=y_max + y_buffer)
ax.set_xlabel("Time [h]", fontsize=12)
ax.set_ylabel(feature_plot, fontsize=12)
ax.legend(loc="upper right", fontsize=12)
ax.xaxis.set_major_locator(MultipleLocator(4))

# Save the figure
fig.savefig(os.path.join("figures", f"figure_01_LSTM.png"))
plt.close(fig)
