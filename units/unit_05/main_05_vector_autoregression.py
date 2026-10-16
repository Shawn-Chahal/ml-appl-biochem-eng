# ============================================================================
#   Vector Autoregression
# ============================================================================
#
#   * Vector autoregression (VAR) extends autoregressive modelling
#     to multiple time series variables simultaneously.
#
#   * In a VAR(p) model, the current value of each variable is predicted
#     using p lagged values from all variables included in the model.
#
#   * By modelling all variables simultaneously, a VAR model can learn
#     interactions between process variables and use those relationships
#     to improve forecasts.
#
# ============================================================================

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

from units.unit_05.src.classes import StandardScalerVAR
from units.unit_05.src.functions import generate_x_lag_cur_var

# Number of forecast simulations
n_simulations = 1000

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

# Select all process variables to include in the VAR model
features = df.columns[2:]

# Select the variable that will be plotted and evaluated
feature_plot = "Biomass [g/L]"
idx_fp = np.argmax(features == feature_plot)

# Extract units for plotting and reporting
units = feature_plot.split(" [")[-1][:-1]

# These will be used to organize the subplots according to the experimental initial conditions
ic_glucose = np.unique(df_ic.loc[:, "Glucose [g/L]"])
ic_biomass = np.unique(df_ic.loc[:, "Biomass [g/L]"])

nrows = ic_glucose.shape[0]
ncols = ic_biomass.shape[0]

# You can run this segment of code if you want to inspect how the lagged value arrays used
# by the VAR model are constructed.
print_var_arrays = False
if print_var_arrays:
    x_lag_print, x_cur_print, _ = generate_x_lag_cur_var(p=3, batch_ids=batch_ids, df=df, features=features)
    for i in range(x_cur_print.shape[0]):
        print(x_cur_print[i], x_lag_print[i])

# Evaluate VAR models with different numbers of lagged values
for p in range(1, 4):
    # Create a figure containing one subplot for each set of experimental initial conditions.
    fig, axes = plt.subplots(nrows, ncols, figsize=((29.21 / 2.54), (12.09 / 2.54)), dpi=200, layout="constrained")

    y_min = 0
    y_max = np.max(df.loc[:, feature_plot])
    rmse_forecasts = []

    # Leave-one-batch-out cross-validation
    for batch_id_test in batch_ids:

        # Training batches contain all batches except the test batch
        batch_ids_train = np.array([i for i in batch_ids if i != batch_id_test])

        # Generate the lagged value dataset used to train VAR(p)
        x_lag_train, x_cur_train, _ = generate_x_lag_cur_var(p=p, batch_ids=batch_ids_train, df=df, features=features)

        # Create and train the model
        model = make_pipeline(StandardScalerVAR(p=p), LinearRegression())
        model.fit(x_lag_train, x_cur_train)

        # Calculate residual errors on the training data.
        # These residuals will later be sampled during forecast simulations to estimate white noise.
        x_cur_train_pred = model.predict(x_lag_train)
        x_cur_train_res = x_cur_train_pred - x_cur_train

        # Initial state containing all process variables
        idx_ic = (df_ic.loc[:, "Batch ID"] == batch_id_test).idxmax()
        x_initial = df_ic.loc[idx_ic, features].to_numpy()

        # Determine which subplot corresponds to the test batch initial conditions
        initial_glucose = df_ic.at[idx_ic, "Glucose [g/L]"]
        initial_biomass = df_ic.at[idx_ic, "Biomass [g/L]"]
        i_ax = np.argmax(ic_glucose == initial_glucose)
        j_ax = np.argmax(ic_biomass == initial_biomass)
        ax = axes[i_ax, j_ax]

        # Extract the data for the test batch
        mask = df.loc[:, "Batch ID"] == batch_id_test
        t_test = df.loc[mask, "Time [h]"].to_numpy()
        x_test = df.loc[mask, features].to_numpy()

        # Perform multiple forecast simulations to estimate uncertainty
        x_forecasts = []
        for simulation_id in range(n_simulations):
            # Initialize the forecast using the known initial conditions.
            # Unlike the AR model, each state contains all process variables.
            x_forecast = [x_initial]

            # Generate forecasts recursively.
            # Each newly predicted state becomes available for future predictions.
            while len(x_forecast) < t_test.shape[0]:

                x_lag = x_forecast[::-1][:p].copy()

                # At the beginning of the simulation there may be fewer than p available lagged vectors.
                # If necessary, repeat the most recent state until p lagged vectors are available.
                while len(x_lag) < p:
                    x_lag.append(x_lag[-1])

                # Concatenate all lagged vectors into a single feature vector expected by the VAR model.
                x_lag = np.concatenate(x_lag, axis=0)
                x_lag = np.expand_dims(x_lag, axis=0)

                # Sample residual errors from the training data.
                # This assumes the residuals behave like white noise.
                white_noise = rng.choice(x_cur_train_res, size=x_lag.shape[0])

                # Predict the next process state and add white noise
                x_cur = model.predict(x_lag) + white_noise

                # Prevent negative concentration predictions
                x_cur = np.clip(x_cur[0], a_min=0, a_max=None)

                x_forecast.append(x_cur)

            x_forecast = np.array(x_forecast)

            x_forecasts.append(x_forecast)

        x_forecasts = np.array(x_forecasts)

        # Calculate the median forecast and prediction intervals at each time point.
        x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
        x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
        x_forecast_pi_higher = np.quantile(x_forecasts, q=1 - alpha / 2, axis=0)

        # Calculate RMSE for the selected plotting variable
        rmse_forecast = root_mean_squared_error(x_test[:, idx_fp], x_forecast_median[:, idx_fp])
        rmse_forecasts.append(rmse_forecast)

        message = f"RMSE: {rmse_forecast:.2f} {units}\n"
        message += f"Initial:\n"
        message += f"* GLC: {initial_glucose:3.0f} g/L\n"
        message += f"* BIO: {initial_biomass:3.1f} g/L\n"

        # Plot the test values, forecast median, and prediction interval
        ax.scatter(t_test, x_test[:, idx_fp], label="Test", s=10, color="tab:orange", edgecolors="black", zorder=3)
        ax.plot(t_test, x_forecast_median[:, idx_fp], label=f"VAR({p})", color="black", zorder=4)
        ax.fill_between(t_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_higher[:, idx_fp],
                        label=f"{prediction_interval:.0%} PI", color="tab:blue", edgecolors="none", alpha=0.3, zorder=2)

        # Display forecast error and initial conditions
        ax.text(x=0.02, y=0.96, s=message, va='top', ha='left', transform=ax.transAxes, fontname="monospace",
                fontsize=7)

        # Track the maximum value needed for consistent subplot limits
        y_max = max(y_max, np.max(x_test[:, idx_fp]), np.max(x_forecast_pi_higher[:, idx_fp]))

    # Apply consistent axis limits and formatting to all subplots
    y_range = y_max - y_min
    y_buffer = 0.05 * y_range

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_ylim(bottom=y_min - y_buffer, top=y_max + y_buffer)

            if i == axes.shape[0] - 1:
                axes[i, j].set_xlabel("Time [h]")
                axes[i, j].xaxis.set_major_locator(MultipleLocator(4))

            else:
                axes[i, j].set_xticks([])

            if j == 0:
                axes[i, j].set_ylabel(feature_plot)
            else:
                axes[i, j].set_yticks([])

            # Add the legend to only one subplot to avoid repetition
            if i == 0 and j == 0:
                axes[i, j].legend(loc="upper right", fontsize=7)

    # Save the figure
    fig.savefig(os.path.join("figures", f"figure_05_p{p:02d}_N{n_simulations:06d}.png"))
    plt.close(fig)

    # Calculate and report the forecast error
    rmse_mean = np.mean(rmse_forecasts)
    rmse_sd = np.std(rmse_forecasts)
    print(f"{p = :2d} | RMSE (Forecast): {rmse_mean:6.2f} +/- {rmse_sd:6.2f} {units}")
