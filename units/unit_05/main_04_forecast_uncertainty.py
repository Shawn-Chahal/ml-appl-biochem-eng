# ============================================================================
#   Forecast Uncertainty
# ============================================================================
#
#   * Future values cannot be predicted with complete certainty.
#
#   * In autoregressive models, forecast uncertainty generally increases
#     as predictions extend further into the future because prediction
#     errors can accumulate.
#
#   * Repeated simulations can be used to estimate prediction intervals,
#     which provide a likely range for future values.
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

from units.unit_05.src.classes import StandardScalerAR
from units.unit_05.src.functions import generate_x_lag_cur_ar

# Create a random number generator
rng = np.random.default_rng(0)

# Number of simulations per forecast
n_simulations = 1000

# Import dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series.csv"))

# Extract the unique batch IDs
batch_ids = np.unique(df.loc[:, "Batch ID"])

# Select the time series feature of interest
feature = "Biomass [g/L]"
units = feature.split(" [")[-1][:-1]

# Fraction of predictions to visualize
prediction_interval = 0.95
alpha = 1 - prediction_interval

# Import information describing the initial conditions of each batch
df_ic = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series_ic.csv"))

# These will be used to organize the subplots
ic_glucose = np.unique(df_ic.loc[:, "Glucose [g/L]"])
ic_biomass = np.unique(df_ic.loc[:, "Biomass [g/L]"])

nrows = ic_glucose.shape[0]
ncols = ic_biomass.shape[0]

# Evaluate autoregressive models with different numbers of lagged values.
for p in range(1, 4):
    fig, axes = plt.subplots(nrows, ncols, figsize=((29.21 / 2.54), (12.09 / 2.54)), dpi=200, layout="constrained")
    y_min = 0
    y_max = np.max(df.loc[:, feature])
    rmse_forecasts = []

    # Leave-one-batch-out cross-validation
    # One batch is excluded from training and used as the test dataset
    for batch_id_test in batch_ids:
        # Training batches contain all batches except the test batch
        batch_ids_train = np.array([i for i in batch_ids if i != batch_id_test])

        # Generate the lagged-value dataset used to train AR(p)
        x_lag_train, x_cur_train, _ = generate_x_lag_cur_ar(p=p, batch_ids=batch_ids_train, df=df, feature=feature)

        # Create and train the model
        model = make_pipeline(StandardScalerAR(), LinearRegression())
        model.fit(x_lag_train, x_cur_train)

        # Calculate the residual errors (noise)
        x_cur_train_predict = model.predict(x_lag_train)
        x_cur_train_noise = x_cur_train_predict - x_cur_train

        # Identify the initial condition corresponding to the test batch
        idx_ic = df_ic.index[np.argmax(df_ic.loc[:, "Batch ID"] == batch_id_test)]
        x_initial = df_ic.at[idx_ic, feature]
        initial_glucose = df_ic.at[idx_ic, "Glucose [g/L]"]
        initial_biomass = df_ic.at[idx_ic, "Biomass [g/L]"]

        # Determine which subplot corresponds to the test batch initial conditions
        i_ax = np.argmax(ic_glucose == initial_glucose)
        j_ax = np.argmax(ic_biomass == initial_biomass)
        ax = axes[i_ax, j_ax]

        # Extract the data for the test batch
        mask = df.loc[:, "Batch ID"] == batch_id_test
        t_test = df.loc[mask, "Time [h]"].to_numpy()
        x_test = df.loc[mask, feature].to_numpy()

        # This time we will perform multiple simulations
        x_forecasts = []
        for simulation_id in range(n_simulations):
            # Initialize the forecast using the known initial value
            x_forecast = [x_initial]

            # Generate the forecast recursively
            # Each newly predicted value becomes available for future predictions
            while len(x_forecast) < t_test.shape[0]:
                # We need to reverse the list and then take the first p values
                # This makes: x_lag[0] --> x_(t-1); x_lag[1] --> x_(t-2); etc.
                x_lag = x_forecast[::-1][:p].copy()

                # At the start of the simulation there may be fewer than p available lagged values.
                # If needed, repeat the most recent value until p lagged values are available.
                while len(x_lag) < p:
                    x_lag.append(x_lag[-1])

                x_lag = np.array([x_lag])

                # Get white noise from the residuals fo the training data
                white_noise = rng.choice(x_cur_train_noise, size=x_lag.shape[0])

                # Predict the next value with white noise
                x_cur = model.predict(x_lag) + white_noise

                # Prevent negative concentration predictions
                x_cur = max(0, x_cur[0])

                x_forecast.append(x_cur)

            x_forecast = np.array(x_forecast)

            # append this forecast to the list of simulated forecasts
            x_forecasts.append(x_forecast)

        x_forecasts = np.array(x_forecasts)

        # Calculate the median and lower and upper prediction intervals at each time point
        x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
        x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
        x_forecast_pi_higher = np.quantile(x_forecasts, q=1 - alpha / 2, axis=0)

        # Calculate RMSE
        rmse_forecast = root_mean_squared_error(x_test, x_forecast_median)
        rmse_forecasts.append(rmse_forecast)

        message = f"RMSE: {rmse_forecast:.2f} {units}\n"
        message += f"Initial:\n"
        message += f"* GLC: {initial_glucose:3.0f} g/L\n"
        message += f"* BIO: {initial_biomass:3.1f} g/L\n"

        # Plot test values, forecast, and prediction intervals
        ax.scatter(t_test, x_test, label="Test", s=10, color="tab:orange", edgecolors="black", zorder=3)
        ax.plot(t_test, x_forecast_median, label=f"AR({p})", color="black", zorder=4)
        ax.fill_between(t_test, x_forecast_pi_lower, x_forecast_pi_higher, label=f"{prediction_interval:.0%} PI",
                        color="tab:blue", edgecolors="none", alpha=0.3, zorder=2)

        # Print RMSE
        ax.text(x=0.02, y=0.96, s=message, va='top', ha='left', transform=ax.transAxes, fontname="monospace",
                fontsize=7)

        y_max = max(y_max, np.max(x_test), np.max(x_forecast_pi_higher))

        # print(f"{p = :2d} | {batch_id_test = :4d}")

    # Apply consistent axis limits to all subplots
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
                axes[i, j].set_ylabel(feature)
            else:
                axes[i, j].set_yticks([])

            if i == 0 and j == 0:
                axes[i, j].legend(loc="upper right", fontsize=7)

    fig.savefig(os.path.join("figures", f"figure_04_p{p:02d}_N{n_simulations:06d}.png"))
    plt.close(fig)

    rmse_mean = np.mean(rmse_forecasts)
    rmse_sd = np.std(rmse_forecasts)
    print(f"{p = :2d} | RMSE (Forecast): {rmse_mean:6.2f} +/- {rmse_sd:6.2f} {units}")
