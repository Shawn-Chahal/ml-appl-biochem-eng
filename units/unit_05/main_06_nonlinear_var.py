# ============================================================================
#   Non-linear Vector Autoregression
# ============================================================================
#
#   * Non-linear vector autoregression (NVAR) extends VAR
#     to allow for the use of non-linear models
#
#   * NVAR allows for greater flexibility in modelling complex temporal
#     patterns and interactions between process variables
#
#   * In this script we will use polynomial and k-NN regression to create
#     NVAR models.
#
# ============================================================================

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from units.unit_05.src.classes import StandardScalerVAR
from units.unit_05.src.functions import generate_x_lag_cur_var

# Define training parameters
n_simulations = 1000
p_max = 5
cv_splits = 3
poly_degree_max = 3
knn_k_max = 5

# Fraction of simulated forecasts used to calculate prediction intervals
prediction_interval = 0.95
alpha = 1 - prediction_interval

# List of non-linear models to explore
model_types = ["poly", "knn"]

# Define our GridSearchCV parameters.
# We skip degree = 1 to save on compute time and to avoid repeateing a purely linear model.
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(2, poly_degree_max + 1)])

# We skip k=1 and weights="distance" to save on compute time and to avoid overfitting.
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(2, knn_k_max + 1)],
                      kneighborsregressor__weights=["uniform"])

# Create a random number generator
rng = np.random.default_rng(0)

# Import the time series dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series.csv"))

# Import information describing the initial conditions of each batch
df_ic = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series_ic.csv"))

# Extract the unique batch IDs
batch_ids = np.unique(df.loc[:, "Batch ID"])

# Select all process variables to include in the NVAR model
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

for model_type in model_types:
    # Evaluate NVAR models with different numbers of lagged values
    for p in range(1, p_max + 1):
        # Create a figure containing one subplot for each set of experimental initial conditions.
        fig, axes = plt.subplots(nrows, ncols, figsize=((29.21 / 2.54), (12.09 / 2.54)), dpi=200, layout="constrained")

        y_min = 0
        y_max = np.max(df.loc[:, feature_plot])
        rmse_forecasts = []

        # Leave-one-batch-out cross-validation
        for batch_id_test in batch_ids:

            # Training batches contain all batches except the test batch
            batch_ids_train = np.array([i for i in batch_ids if i != batch_id_test])

            # Generate the lagged value dataset used to train NVAR(p)
            x_lag_train, x_cur_train, _, b_cur_train = generate_x_lag_cur_var(p=p, batch_ids=batch_ids_train, df=df,
                                                                              features=features)

            a_min = np.min(x_cur_train)
            a_max = np.max(x_cur_train)

            # Create the model
            if model_type == "poly":
                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    PolynomialFeatures(include_bias=False),
                    LinearRegression()
                )
                param_grid = param_grid_poly
            elif model_type == "knn":
                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    KNeighborsRegressor()
                )
                param_grid = param_grid_knn
            else:
                model = None
                param_grid = {}

            # Keep the batches separate in GridSearchCV
            gkf = GroupKFold(n_splits=cv_splits)
            model_gscv = GridSearchCV(model, param_grid=param_grid, cv=gkf, scoring="neg_root_mean_squared_error",
                                      return_train_score=True)

            # Fit the model
            model_gscv.fit(x_lag_train, x_cur_train, groups=b_cur_train)

            # Calculate residual errors on the training data.
            # These residuals will later be sampled during forecast simulations to estimate white noise.
            x_cur_train_pred = model_gscv.predict(x_lag_train)
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
            for _ in range(n_simulations):
                # Initialize the forecast using the known initial conditions.
                x_forecast = [x_initial]

                # Generate forecasts recursively.
                while len(x_forecast) < t_test.shape[0]:

                    x_lag = x_forecast[::-1][:p].copy()

                    # At the beginning of the simulation there may be fewer than p available lagged vectors.
                    # If necessary, repeat the most recent state until p lagged vectors are available.
                    while len(x_lag) < p:
                        x_lag.append(x_lag[-1])

                    # Concatenate all lagged vectors into a single feature vector expected by the NVAR model.
                    x_lag = np.concatenate(x_lag, axis=0)
                    x_lag = np.expand_dims(x_lag, axis=0)

                    # Sample residual errors from the training data.
                    # This assumes the residuals behave like white noise.
                    white_noise = rng.choice(x_cur_train_res, size=x_lag.shape[0])

                    # Predict the next process state baseline
                    x_cur_baseline = model_gscv.predict(x_lag)

                    # Don't let the model predict values outside the range it has been trained on.
                    # k-NN has this behaviour by default. Polynomials do not.
                    x_cur_baseline = np.clip(x_cur_baseline, a_min=a_min, a_max=a_max)

                    # Add white noise
                    x_cur = x_cur_baseline + white_noise

                    # Prevent negative concentration predictions after white noise is added
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

            # Extract the best hyperparameters
            if model_type == "poly":
                best_degree = model_gscv.best_params_["polynomialfeatures__degree"]
                message += f"Degree: {best_degree}"
            elif model_type == "knn":
                best_n_neighbors = model_gscv.best_params_["kneighborsregressor__n_neighbors"]
                best_weight = model_gscv.best_params_["kneighborsregressor__weights"]
                message += f"k={best_n_neighbors}, {best_weight}"

            # Plot the test values, forecast median, and prediction interval
            ax.scatter(t_test, x_test[:, idx_fp], label="Test", s=10, color="tab:orange", edgecolors="black", zorder=3)
            ax.plot(t_test, x_forecast_median[:, idx_fp], label=f"NVAR({p})", color="black", zorder=4)
            ax.fill_between(t_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_higher[:, idx_fp],
                            label=f"{prediction_interval:.0%} PI", color="tab:blue", edgecolors="none", alpha=0.3,
                            zorder=2)

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
        fig.savefig(os.path.join("figures", f"figure_06_{model_type}_p{p:02d}_N{n_simulations:06d}.png"))
        plt.close(fig)

        # Calculate and report the forecast error
        rmse_mean = np.mean(rmse_forecasts)
        rmse_sd = np.std(rmse_forecasts)
        print(f"{model_type:6s} | {p = :2d} | RMSE (Forecast): {rmse_mean:6.2f} +/- {rmse_sd:6.2f} {units}")
