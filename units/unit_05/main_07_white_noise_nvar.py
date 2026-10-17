# ============================================================================
#   White Noise Analysis (NVAR)
# ============================================================================
#
#   * Residual error is the difference between the model
#     prediction and the observed value.
#
#   * A good time series model should leave only random residuals
#     centered around zero over time (white noise).
#
#   * Patterns remaining in the residuals suggest that the model
#     has not captured all of the information in the data.
#
#   * White noise should:
#         - Show no systematic trends over time
#         - Have a mean close to zero
#         - Appear randomly distributed around zero
#
#   * In this example, the residuals of NVAR models are evaluated
#     to determine whether they have successfully captured the
#     relationships between the process variables and their lagged
#     values.
#
# ============================================================================


import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from units.unit_05.src.classes import StandardScalerVAR
from units.unit_05.src.functions import generate_x_lag_cur_var

# Define training parameters
p_max = 5
cv_splits = 3
poly_degree_max = 3
knn_k_max = 5

# List of non-linear models to explore
model_types = ["poly", "knn"]

# Define our GridSearchCV parameters.
# We skip degree = 1 to save on compute time and to avoid repeateing a purely linear model.
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(2, poly_degree_max + 1)])

# We skip k=1 and weights="distance" to save on compute time and to avoid overfitting.
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(2, knn_k_max + 1)],
                      kneighborsregressor__weights=["uniform"])

# Import dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series.csv"))

# Extract the unique batch IDs
batch_ids = np.unique(df.loc[:, "Batch ID"])

# Select all process variables to include in the NVAR model
features = df.columns[2:]

# Select the variable that will be plotted and evaluated
feature_plot = "Biomass [g/L]"
idx_fp = np.argmax(features == feature_plot)

# Extract units for plotting and reporting
units = feature_plot.split(" [")[-1][:-1]

# Fraction of residuals expected to lie within the prediction interval.
prediction_interval = 0.95
alpha = 1 - prediction_interval

# Define common x-axis limits
x_left = np.min(df.loc[:, "Time [h]"]) - 1
x_right = np.max(df.loc[:, "Time [h]"]) + 1

# Evaluate each non-linear model type separately.
for model_type in model_types:

    # Evaluate NVAR models with different numbers of lagged values.
    for p in range(1, p_max + 1):

        # Generate the lagged-value dataset used to train NVAR(p)
        x_lag, x_cur, t_cur, b_cur = generate_x_lag_cur_var(p=p, batch_ids=batch_ids, df=df, features=features)

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
        model_gscv.fit(x_lag, x_cur, groups=b_cur)

        # Note: Here we predicted using the training data because we are evaluating whether the model was able to learn
        # the time series patterns in the dataset. We are NOT evaluating how well it generalizes to new data.
        x_cur_pred = model_gscv.predict(x_lag)

        # Calculate the residual errors (noise)
        x_cur_res = x_cur_pred - x_cur

        # Determine the unique time points in the dataset
        t_unique = np.unique(t_cur)

        # We will store the residual mean and prediction intervals at each time point
        res_mean = []
        res_pi_upper = []
        res_pi_lower = []

        # Analyze the residual distribution at each time point for the feature of interest
        for t in t_unique:
            mask = t_cur == t
            x_cur_res_t = x_cur_res[mask, idx_fp]
            n_mean = np.mean(x_cur_res_t)
            n_sd = np.std(x_cur_res_t)

            # Calculate the prediction interval using Student's t distribution
            q = alpha / 2
            dof = x_cur_res_t.shape[0] - 1
            pi_sd = stats.t.isf(q=q, df=dof)

            res_mean.append(n_mean)
            res_pi_upper.append(n_mean + pi_sd * n_sd)
            res_pi_lower.append(n_mean - pi_sd * n_sd)

        res_mean = np.array(res_mean)
        res_pi_upper = np.array(res_pi_upper)
        res_pi_lower = np.array(res_pi_lower)

        # Determine the plotting range needed to display all residual statistics symmetrically about zero
        y_max = max(np.max(np.abs(res_mean)),
                    np.max(np.abs(res_pi_upper)),
                    np.max(np.abs(res_pi_lower)),
                    np.max(np.abs(x_cur_res[:, idx_fp])))

        fig, ax = plt.subplots(1, 1, figsize=((14.39 / 2.54), (12.09 / 2.54)), dpi=300, layout="constrained")

        ax.scatter(t_cur, x_cur_res[:, idx_fp], label="Noise", color="tab:orange", s=12, edgecolors="black",
                   linewidth=0.5,
                   alpha=0.7, zorder=3)
        ax.plot(t_unique, res_mean, label="Mean", color="black", zorder=4)
        ax.fill_between(t_unique, res_pi_lower, res_pi_upper, label=f"{prediction_interval:.0%} PI", color="tab:blue",
                        edgecolors="none", alpha=0.3, zorder=2)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5, zorder=1)

        ax.set_xlabel("Time [h]", fontsize=14)
        ax.set_ylabel(f"{feature_plot} (Residuals)", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc="upper left", fontsize=12)

        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.set_xlim(left=x_left, right=x_right)
        ax.set_ylim(bottom=-1.05 * y_max, top=1.05 * y_max)

        fig.savefig(os.path.join("figures", f"figure_07_{model_type}_p{p:02d}.png"))
        plt.close(fig)

        if model_type == "poly":
            best_degree = model_gscv.best_params_["polynomialfeatures__degree"]
            print(f"{p = } | Degree: {best_degree}")
        elif model_type == "knn":
            best_n_neighbors = model_gscv.best_params_["kneighborsregressor__n_neighbors"]
            best_weight = model_gscv.best_params_["kneighborsregressor__weights"]
            print(f"{p = } | k={best_n_neighbors}, {best_weight}")
