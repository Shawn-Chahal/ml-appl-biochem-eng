# ============================================================================
#   White Noise Analysis
# ============================================================================
#
#   * Residual error (or noise) is the difference between the model
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
#   * In this example, the residuals of AR(p) models are evaluated
#     for different values of p to determine whether increasing the
#     number of lagged values improves the fit.
#
# ============================================================================

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from units.unit_05.src.classes import StandardScalerAR
from units.unit_05.src.functions import generate_x_lag_cur_ar

# Import dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series.csv"))

# Extract the unique batch IDs
batch_ids = np.unique(df.loc[:, "Batch ID"])

# Select the time series feature of interest
feature = "Biomass [g/L]"

# Confidence level used to visualize uncertainty
confidence_interval = 0.95
alpha = 1 - confidence_interval

# Define common x-axis limits
x_left = np.min(df.loc[:, "Time [h]"]) - 1
x_right = np.max(df.loc[:, "Time [h]"]) + 1

# Evaluate autoregressive models with different numbers of lagged values.
for p in range(1, 6):

    # Generate the lagged-value dataset used to train AR(p)
    x_lag, x_cur, t_cur = generate_x_lag_cur_ar(p=p, batch_ids=batch_ids, df=df, feature=feature)

    # Create and train the model
    model = make_pipeline(StandardScalerAR(), LinearRegression())
    model.fit(x_lag, x_cur)

    # Note: Here we predicted using the training data because we are evaluating whether the model was able to learn
    # the time series patterns in the dataset. We are NOT evaluating how well it generalizes to new data.
    x_cur_predict = model.predict(x_lag)

    # Calculate the residual errors (noise)
    x_cur_noise = x_cur_predict - x_cur

    # Determine the unique time points in the dataset
    t_unique = np.unique(t_cur)

    # We will store the residual mean and confidence intervals at each time point
    noise_mean = []
    noise_ci_upper = []
    noise_ci_lower = []

    # Analyze the residual distribution at each time point
    for t in t_unique:
        mask = t_cur == t
        x_cur_noise_t = x_cur_noise[mask]
        n_mean = np.mean(x_cur_noise_t)
        n_sd = np.std(x_cur_noise_t)

        # Calculate the confidence interval using the Student t distribution
        q = alpha / 2
        dof = x_cur_noise_t.shape[0] - 1
        ci_sd = stats.t.isf(q=q, df=dof)

        noise_mean.append(n_mean)
        noise_ci_upper.append(n_mean + ci_sd * n_sd)
        noise_ci_lower.append(n_mean - ci_sd * n_sd)

    noise_mean = np.array(noise_mean)
    noise_ci_upper = np.array(noise_ci_upper)
    noise_ci_lower = np.array(noise_ci_lower)

    # Determine the plotting range needed to display all residual statistics symmetrically about zero
    y_max = max(np.max(np.abs(noise_mean)),
                np.max(np.abs(noise_ci_upper)),
                np.max(np.abs(noise_ci_lower)),
                np.max(np.abs(x_cur_noise)))

    # =============================
    #   PLOT
    # =============================
    fig, ax = plt.subplots(1, 1, figsize=((14.39 / 2.54), (12.09 / 2.54)), dpi=200, layout="constrained")

    ax.scatter(t_cur, x_cur_noise, label="Noise", color="tab:orange", s=12, edgecolors="black", linewidth=0.5,
               alpha=0.7, zorder=3)
    ax.plot(t_unique, noise_mean, label="Mean", color="black", zorder=4)
    ax.fill_between(t_unique, noise_ci_lower, noise_ci_upper, label=f"{confidence_interval:.0%} CI", color="tab:blue",
                    edgecolors="none", alpha=0.3, zorder=2)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5, zorder=1)

    ax.set_xlabel("Time [h]", fontsize=14)
    ax.set_ylabel(f"{feature} (Noise)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc="upper left", fontsize=12)

    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlim(left=x_left, right=x_right)
    ax.set_ylim(bottom=-1.05 * y_max, top=1.05 * y_max)

    fig.savefig(os.path.join("figures", f"figure_03_p{p:02d}.png"))
    plt.close(fig)
