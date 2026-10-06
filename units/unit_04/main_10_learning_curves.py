# ============================================================================
#   Learning Curves
# ============================================================================
#
#   * A learning curve shows how model performance changes as the
#     number of training samples increases.
#
#   * Comparing training and test error can help identify overfitting,
#     underfitting, and whether additional training data is likely to
#     improve model performance.
#
#   * In this example, polynomial regression and k-nearest neighbors
#     (kNN) regression are trained using progressively larger subsets
#     of the dataset.
#
#   * GridSearchCV is used to select the optimal hyperparameters for
#     each training set size before evaluating model performance.
#
#   * The resulting learning curves illustrate how training and test
#     RMSE evolve as more data becomes available.
#
# ============================================================================

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# By default, matplotlib makes math in italics,
# but we want our units in regular.
plt.rcParams['mathtext.default'] = 'regular'

df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_temperature.csv"))

features = df.columns[:1]
target = df.columns[-1]

x_all = df.loc[:, features].to_numpy()
y_all = df.loc[:, target].to_numpy()

# Initialize a random number generator
rng = np.random.default_rng(1)

kwargs_gscv = dict(cv=4, scoring="neg_root_mean_squared_error", return_train_score=True)

param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, 21)])

param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(1, 11)],
                      kneighborsregressor__weights=["uniform", "distance"])

fig_width = 14.39 / 2.54
fig_height = 12.09 / 2.54
fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12
fontsize_text = 8
str_hm1 = r"$h^{-1}$"  # Will show a proper superscript in the figure.
dx = 0.1  # Plot every 0.1 C on the x-axis

c_poly = "tab:blue"
c_knn = "tab:orange"
c_test = "tab:green"

# We are going to generate a learning curve to monitor how RMSE changes with number of training samples
lc_n_training_samples = []
lc_rmse_poly_train = []
lc_rmse_knn_train = []
lc_rmse_poly_test = []
lc_rmse_knn_test = []

for n_samples in range(20, 110, 10):
    # We want to look at the effect of number of training samples on model performance.
    # We will select a random subset of the dataset before splitting into train and test.
    idxs = rng.choice(x_all.shape[0], size=n_samples, replace=False)
    x = x_all[idxs]
    y = y_all[idxs]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model_poly = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())
    model_gscv_poly = GridSearchCV(model_poly, param_grid=param_grid_poly, **kwargs_gscv)
    model_gscv_poly.fit(x_train, y_train)

    model_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())
    model_gscv_knn = GridSearchCV(model_knn, param_grid=param_grid_knn, **kwargs_gscv)
    model_gscv_knn.fit(x_train, y_train)

    x_min = np.min(x)
    x_max = np.max(x)

    # Python's built-in round() function rounds to the nearest integer
    resolution = round((x_max - x_min) / dx) + 1

    # We want to plot the model output at many evenly spaced values
    x_plot = np.expand_dims(np.linspace(x_min, x_max, num=resolution), axis=1)
    y_plot_poly = model_gscv_poly.predict(x_plot)
    y_plot_knn = model_gscv_knn.predict(x_plot)

    rmse_poly_train = root_mean_squared_error(y_train, model_gscv_poly.predict(x_train))
    rmse_knn_train = root_mean_squared_error(y_train, model_gscv_knn.predict(x_train))

    rmse_poly_test = root_mean_squared_error(y_test, model_gscv_poly.predict(x_test))
    rmse_knn_test = root_mean_squared_error(y_test, model_gscv_knn.predict(x_test))

    best_degree = model_gscv_poly.best_params_["polynomialfeatures__degree"]
    best_n_neighbors = model_gscv_knn.best_params_["kneighborsregressor__n_neighbors"]
    best_weight = model_gscv_knn.best_params_["kneighborsregressor__weights"]

    n_training_samples = x_train.shape[0]

    # x_buffer will help determine the limits of the x-axis when plotting
    x_buffer = 0.05 * (x_max - x_min)

    message = (
        f"{n_training_samples} training samples\n"
        f"Poly degree={best_degree}\n"
        f"RMSE Train: {rmse_poly_train:.5f} {str_hm1}\n"
        f"RMSE Test:  {rmse_poly_test:.5f} {str_hm1}\n"
        f"kNN: k={best_n_neighbors}, weights={best_weight}\n"
        f"RMSE Train: {rmse_knn_train:.5f} {str_hm1}\n"
        f"RMSE Test:  {rmse_knn_test:.5f} {str_hm1}"
    )

    lc_n_training_samples.append(n_training_samples)
    lc_rmse_poly_train.append(rmse_poly_train)
    lc_rmse_knn_train.append(rmse_knn_train)
    lc_rmse_poly_test.append(rmse_poly_test)
    lc_rmse_knn_test.append(rmse_knn_test)

    # Note how we are only plotting below here, all other calculations or definitions were done above.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), dpi=200, layout="constrained")

    ax.scatter(x_train, y_train, label="Training samples", s=20, color="grey", marker="o", edgecolors="none",
               alpha=0.3, zorder=1)
    ax.scatter(x_test, y_test, label="Test samples", s=36, color=c_test, marker="D", edgecolors="black",
               linewidth=1.5, alpha=0.9, zorder=3)

    ax.plot(x_plot, y_plot_poly, color=c_poly, label="Model (Poly)", linestyle=":", linewidth=2, zorder=2)
    ax.plot(x_plot, y_plot_knn, color=c_knn, label="Model (kNN)", linestyle="--", linewidth=2, zorder=2)

    ax.set_xlabel(features[0], fontsize=fontsize_xy_label)
    ax.set_ylabel(target.replace("h^-1", str_hm1), fontsize=fontsize_xy_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.legend(loc="upper left", ncols=1, fontsize=fontsize_legend)

    ax.set_xlim(left=x_min - x_buffer, right=x_max + x_buffer)
    ax.hlines(y=0, xmin=x_min - 2 * x_buffer, xmax=x_max + 2 * x_buffer,
              color="black", linewidth=0.5, alpha=0.3, zorder=0)

    # When using ax.text() with transform=ax.transAxes, the (x, y) positions range from
    # (0,0) in the top-left corner to (1,1) in the bottom right corner.
    # va and ha represent vertical and horizontal alignment, respectively.
    ax.text(x=0.02, y=0.71, s=message, va='top', ha='left', transform=ax.transAxes,
            fontname='monospace', fontsize=fontsize_text)

    fig.savefig(os.path.join("figures", f"figure_10_N{n_samples:03d}.png"))
    plt.close(fig)

# Plot the learning curve
fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=200, layout="constrained")
ax.plot(lc_n_training_samples, lc_rmse_poly_train, label="RMSE Train (Poly)", color=c_poly, linestyle="-")
ax.plot(lc_n_training_samples, lc_rmse_poly_test, label="RMSE Test (Poly)", color=c_poly, linestyle="--")
ax.plot(lc_n_training_samples, lc_rmse_knn_train, label="RMSE Train (kNN)", color=c_knn, linestyle="-")
ax.plot(lc_n_training_samples, lc_rmse_knn_test, label="RMSE Test (kNN)", color=c_knn, linestyle="--")
ax.set_xlabel("Number of training samples", fontsize=fontsize_xy_label)
ax.set_ylabel(f"RMSE [{str_hm1}]", fontsize=fontsize_xy_label)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)

fig.savefig(os.path.join("figures", f"figure_10_learning_curve.png"))
plt.close(fig)
