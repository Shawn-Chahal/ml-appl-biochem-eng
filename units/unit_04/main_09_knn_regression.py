# ============================================================================
#   k-Nearest Neighbors (kNN) Regression
# ============================================================================
#
#   * kNN regression predicts the target value of new samples by taking
#     the (weighted) average of the values of the k nearest training
#     samples.
#
#   * Feature scaling is often important because distance-based models
#     are sensitive to differences in feature scale.
#
#   * The choice of k controls model complexity:
#       - Smaller k values can capture local patterns but may overfit.
#       - Larger k values produce smoother predictions but may underfit.
#
# ============================================================================

import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(os.path.join("datasets", "dataset_protein_yield.csv"))

features = df.columns[:-1]
targets = df.columns[-1]

x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor()
)

# Define the "Grid" that we want to "Search" when performing "CV" in GridSearchCV
param_grid = dict(
    kneighborsregressor__n_neighbors=[i for i in range(1, 81)],  # Try every value of n_neighbors from 1 to 80
    kneighborsregressor__weights=["uniform", "distance"],  # Try both weight algorithms
)

# Create the model using GridSearchCV
model_gscv = GridSearchCV(model, param_grid=param_grid, cv=4, scoring="neg_root_mean_squared_error",
                          return_train_score=True)

model_gscv.fit(x_train, y_train)

df = pd.DataFrame(model_gscv.cv_results_)

# We can sort the DataFrame by rank_test_score (i.e., best to worst)
df = df.sort_values(by=['rank_test_score'], ascending=True)

# Define the columns of interest with the relevant hyperparameters
columns_view = [
    "rank_test_score",
    "param_kneighborsregressor__n_neighbors",
    "param_kneighborsregressor__weights",
    "mean_test_score", "std_test_score",
    "mean_train_score", "std_train_score"
]

print("\nRecall that we used negative RMSE, therefore a higher score (i.e., closer to zero) is better:\n")
print(df.loc[:, columns_view].to_string())
print()

print(f"Final Test negative RMSE: {model_gscv.score(x_test, y_test):.6f} mg/L\n")
print()

# ==========================
#   Plot Validation Curves
# ==========================

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54

fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

for i, weight in enumerate(param_grid["kneighborsregressor__weights"], start=1):
    mask_weight = (df.loc[:, "param_kneighborsregressor__weights"] == weight)
    df_sub = df.loc[mask_weight, :]
    df_sub = df_sub.sort_values(by=['param_kneighborsregressor__n_neighbors'], ascending=True)
    n_neighbors = df_sub.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy()
    rmse_train = - df_sub.loc[:, "mean_train_score"].to_numpy()
    rmse_test = - df_sub.loc[:, "mean_test_score"].to_numpy()

    ax.plot(n_neighbors, rmse_train, label=f"Train (kNN: {weight})", color=COLORS[i], linestyle="-")
    ax.plot(n_neighbors, rmse_test, label=f"Valid (kNN: {weight})", color=COLORS[i], linestyle="--")

ax.set_xlabel("k", fontsize=fontsize_xy_label)
ax.set_ylabel("Protein Yield RMSE [mg/L]", fontsize=fontsize_xy_label)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)
ax.set_ylim(bottom=0)

fig.savefig(os.path.join("figures", "figure_09_validation_curve_knn.png"))
plt.close(fig)
