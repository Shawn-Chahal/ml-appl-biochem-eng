# ============================================================================
#   Hyperparameter Optimization
# ============================================================================
#
#   * Hyperparameters are model settings that must be chosen before
#     training and can strongly influence model performance.
#
#   * Grid search systematically evaluates multiple hyperparameter values
#     to identify which produce the best cross-validation score.
#
#   * Hyperparameter optimization helps balance model complexity and
#     generalization to improve performance on unseen data.
#
# ============================================================================

import os

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

df = pd.read_csv(os.path.join("datasets", "dataset_protein_yield.csv"))

features = df.columns[:-1]
targets = df.columns[-1]

x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# We will want to try different values for "degree" in PolynomialFeatures so we will leave it undefined here.
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(include_bias=False),
    LinearRegression()
)

# We will use a list comprehnesion to try every degree value from 1 to 10
param_grid = dict(polynomialfeatures__degree=[i for i in range(1, 11)])

# Create the model using GridSearchCV
model_gscv = GridSearchCV(model, param_grid=param_grid,
                          cv=5, scoring="neg_root_mean_squared_error", return_train_score=True)

# Fit the model
model_gscv.fit(x_train, y_train)

# Place the results in a DataFrame
df = pd.DataFrame(model_gscv.cv_results_)

# Define the columns of interest
columns_view = ["rank_test_score", "param_polynomialfeatures__degree",
                "mean_test_score", "std_test_score",
                "mean_train_score", "std_train_score"]

# Print the DataFrame
print("\nRecall that we used negative RMSE, therefore a higher score (i.e., closer to zero) is better:\n")
print(df.loc[:, columns_view].to_string())
print()

# Print the final test score on data that was never seen by GridSearchCV
print(f"Final Test negative RMSE: {model_gscv.score(x_test, y_test):.6f} mg/L\n")
print("Note how it is slightly worse than the best mean_test_score in the GridSearchCV results.")
print()

# =========
#   PLOT
# =========

LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54

fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12

plot_n = 5
degrees = df.loc[:, "param_polynomialfeatures__degree"].to_numpy()
rmse_train = - df.loc[:, "mean_train_score"].to_numpy()
rmse_test = - df.loc[:, "mean_test_score"].to_numpy()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

ax.plot(degrees[:plot_n], rmse_train[:plot_n], label=f"Train", linestyle=LINESTYLES[0])
ax.plot(degrees[:plot_n], rmse_test[:plot_n], label=f"Test", linestyle=LINESTYLES[1])
ax.set_xlabel("Polynomial Degree", fontsize=fontsize_xy_label)
ax.set_ylabel("Protein Yield RMSE [mg/L]", fontsize=fontsize_xy_label)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(MultipleLocator(1))

fig.savefig(os.path.join("figures", "figure_02.png"))
plt.close(fig)
