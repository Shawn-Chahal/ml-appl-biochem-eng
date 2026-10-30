# ============================================================================
#   Fully Connected Neural Networks
# ============================================================================
#
#   * Fully connected neural networks (FCNNs) can learn complex non-linear
#     relationships between process features and targets.
#
#   * Network architecture, activation function, and learning rate are
#     important hyperparameters that must be optimized.
#
#   * This example performs a simple grid search search over multiple
#     FCNN architectures and hyperparameter settings.
#
#   * Model performance is evaluated using RMSE and NRMSE on a held-out
#     test set.
#
# ============================================================================

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, ParameterGrid

from units.unit_06.src.classes import FullyConnectedNeuralNetwork

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Load the dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_7factor.csv"))

# Define input features and target variables
features = df.columns[3:]
targets = df.columns[:3]

# Convert DataFrame columns to NumPy arrays
x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define the hyperparameter search space
param_grid = dict(
    hidden_layers=[
        (5,),
        (6, 5, 4),
        (8, 16, 32, 16, 8, 4),
    ],
    activation=["tanh", "relu"],
    learning_rate=[0.001, 0.01],
)

# Train and evaluate every hyperparameter combination.
# ParameterGrid enables GridSearchCV-type behaviour.
for hyperparameters in ParameterGrid(param_grid):

    # Create the neural network
    model = FullyConnectedNeuralNetwork(
        epochs=400,
        batch_size=1024,
        verbose=False,
        random_state=0,
        **hyperparameters
    )

    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Predict the test set targets
    y_pred = model.predict(x_test)

    # Calculate the interquartile range (IQR)
    iqr = np.quantile(y_test, q=0.75, axis=0) - np.quantile(y_test, q=0.25, axis=0)

    # Display the current hyperparameters
    print(100 * "=")
    for hp_name, hp_value in hyperparameters.items():
        print(f"{hp_name:20s}: {hp_value}")

    # Calculate performance metrics for each target variable
    print(80 * "-")
    for j, target in enumerate(targets):
        # Compute RMSE
        rmse = root_mean_squared_error(y_test[:, j], y_pred[:, j])

        # Compute NRMSE using the IQR for normalization
        nrmse = rmse / iqr[j]

        print(f"Target: {target:50s} | NRMSE: {nrmse:10.6f} | RMSE:  {rmse:10.6f}")

    print()
