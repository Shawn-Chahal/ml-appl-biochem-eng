# ============================================================================
#   Lagged Values
# ============================================================================
#
#   * Time series datasets contain observations collected sequentially
#     over time.
#
#   * In time series forecasting, the current value of a variable often
#     depends on its lagged (i.e., past) values.
#
#   * Lagged values are past measurements included as input features
#     to help a model learn temporal relationships.
#
#   * For example, if p = 3, a model can use:
#
#           x_(t-1), x_(t-2), x_(t-3)
#
#     to predict the current value:
#
#           x_(t)
#
# ============================================================================

import os

import numpy as np
import pandas as pd

# Import dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series.csv"))

# Extract the unique batch IDs.
batch_ids = np.unique(df.loc[:, "Batch ID"])

# Select the time series feature of interest
feature = "Biomass [g/L]"

# Number of lagged values to include
p = 3

# We can store lagged values and current values from each batch in a list
x_lag = []
x_cur = []

# We will process each batch separately. Each batch is treated independently because
# lagged values should not connect observations from different batches.
for batch_id in batch_ids:

    # Extract data corresponding to the current batch_id
    mask = df.loc[:, "Batch ID"] == batch_id
    data_batch = df.loc[mask, feature].to_numpy()

    # Current values: x_(t)
    # The first p values are excluded because they do not have enough
    # past observations to create p lagged values.
    current_values = data_batch[p:]

    # Construct lagged values. E.g.,
    #   Column 1 -> x_(t-1)
    #   Column 2 -> x_(t-2)
    #   Column 3 -> x_(t-3)
    #   Column i -> x_(t-i)
    lagged_values = []
    for i in range(1, p + 1):
        lagged_values.append(data_batch[p - i:-i])

    # Combine individual lagged columns into a single numpy array
    lagged_values = np.stack(lagged_values, axis=1)

    # Append the arrays from this batch into the overall list of
    # lagged values and current values
    x_lag.append(lagged_values)
    x_cur.append(current_values)

# Combine all batches into a single lagged values array and
# single current values array
x_lag = np.concatenate(x_lag, axis=0)
x_cur = np.concatenate(x_cur, axis=0)

# ==========================
#  Print to Terminal
# ==========================

print()
print(f"{x_lag.shape = }")
print(f"{x_cur.shape = }")
print()

message = f"{'x_(t)':>8s} | "

for j in range(1, p + 1):
    col = f"x_(t-{j})"
    message += f"{col:>8s} "

print(message)

for i in range(x_lag.shape[0]):
    message = f"{x_cur[i]:8.3f} | "

    for j in range(p):
        message += f"{x_lag[i, j]:8.3f} "

    print(message)
