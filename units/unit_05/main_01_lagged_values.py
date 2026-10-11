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

from units.unit_05.src.functions import generate_x_lag_cur_ar

# Import dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_time_series.csv"))

# Extract the unique batch IDs.
batch_ids = np.unique(df.loc[:, "Batch ID"])

# Select the time series feature of interest
feature = "Biomass [g/L]"

# Number of lagged values to include
p = 3

# Generate the lagged-value dataset used to train AR(p)
x_lag, x_cur = generate_x_lag_cur_ar(p=p, batch_ids=batch_ids, df=df, feature=feature)

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
