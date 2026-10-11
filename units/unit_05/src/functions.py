import numpy as np


def generate_x_lag_cur_ar(batch_ids, p, df, feature):
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
        #   Column 0 -> x_(t-1)
        #   Column 1 -> x_(t-2)
        #   Column 2 -> x_(t-3)
        #   Column i -> x_(t-(i+1))
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

    return x_lag, x_cur
