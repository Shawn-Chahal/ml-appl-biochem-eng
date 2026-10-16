import numpy as np


def generate_x_lag_cur_ar(p, batch_ids, df, feature, col_batch="Batch ID", col_time="Time [h]"):
    """
    Generate lagged autoregressive features and corresponding target values.

    This utility converts a time series signal into a supervised learning
    dataset suitable for autoregressive (AR) modelling.

    For each observation x_(t), the function constructs a feature vector
    containing the previous `p` observations:

        [x_(t-1), x_(t-2), ..., x_(t-p)]

    and uses the current value x_(t) as the target.

    Unlike many generic lag-generation functions, this implementation
    processes each batch independently. Lagged values are never allowed
    to cross batch boundaries, which prevents information leakage between
    separate production runs, experiments, trajectories, or time series
    segments.

    Example
    -------
    For a batch with:

        x = [10, 20, 30, 40, 50]
        p = 2

    The generated dataset becomes:

        x_lag =
        [[20, 10],
         [30, 20],
         [40, 30]]

        x_cur =
        [30, 40, 50]

    where:

        x_lag[i] = [x[i+p-1], x[i+p-2]]
        x_cur[i] = x[i+p]

        x_lag[:, 0] -> x_(t-1)
        x_lag[:, 1] -> x_(t-2)
        x_cur       -> x_(t)

    Note:
        The first p observations cannot be targets because there
        are not enough historical samples available to construct
        p lagged terms.

    Parameters
    ----------
    batch_ids : iterable
        Collection of batch identifiers to process.

    p : int
        Autoregressive order (number of lagged observations).

    df : pandas.DataFrame
        DataFrame containing the time series data and a
        "Batch ID" column.

    feature : str
        Name of the column containing the feature to be modelled.

    Returns
    -------
    x_lag : np.ndarray of shape (n_samples, p)
        Matrix of autoregressive lag features.

    x_cur : np.ndarray of shape (n_samples,)
        Target values corresponding to each lagged feature vector.

    Notes
    -----
    A key design decision is that batches are treated independently.

    Consider two batches:

        Batch A: [1, 2, 3]
        Batch B: [10, 11, 12]

    A naive lag-generation approach applied to the combined sequence
    would incorrectly generate transitions such as:

        [3] -> 10

    which implies temporal continuity between batches.

    This function avoids that problem by generating lagged observations
    separately within each batch before combining the results.
    """

    # Store lagged predictors (x_lag) and current values (x_cur) from each batch separately before combining them.
    x_lag = []
    x_cur = []
    t_cur = []

    # Process each batch independently.
    for batch_id in batch_ids:

        # Select rows belonging to the current batch and extract the feature of interest as a NumPy array.
        mask = df.loc[:, col_batch] == batch_id
        data_batch = df.loc[mask, feature].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()

        # Current observations x_(t)
        current_values = data_batch[p:]
        current_times = t_batch[p:]

        # Build autoregressive lag features, e.g., x_(t-1), x_(t-2), ..., x_(t-p).
        lagged_values = []
        for i in range(1, p + 1):
            # Extract the i-th lag, i.e., x_(t-i)
            lagged_values.append(data_batch[p - i:-i])

        # Combine individual lag vectors into a feature matrix where each column represents one lag order.
        lagged_values = np.stack(lagged_values, axis=1)

        # Store this batch's feature matrix and target vector.
        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)

    # Concatenate the independently generated batches into overall input (x_lag) and output (x_cur) arrays.
    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)

    return x_lag, x_cur, t_cur


def generate_x_lag_cur_var(p, batch_ids, df, features, col_batch="Batch ID", col_time="Time [h]"):
    """
    Generate lagged vector autoregressive features and corresponding target values.

    This function converts a multivariable time series dataset into a
    supervised learning dataset suitable for vector autoregression (VAR).

    In a VAR(p) model, the current values of all variables are predicted
    using the previous p observations of all variables.

    For each observation vector:

        x(t) = [x1(t), x2(t), ..., xm(t)]

    the function constructs a feature vector containing:

        [x(t-1), x(t-2), ..., x(t-p)]

    where each lag contains all variables.

    Unlike many generic lag-generation functions, this implementation
    processes each batch independently. Lagged values are never allowed
    to cross batch boundaries, which prevents information leakage
    between separate production runs, experiments, trajectories, or
    time series segments.

    Example
    -------
    For two variables:

        A = [1, 2, 3, 4]
        B = [10, 20, 30, 40]

    and

        p = 2

    the state vectors become:

        x(0) = [1, 10]
        x(1) = [2, 20]
        x(2) = [3, 30]
        x(3) = [4, 40]

    The generated dataset is:

        x_lag =
        [[ 2, 20,  1, 10],
         [ 3, 30,  2, 20]]

        x_cur =
        [[ 3, 30],
         [ 4, 40]]

    where:

        x_cur         --> x(t)
        x_lag[:, 0:2] --> x(t-1)
        x_lag[:, 2:4] --> x(t-2)


    Parameters
    ----------
    p : int
        VAR order (number of lagged observations).

    batch_ids : iterable
        Collection of batch identifiers to process.

    df : pandas.DataFrame
        DataFrame containing the time series data and a batch identifier column.

    features : iterable of str
        Names of the variables included in the VAR model.

    col_batch : str, default="Batch ID"
        Column identifying individual batches.

    col_time : str, default="Time [h]"
        Column containing time values.

    Returns
    -------
    x_lag : np.ndarray of shape (n_samples, n_features * p)
        Matrix containing lagged predictor variables.

    x_cur : np.ndarray of shape (n_samples, n_features)
        Matrix containing the current values of all variables.

    t_cur : np.ndarray of shape (n_samples,)
        Time values corresponding to each target observation.

    """

    # Store lagged predictors (x_lag), current values (x_cur), and corresponding times (t_cur)
    # from each batch separately before combining them.
    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []

    # Process each batch independently.
    for batch_id in batch_ids:
        # Select rows belonging to the current batch and extract all variables included in the VAR model.
        mask = df.loc[:, col_batch] == batch_id
        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()
        b_batch = df.loc[mask, col_batch].to_numpy()

        # Current observations x(t).
        # Each row contains the simultaneous values of all variables at a particular time point.
        current_values = x_batch[p:]
        current_times = t_batch[p:]
        current_batches = b_batch[p:]

        # Build lagged state vectors:
        #
        #     x(t-1), x(t-2), ..., x(t-p)
        #
        # where each lag contains all process variables.
        lagged_values = []
        for i in range(1, p + 1):
            # Extract the i-th lagged state.
            lagged_values.append(x_batch[p - i:-i])

        # Combine all lagged state vectors into a single feature matrix.
        #
        # For example, with two variables and p = 3:
        #
        #     [x1(t-1), x2(t-1), x1(t-2), x2(t-2), x1(t-3), x2(t-3)]
        #
        # Each row therefore contains the lag history used to predict the current state.
        lagged_values = np.concatenate(lagged_values, axis=1)

        # Store this batch's feature matrix, target matrix, and time values.
        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)

    # Concatenate the independently generated batches into overall input and output arrays.
    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur
