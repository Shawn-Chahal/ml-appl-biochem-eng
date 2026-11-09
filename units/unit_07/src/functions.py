import sys

import numpy as np


def progress_bar(c_iter, n_iters, n_updates=1000, length_bar=20):
    """
    Display a simple text-based progress bar in the terminal.

    Parameters
    ----------
    c_iter : int
        Current iteration number.
    n_iters : int
        Total number of iterations.
    n_updates : int, optional
        Maximum number of progress bar updates displayed during execution.
        This avoids excessive terminal output for large loops.
    length_bar : int, optional
        Length of the progress bar in characters.

    """

    # Determine how frequently the progress bar should be updated
    update_every = max(1, round(n_iters / n_updates))

    # Update only at specified intervals
    if c_iter % update_every == 0:
        progress = c_iter / n_iters

        # Compute the number of filled and empty characters
        length_fill = round(length_bar * progress)
        length_void = length_bar - length_fill

        # Print progress on the same terminal line
        sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + length_void * " " + "]")


def prepare_data(x, group_ids, seq_length):
    """
    Convert time-series data into input-output sequences for LSTM models.

    Parameters
    ----------
    x : numpy.ndarray
        Time-series observations. Rows correspond to time points and
        columns correspond to features or measured variables.
    group_ids : numpy.ndarray
        Identifier for each sample indicating which experimental run,
        batch, bioreactor, etc. it belongs to.
    seq_length : int
        Number of time steps in each input sequence.

    Returns
    -------
    x_sequences : numpy.ndarray
        Input sequences with shape
        (n_sequences, seq_length, n_features).
    y_sequences : numpy.ndarray
        Output sequences shifted forward by one time step relative
        to the inputs.
    g_sequences : numpy.ndarray
        Group identifier associated with each sequence.

    """

    # Store generated sequences
    x_sequences = []
    y_sequences = []
    g_sequences = []

    # Find all unique groups (e.g., independent batches)
    u_group_ids = np.unique(group_ids)

    # Process each group independently
    for group_id in u_group_ids:

        # Select rows belonging to the current group
        mask = (group_ids == group_id)
        x_group = x[mask]

        # Generate sliding windows within the group
        for i in range(x_group.shape[0] - seq_length):
            # Define the start and end indices of the current window
            idx_start = i
            idx_end = i + seq_length

            # Input sequence: [x_t, x_(t+1), ..., x_(t+k-1)]
            x_seq = x_group[idx_start:idx_end]

            # Output sequence: [x_(t+1), x_(t+2), ..., x_(t+k)]
            y_seq = x_group[idx_start + 1:idx_end + 1]

            # Store the generated sequence pair
            x_sequences.append(x_seq)
            y_sequences.append(y_seq)
            g_sequences.append(group_id)

    # Convert Python lists to NumPy arrays
    x_sequences = np.array(x_sequences)
    y_sequences = np.array(y_sequences)
    g_sequences = np.array(g_sequences)

    return x_sequences, y_sequences, g_sequences
