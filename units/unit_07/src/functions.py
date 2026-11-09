import sys

import numpy as np


def progress_bar(c_iter, n_iters, n_updates=1000, length_bar=20):
    update_every = max(1, round(n_iters / n_updates))
    if c_iter % update_every == 0:
        progress = c_iter / n_iters
        length_fill = round(length_bar * progress)
        length_void = length_bar - length_fill
        sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + length_void * " " + "]")


def prepare_data(x, group_ids, seq_length):
    x_sequences = []
    y_sequences = []
    g_sequences = []

    u_group_ids = np.unique(group_ids)

    for group_id in u_group_ids:
        mask = (group_ids == group_id)
        x_group = x[mask]

        for i in range(x_group.shape[0] - seq_length):
            idx_start = i
            idx_end = i + seq_length

            x_seq = x_group[idx_start + 0:idx_end + 0]
            y_seq = x_group[idx_start + 1:idx_end + 1]

            x_sequences.append(x_seq)
            y_sequences.append(y_seq)
            g_sequences.append(group_id)

    x_sequences = np.array(x_sequences)
    y_sequences = np.array(y_sequences)
    g_sequences = np.array(g_sequences)

    return x_sequences, y_sequences, g_sequences
