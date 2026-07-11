import numpy as np


def make_windows(series, window_size):
    X = []
    Y = []

    for i in range(0, len(series) - 2 * window_size + 1, 2 * window_size):
        X.append(series[i:i + window_size])
        Y.append(series[i + window_size:i + 2 * window_size])

    return np.array(X), np.array(Y)
