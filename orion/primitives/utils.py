# -*- coding: utf-8 -*-
import numpy as np


def aggregate_rolling_window(y, aggregation="median"):
    """Aggregate a rolling window sequence.

    Convert a rolling window sequence into a flattened time series.
    Use the aggregation specified to make each timestamp a single value.

    Args:
        y (ndarray):
            Windowed sequences. Each timestamp has multiple predictions.
        aggregation (string):
            String denoting the aggregation method to use. Default is "median".

    Return:
        ndarray:
            Flattened sequence.
    """
    window_size = y.shape[1]
    num_windows = y.shape[0]
    seq_length = num_windows + window_size - 1

    method = getattr(np, "nan" + aggregation)

    y = y.reshape(num_windows, window_size,)
    X = np.zeros(shape=(num_windows, seq_length))
    X[:] = np.nan

    for i in range(num_windows):
        X[i, i:i + window_size] = y[i]

    return method(X, axis=0)
