import math

import numpy as np
import pandas as pd


def regression_errors_nd(y, y_hat, smoothing_window):
    y = y.reshape(y_hat.shape)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    true = [item[0] for item in y.reshape((y_hat.shape[0], -1))]
    for item in y[-1][1:]:
        true.extend(item)

    step_size = 1
    predictions = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)
    y_hat = np.asarray(y_hat)

    for i in range(num_errors):
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])
        if intermediate:
            predictions.append(np.median(np.asarray(intermediate)))

    predictions = np.asarray(predictions)

    errors = abs(pd.Series(np.array(true).flatten()) - pd.Series(np.array(predictions).flatten()))
    # dyu added smooth
    smoothing_window = min(math.trunc(errors.shape[0] * 0.01), 200)
    print(errors.shape[0], smoothing_window)

    errors = pd.Series(errors).rolling(
        smoothing_window,
        center=True,
        min_periods=smoothing_window //
        2).mean().values

    return errors
