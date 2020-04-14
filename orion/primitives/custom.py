import numpy as np
import pandas as pd
import math

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

    errors = abs(pd.Series(np.array(true).flatten()) -  pd.Series(np.array(predictions).flatten()))
    # dyu added smooth
    smoothing_window = min(math.trunc(errors.shape[0] * 0.01), 200)
    print(errors.shape[0], smoothing_window)
    
    errors = pd.Series(errors).rolling(smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values
#     errors = pd.Series(errors).rolling(smoothing_window, center=True, min_periods=smoothing_window // 2, win_type='triang').mean().values

    return errors


def rolling_window_sequences_autoencoder(X, index, window_size, step_size, target_size, drop=None, drop_windows=False):
    """Create rolling window sequences for autoencoder"""

    out_X = list()
    X_index = list()

    if drop_windows:
        if hasattr(drop, '__len__') and (not isinstance(drop, str)):
            if len(drop) != len(X):
                raise Exception('Arrays `drop` and `X` must be of the same length.')
        else:
            if isinstance(drop, float) and np.isnan(drop):
                drop = np.isnan(X)
            else:
                drop = X == drop

    start = 0
    while start < len(X) - window_size + 1:
        end = start + window_size

        if drop_windows:
            drop_window = drop[start:end + target_size]
            to_drop = np.where(drop_window)[0]
            if to_drop.size:
                start += to_drop[-1] + 1
                continue

        out_X.append(X[start:end])
        X_index.append(index[start:end])
        start = start + step_size

    #return np.flip(np.asarray(out_X), axis=1), np.asarray(out_X).reshape(-1, window_size), np.asarray(X_index).flatten(), np.asarray(X_index).flatten(), index
    return np.asarray(out_X), np.asarray(out_X).reshape(-1, window_size), np.asarray(X_index).flatten(), np.asarray(X_index).flatten(), index
