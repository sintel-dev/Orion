import numpy as np


def ffill(X):
    """Fill NaN values with last known value.

    This function fills the missing values of the input sequence with the last
    known value. If there are contigous NaN values, they will all be filled with
    the same last known value.

    Args:
        X (ndarray):
            Array of input sequence.

        Returns:
        ndarray:
            Array of input sequence with imputed values.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    mask = np.isnan(X)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    X[mask] = X[np.nonzero(mask)[0], idx[mask]]

    return X


def bfill(X):
    """Fill NaN values with next known value.

    This function fills the missing values of the input sequence with the upcoming
    known value. If there are contigous NaN values, they will all be filled with
    the same next known value.

    Args:
        X (ndarray):
            Array of input sequence.

    Returns:
        ndarray:
            Array of input sequence with imputed values.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    mask = np.isnan(X)
    idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    X[mask] = X[np.nonzero(mask)[0], idx[mask]]

    return X


def fillna(X, method="ffill"):
    """Impute missing values.

    This function fills the missing values of the input sequence with the next or
    previous known value. If there are contigous NaN values, they will all be
    filled with the same next or previous known value.

    Args:
        X (ndarray):
            Array of input sequence.
        method (str or list):
            Optional. String describing whether to use forward or backward fill.

    Returns:
        ndarray:
            Array of input sequence with imputed values.
    """
    if isinstance(method, str):
        method = [method]

    dim = X.ndim
    X_ = X.copy()

    if dim == 1:
        X_ = X_.reshape(1, -1)

    for fill in method:
        if fill == "ffill":
            X_ = ffill(X_)

        elif fill == "bfill":
            X_ = bfill(X_)

    if dim == 1:
        X_ = X_.reshape(-1,)

    return X_
