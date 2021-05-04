import numpy as np
import pandas as pd


def fillna(X, value=None, method=None, axis=None, limit=None, downcast=None):
    """Impute missing values.

    This function fills the missing values of the input sequence with the next/
    previous known value. If there are contigous NaN values, they will all be
    filled with the same next/previous known value.

    Args:
        X (ndarray or pandas.DataFrame):
            Array of input sequence.
        value:
            Optional. Value to use to fill holes (e.g. 0), alternately a dict/Series/DataFrame of
            values specifying which value to use for each index (for a Series) or column
            (for a DataFrame). Values not in the dict/Series/DataFrame will not be filled.
            This value cannot be a list. Default is None.
        method (str or list):
            Optional. String or list of strings describing whether to use forward or backward
            fill. pad / ffill: propagate last valid observation forward to next valid.
            backfill / bfill: use next valid observation to fill gap. Otherwise use ``None`` to
            fill with desired value. Possible values include
            ``[‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None]``. Default is None.
        axis (int or str):
            Optional. Axis along which to fill missing value. Possible values include 0 or
            "index", 1 or "columns". Default is None.
        limit (int):
            Optional. If method is specified, this is the maximum number of consecutive NaN values
            to forward/backward fill. In other words, if there is a gap with more than this number
            of consecutive NaNs, it will only be partially filled. If method is not specified,
            this is the maximum number of entries along the entire axis where NaNs will be filled.
            Must be greater than 0 if not None. Default is None.
        downcast (dict):
            Optional. A dict of item->dtype of what to downcast if possible, or the string "infer"
            which will try to downcast to an appropriate equal type (e.g. float64 to int64 if
            possible). Default is None.

    Returns:
        ndarray:
            Array of input sequence with imputed values.
    """
    if isinstance(method, str) or method is None:
        method = [method]

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X_ = pd.Series(X)
        else:
            X_ = pd.DataFrame(X)

    else:
        X_ = X.copy()

    for fill in method:
        X_ = X_.fillna(value=value, method=fill, axis=axis, limit=limit, downcast=downcast)

    return X_.values


def rolling_window_sequences_labels(X, index, window_size, target_size=1, step_size=1, target_column=1,
                                    positive_class=1, min_percent=0.01):
    """Create rolling window sequences out of time series data.

    The function creates an array of input sequences and an array of label sequences by rolling
    over the input sequence with a specified window.

    Args:
        X (ndarray):
            N-dimensional sequence to iterate over.
        index (ndarray):
            Array containing the index values of X.
        window_size (int):
            Length of the input sequences.
        target_size (int):
            Length of the target sequences.
        step_size (int):
            Indicating the number of steps to move the window forward each round.
        target_column (int):
            Indicating which column of X is the target.
        positive_class (int or str):
            Indicating which value is considered the positive class in the target column.
        min_percent (float):
            Optional. Indacting the minimum percentage of anomalous values to consider
            the entire window as anomalous.

    Returns:
        ndarray, ndarray, ndarray, ndarray:
            * input sequences.
            * label sequences.
            * first index value of each input sequence.
            * first index value of each label sequence.
    """
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()

    target = X[:, target_column]
    X_ = X[:, :target_column]  # remove label

    start = 0
    max_start = len(X) - window_size - target_size + 1
    while start < max_start:
        end = start + window_size

        labels = target[start:end]
        classes, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(classes, counts))
        if class_counts.get(positive_class, 0) / sum(class_counts.values()) > min_percent:
            out_y.append([positive_class])
        else:
            out_y.append([1 - positive_class])

        out_X.append(X_[start:end])
        X_index.append(index[start])
        y_index.append(index[end])
        start = start + step_size

    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)
