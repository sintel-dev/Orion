import operator

import numpy as np
import pandas as pd

ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '==': operator.eq}


def extract_dimension(X, dim=None):
    """Validate data dimension.

    The function checks if the dataset being used is valid i.e has a length
    greater than 0 and contains the dimension required.

    Args:
        X (pd.DataFrame):
            Data to validate and extract dimension from.
        dim (str):
            Column indicating the dimension number for a multi-dimensional dataset

    Returns:
        pd.DataFrame:
            Returns a dataframe that contains a dataset with 2 columns ['timestamp', 'value']
    """
    if len(X) == 0:
        return []

    columns = X.columns.values

    if 'timestamp' not in columns:
        X['timestamp'] = X.index.values
        X = X.reset_index(drop=True)

    if dim is not None and dim in columns:
        X['value'] = X[dim]
        X = pd.DataFrame().assign(timestamp=X['timestamp'], value=X[dim])

    if 'value' not in columns:
        return []

    return X[['timestamp', 'value']]


def rolling_std_thresh(X, thresh, op=">", window_size=5):
    """Apply moving standard deviation thesholding.

    The function flags anomalies based on moving standard deviation thresholding.

    Args:
        X (pd.DataFrame):
            N-dimensional value sequence to iterate over.
        thresh (float):
            Float used to indicate the threshold of the function.
        op (str):
            String indicating the operator used to compare with the threshold.
            Possible values are '<', '>', '<=', '>=', '=='.
        window_size (int):
            Integer indicating the number of observations used for each window.

    Returns:
        list:
            List of indices indicating the timestamps that were flagged.
    """
    a = X['value'].rolling(window=window_size).std().values
    idx_arr = [idx for idx in range(len(a)) if ops[op](a[idx], thresh)]
    return X.iloc[idx_arr]


def diff_thresh(X, thresh=0.1, op=">"):
    """Apply discrete difference thresholding.

    The function flags anomalies based on n-th discrete difference thresholding.

    Args:
        X (ndarray):
            N-dimensional value sequence to iterate over.
        thresh (float):
            Integer used to indicate the threshold of the function.
        op (str):
            String indicating the operator used to compare with the threshold.
            Possible values are '<', '>', '<=', '>=', '=='.

    Returns:
        list:
            List of indices indicating the timestamps that were flagged.
    """
    a = np.diff(X['value'])
    idx_arr = [idx for idx in range(len(a)) if ops[op](a[idx], thresh)]
    return X.iloc[idx_arr]


def thresholding(X, thresh, op):
    """Apply simple thresholding.

    The function flags anomalies based on simple thresholding

    Args:
        X (ndarray):
            N-dimensional value sequence to iterate over.
        thresh (float):
            Integer used to indicate the threshold of the function.
        op (str):
            String indicating the operator used to compare with the threshold.
            Possible values are '<', '>', '<=', '>=', '=='.

    Returns:
        list:
            List of indices indicating the timestamps that were flagged.
    """
    a = X['value']
    idx_arr = [idx for idx in range(len(a)) if ops[op](a[idx], thresh)]
    return X.iloc[idx_arr]


def get_intervals(y, severity=True):
    """Group together consecutive anomalies in anomaly internals.

    The function groups together samples that have been consecutively flagged
    as an anomaly and returns the lower and upper bound of the boundary.
    Optionally, it computes the average severity of each interval.

    Args:
        y (ndarray):
            N-dimensional array containing the flagged anomalies of the dataset.
        severity (bool):
            Optional. Indicates whether the average severity of each interval
            should be calculated.

    Returns:
        ndarray:
            Array containing the anomaly intervals
    """
    intervals = np.split(y, np.where(np.diff(y.index.values) > 1)[0] + 1)

    anomalies = list()
    for interval in intervals:
        timestamp = interval['timestamp'].values

        if severity:
            anomalies.append((timestamp[0], timestamp[-1], np.mean(interval['value'])))

        else:
            anomalies.append((timestamp[0], timestamp[-1]))

    return anomalies
