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
    greater than 0 and contains the dimension required
    
    Args:
        X (ndarray): 
            N-dimensional value sequence to iterate over
        dim (int):
            Integer indicating the dimension number for a multi-dimensional dataset
    
    Returns:
        ndarray:
            Returns an nd array that contains a dataset with 2 columns ['timestamp', 'value']
    """
    if (len(X) == 0):
        return []
    
    columns = X.columns.values
    
    if 'timestamp' not in columns:
        X['timestamp'] = X.index.values
    
    if dim != None:
        if dim in columns:
            X['value'] = X[dim]
            X = pd.DataFrame().assign(timestamp=X['timestamp'], value=X[dim])
    
    if 'value' not in X.columns.values:
        return []          
    
    return X[['timestamp', 'value']]


def rolling_std_thres(X, thres, op = ">", window_size=5):
    """Apply moving standard deviation thesholding.

    The function flags anomalies based on moving standard deviation thresholding

    Args:
        X (ndarray): 
            N-dimensional value sequence to iterate over.
        thres (float):
            Integer used to indicate the threshold of the function
        op (str):
            String indicating the operator used to compare with the threshold.
            Possible values are '<', '>', '<=', '>=', '=='
        window_size (int):
            Integer indicating the number of observations used for each window

    Returns:
        ndarray:
            Dataframe containing the timestamp and value of the flagged indices
    """ 
    a = X['value'].rolling(window=window_size).std().values
    idx_arr = [idx for idx in range(len(a)) if ops[op](a[idx],  thres)]
    return X.loc[idx_arr]


def diff_thres(X, thres = "0.1", op = ">"):
    """Apply discrete difference thresholding.

    The function flags anomalies based on n-th discrete difference thresholding

    Args:
        X (ndarray): 
            N-dimensional value sequence to iterate over.
        thres (float):
            Integer used to indicate the threshold of the function
        op (str):
            String indicating the operator used to compare with the threshold.
            Possible values are '<', '>', '<=', '>=', '=='

    Returns:
        ndarray:
            Dataframe containing the timestamp and value of the flagged indices
    """
    a = np.diff(X['value'])
    idx_arr = [idx for idx in range(len(a)) if ops[op](a[idx],  thres)]
    return X.loc[idx_arr]


def thresholding(X, thres, op):
    """Apply simple thresholding.

    The function flags anomalies based on simple thresholding

    Args:
        X (ndarray): 
            N-dimensional value sequence to iterate over.
        thres (float):
            Integer used to indicate the threshold of the function
        op (str):
            String indicating the operator used to compare with the threshold.
            Possible values are '<', '>', '<=', '>=', '=='

    Returns:
        list:
            integers indicating the timestamps that were flagged
    """
    a = X['value']
    idx_arr = [idx for idx in range(len(a)) if ops[op](a[idx],  thres)]
    return X.loc[idx_arr]
    
    
def get_intervals(y, severity=True):
    """Group together consecutive anomalies in anomaly internals.

    The function groups together samples that have been consecutively flagged
    as an anomaly and returns the lower and upper bound of the boundary.
    Optionally, it computes the average severity of each interval.

    Args:
        y (ndarray): 
            N-dimensional array containing the flagged anomalies of the dataset
        thres (bool):
            Optional. Indicates whether the average severity of each interval
            should be calculated

    Returns:
        ndarray:
            Array containing the anomaly intervals
    """
    intervals = np.split(y, np.where(np.diff(y.index.values) > 1)[0] + 1)
    if(severity):
        return [(interval['timestamp'].values[0], interval['timestamp'].values[-1], np.mean(interval['value'])) for interval in intervals]
    else:
        return [(interval['timestamp'].values[0], interval['timestamp'].values[-1]) for interval in intervals]