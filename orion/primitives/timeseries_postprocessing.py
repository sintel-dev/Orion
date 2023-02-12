import operator
import numpy as np
import pandas as pd

ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '==': operator.eq}

def extract_dimension(X, dim=None):
    """
    The function checks if the dataset being used is valid i.e has a length greater than 0 and contains the dimension required
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
    """
    The function detects anomalies that are flagged through moving standard deviation thresholding
    Args:
        X (ndarray): 
            N-dimensional value sequence to iterate over.
        thres (float):
            Integer used to indicate the threshold of the function
        op (str):
            String indicating the operator used to compare with the threshold. Possible values are '<', '>', '<=', '>=', '=='
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
    """
    The function detects anomalies that are flagged through moving standard deviation thresholding
    Args:
        X (ndarray): 
            N-dimensional value sequence to iterate over.
        thres (float):
            Integer used to indicate the threshold of the function
        op (str):
            String indicating the operator used to compare with the threshold. Possible values are '<', '>', '<=', '>=', '=='

    Returns:
        ndarray:
            Dataframe containing the timestamp and value of the flagged indices

    """

    a = np.diff(X['value'])
    idx_arr = [idx for idx in range(len(a)) if ops[op](a[idx],  thres)]
    return X.loc[idx_arr]


def thresholding(X, thres, op):
    """
    The function detects anomalies that are flagged through moving standard deviation thresholding
    Args:
        X (ndarray): 
            N-dimensional value sequence to iterate over.
        thres (float):
            Integer used to indicate the threshold of the function
        op (str):
            String indicating the operator used to compare with the threshold. Possible values are '<', '>', '<=', '>=', '=='
        window_size (int):
            Integer indicating the number of observations used for each window

    Returns:
        list:
            integers indicating the timestamps that were flagged

    """
    

    a = X['value']
    idx_arr = [idx for idx in range(len(a)) if ops[op](a[idx],  thres)]
    return X.loc[idx_arr]
    
    


def get_intervals(y, severity=True):
    intervals = np.split(y, np.where(np.diff(y.index.values) > 1)[0] + 1)
    if(severity):
        return [(interval['timestamp'].values[0], interval['timestamp'].values[-1], np.mean(interval['value'])) for interval in intervals]
    else:
        return [(interval['timestamp'].values[0], interval['timestamp'].values[-1]) for interval in intervals]
    

def build_anomaly_intervals(y, severity=True, indices=False):
    """Group together consecutive anomalous samples in anomaly intervals.

    This is a dummy boundary detection function that groups together
    samples that have been consecutively flagged as anomalous and
    returns boundaries of anomalous intervals.

    Optionally, it computes the average severity of each interval.

    This detector is here only to serve as reference of what
    an boundary detection primitive looks like, and is not intended
    to be used in real scenarios.
    """

    timestamps = y['timestamp']
    v = y['value']
    start = None
    start_ts = None
    intervals = list()
    values = list()
    for index, (value, timestamp) in enumerate(zip(v, timestamps)):
        #if value != 0:
            if start_ts is None:
                start = index
                start_ts = timestamp
                if severity:
                    values.append(value)

            elif start_ts is not None:
                interval = [start_ts, timestamp]
                if indices:
                    interval.extend([start, index])
                if severity:
                    interval.append(np.mean(values))
                    values = list()

                intervals.append(tuple(interval))

            start = None
            start_ts = None

    # We might have an open interval at the end
    if start_ts is not None:
        interval = [start_ts, timestamp]
        if indices:
            interval.extend([start, index])
        if severity:
            interval.append(np.mean(values))

        intervals.append(tuple(interval))

    return np.array(intervals)
    