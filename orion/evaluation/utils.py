import numpy as np
import pandas as pd

# helper functions


def from_pandas_contextual(df):
    """ Convert contextual ``pandas.DataFrame`` to list of tuples.

    Args:
        df (DataFrame):
            anomalies, passed as ``pandas.DataFrame``
            containing two columns: start and stop.

    Returns:
        list:
            tuple (start, end) timestamp.

    Raises:
        KeyError:
            If the received ``pandas.DataFrame`` does not contain the required columns.
    """
    require = ['start', 'end']
    columns = df.columns.tolist()

    if all(x in columns for x in require):
        if 'severity' in columns:
            return list(df[require + ['severity']].itertuples(index=False))
        return list(df[require].itertuples(index=False))

    raise KeyError('{} not found in columns: {}.'.format(require, columns))


def from_list_points_timestamps(timestamps, gap=1):
    """ Convert list of timestamps to list of tuples.

    Convert a list of anomalies identified by timestamps,
    to a list of tuples marking the start and end interval
    of anomalies; make it contextually defined.

    Args:
        timestamps (list): contains timestamp of anomalies.
        gap (int): allowed gap between anomalies.

    Returns:
        list:
            tuple (start, end) timestamp.
    """
    timestamps = sorted(timestamps)

    start_ts = 0
    max_ts = len(timestamps) - 1

    anomalies = list()
    break_point = start_ts
    while break_point < max_ts:
        if timestamps[break_point + 1] - timestamps[break_point] <= gap:
            break_point += 1
            continue

        anomalies.append((timestamps[start_ts], timestamps[break_point]))
        break_point += 1
        start_ts = break_point

    anomalies.append((timestamps[start_ts], timestamps[break_point]))
    return anomalies


def from_pandas_points(df):
    """ Convert point ``pandas.DataFrame`` to list of tuples.

    Convert a ``pandas.DataFrame`` of anomalies identified by
    one column (timestamp) to a list of tuples marking the
    start and end interval of anomalies; make it contextually
    defined.

    Args:
        df (DataFrame):
            anomalies, passed as ``pandas.DataFrame``
            containing one column: timestamp.

    Returns:
        list:
            tuple (start, end) timestamp.

    Raises:
        KeyError:
            If the received ``pandas.DataFrame`` does not contain column `timestamp`.
    """
    time_column = 'timestamp'
    columns = df.columns.tolist()
    if time_column not in columns:
        raise KeyError('{} not found in columns: {}.'.format(time_column, columns))

    timestamps = list(df['timestamp'])

    return from_list_points_timestamps(timestamps)


def from_pandas_points_labels(df):
    """ Convert point ``pandas.DataFrame`` to list of tuples.

    Convert a ``pandas.DataFrame`` of labeled data where each
    timestamp is labeled by either 0 or 1 to a list of tuples
    marking the start and end interval of anomalies; make it
    contextually defined.

    Args:
        df (DataFrame):
            anomalies, passed as ``pandas.DataFrame``
            containing two columns: timestamp and label.

    Returns:
        list:
            tuple (start, end) timestamp.

    Raises:
        KeyError:
            If the received ``pandas.DataFrame`` does not contain the required columns.
    """

    require = ['timestamp', 'label']
    columns = df.columns.tolist()
    if not all(x in columns for x in require):
        raise KeyError('{} not found in columns: {}.'.format(require, columns))

    df = df[df['label'] == 1]
    return from_pandas_points(df)


def from_list_points_labels(labels):
    """ Convert list of labels to list of tuples.

    Convert a list of labels to a list of tuples
    marking the start and end interval of anomalies by
    defining a dummy timestamp range for usage.

    Args:
        labels (list): contains binary labels [0, 1].

    Returns:
        list:
            tuple (start, end) timestamp.
    """

    timestamps = np.arange(len(labels))
    return from_pandas_points_labels(pd.DataFrame({"timestamp": timestamps, "label": labels}))
