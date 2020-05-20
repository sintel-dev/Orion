import numpy as np
import pandas as pd

# helper functions


def from_pandas_contextual(df):
    """ convert contextual `pd.DataFrame` to list of tuples.

    Args:
        * df (pd.DataFrame): contains start and end.
    """

    require = ['start', 'end']
    columns = df.columns.tolist()

    if all(x in columns for x in require):
        if 'severity' in columns:
            return list(df[require + ['severity']].itertuples(index=False))
        return list(df[require].itertuples(index=False))

    raise KeyError('{} not found in columns: {}.'.format(require, columns))


def from_list_points_timestamps(timestamps, gap=1):
    """ convert list of timestamps to list of tuples.
    make it contextually defined.

    Args:
        * timestamps (list): contains timestamp of anomalies.
        * gap (int): allowed gap between anomalies.
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
    """ convert point `pd.DataFrame` to list of tuples.
    make it contextually defined.

    Args:
        * df (pd.DataFrame): contains timestamp.
    """

    time_column = 'timestamp'
    columns = df.columns.tolist()
    if time_column not in columns:
        raise KeyError('{} not found in columns: {}.'.format(time_column, columns))

    timestamps = list(df['timestamp'])

    return from_list_points_timestamps(timestamps)


def from_pandas_points_labels(df):
    """ convert point `pd.DataFrame` to list of tuples.
    make it contextually defined.

    Args:
        * df (pd.DataFrame): contains timestamp and label.
    """

    require = ['timestamp', 'label']
    columns = df.columns.tolist()
    if not all(x in columns for x in require):
        raise KeyError('{} not found in columns: {}.'.format(require, columns))

    df = df[df['label'] == 1]
    return from_pandas_points(df)


def from_list_points_labels(labels):
    """ convert list of labels to list of tuples.
    make it contextually defined.

    define a dummy timestamp range for usage.
    assumes both ground truth and predicted anomalies
    cover the same range.

    Args:
        * labels (list): contains label.
    """

    timestamps = np.arange(len(labels))
    return from_pandas_points_labels(pd.DataFrame({"timestamp": timestamps, "label": labels}))
