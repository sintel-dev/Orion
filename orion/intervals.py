import numpy as np


def build_anomaly_intervals(X, y, time_column, severity=True, indices=False):
    """Group together consecutive anomalous samples in anomaly intervals.

    This is a dummy boundary detection function that groups together
    samples that have been consecutively flagged as anomalous and
    returns boundaries of anomalous intervals.

    Optionally, it computes the average severity of each interval.

    This detector is here only to serve as reference of what
    an boundary detection primitive looks like, and is not intended
    to be used in real scenarios.
    """

    timestamps = X[time_column]

    start = None
    start_ts = None
    intervals = list()
    values = list()
    for index, (value, timestamp) in enumerate(zip(y, timestamps)):
        if value != 0:
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
