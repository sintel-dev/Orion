import numpy as np


def build_anomaly_intervals(X, y, time_column, severity=True):
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
    intervals = list()
    values = list()
    for value, timestamp in zip(y, timestamps):
        if value != 0:
            if start is None:
                start = timestamp
            if severity:
                values.append(value)

        elif start is not None:
            if severity:
                intervals.append((start, timestamp, np.mean(values)))
                values = list()
            else:
                intervals.append((start, timestamp))

            start = None

    # We might have an open interval at the end
    if start is not None:
        if severity:
            intervals.append((start, timestamp, np.mean(values)))
        else:
            intervals.append((start, timestamp))

    return np.array(intervals)
