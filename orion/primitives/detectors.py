import numpy as np


class ThresholdDetector:
    """Threshold Detector.

    This is a dummy anomaly detector that classifies as an anomaly
    any value that is further away from the expected value than
    the maximum distance seen during the fit phase.

    Optionally, it also returns the severity of the anomaly, which
    is simply the difference between the actual value and the maximum
    expected value.

    This detector is here only to serve as reference of what
    an anomaly detection primitive looks like, and is not intended
    to be used in real scenarios.
    """

    def __init__(self, value_column, ratio, severity=True):
        if ratio < 0 or ratio > 1:
            raise ValueError('`ratio` must be a value between 0 and 1')

        self._value_column = value_column
        self._ratio = ratio
        self._severity = severity

    def fit(self, X, y):
        truth = X[self._value_column]
        diffs = np.abs(truth - y)
        max_diff = diffs.max()
        self.threshold_ = self._ratio * max_diff

    def detect(self, X, y):
        truth = X[self._value_column]
        diffs = np.abs(truth - y)
        if not self._severity:
            return (diffs > self.threshold_).astype(int)

        else:
            over_threshold = diffs - self.threshold_
            return np.maximum(over_threshold, np.zeros(len(X)))
