from orion.evaluation.common import _accuracy, _f1_score, _precision, _recall, _weighted_segment


def _point_partition(expected, observed, start=None, end=None):
    expected = set(expected)
    observed = set(observed)

    edge_start = min(expected.union(observed))
    if start is not None:
        edge_start = start

    edge_end = max(expected.union(observed))
    if end is not None:
        edge_end = end

    length = int(edge_end) - int(edge_start) + 1

    expected_parts = [0] * length
    observed_parts = [0] * length

    for edge in expected:
        expected_parts[edge - edge_start] = 1

    for edge in observed:
        observed_parts[edge - edge_start] = 1

    return expected_parts, observed_parts, None


def point_confusion_matrix(expected, observed, data=None, start=None, end=None):
    """Compute the confusion matrix between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of timestamps):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        observed (DataFrame or list of timestamps):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.

    Returns:
        tuple:
            number of true negative, false positive, false negative, true positive.
    """

    def _ws(x, y, z, w):
        return _weighted_segment(x, y, _point_partition, z, w)

    if data is not None:
        start = data['timestamp'].min()
        end = data['timestamp'].max()

    if not isinstance(expected, list):
        expected = list(expected['timestamp'])
    if not isinstance(observed, list):
        observed = list(observed['timestamp'])

    return _ws(expected, observed, start, end)


def point_accuracy(expected, observed, data=None, start=None, end=None):
    """Compute an accuracy score between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of timestamps):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        observed (DataFrame or list of timestamps):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.

    Returns:
        float:
            Accuracy score between the ground truth and detected anomalies.
    """
    return _accuracy(expected, observed, data, start, end, cm=point_confusion_matrix)


def point_precision(expected, observed, data=None, start=None, end=None):
    """Compute an precision score between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of timestamps):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        observed (DataFrame or list of timestamps):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.

    Returns:
        float:
            Precision score between the ground truth and detected anomalies.
    """
    return _precision(expected, observed, data, start, end, cm=point_confusion_matrix)


def point_recall(expected, observed, data=None, start=None, end=None):
    """Compute an recall score between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of timestamps):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        observed (DataFrame or list of timestamps):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.

    Returns:
        float:
            Recall score between the ground truth and detected anomalies.
    """
    return _recall(expected, observed, data, start, end, cm=point_confusion_matrix)


def point_f1_score(expected, observed, data=None, start=None, end=None):
    """Compute an f1 score between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of timestamps):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        observed (DataFrame or list of timestamps):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            one column: timestamp.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.

    Returns:
        float:
            F1 score between the ground truth and detected anomalies.
    """
    return _f1_score(expected, observed, data, start, end, cm=point_confusion_matrix)
