""" The evaluator module provides metrics to assess the perfomance of
anomaly detection methods using classification metrics: accuracy, precision,
recall, and f1 score. This class addresses the problem as contextual anomalies.

anomaly = [(start, end)] inclusive interval.
"""

from .common import _accuracy, _any_overlap, _f1_score, _precision, _recall, _weighted_segment


def _contextual_partition(expected, observed, start=None, end=None):
    edges = set()

    if start is not None:
        edges.add(start)

    if end is not None:
        edges.add(end)

    for edge in expected + observed:
        edges.update(edge)

    partitions = list()
    edges = sorted(edges)
    last = edges[0]
    for edge in edges[1:]:
        partitions.append((last, edge))
        last = edge

    expected_parts = list()
    observed_parts = list()
    weights = list()
    for part in partitions:
        weights.append(part[1] - part[0])
        expected_parts.append(_any_overlap(part, expected))
        observed_parts.append(_any_overlap(part, observed))

    return expected_parts, observed_parts, weights


def _pad(lst):
    return [(part[0], part[1] + 1) for part in lst]


def contextual_confusion_matrix(expected, observed, data=None, start=None, end=None, method=None):
    """Compute the confusion matrix between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of tuples):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        observed (DataFrame or list of tuples):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.
        method (function):
            Function for computing confusion matrix (weighted segment, overlap segment).

    Returns:
        tuple:
            number of true negative, false positive, false negative, true positive.
    """

    def ws(x, y, z, w):
        return _weighted_segment(x, y, _contextual_partition, z, w)

    if method is not None:
        function = method
    else:
        function = ws

    if data is not None:
        start = data['timestamp'].min()
        end = data['timestamp'].max()

    if not isinstance(expected, list):
        expected = list(expected[['start', 'end']].itertuples(index=False))
    if not isinstance(observed, list):
        observed = list(observed[['start', 'end']].itertuples(index=False))

    expected = _pad(expected)
    observed = _pad(observed)

    return function(expected, observed, start, end)


def contextual_accuracy(expected, observed, data=None, start=None, end=None, method=None):
    """Compute an accuracy score between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of tuples):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        observed (DataFrame or list of tuples):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.
        method (function):
            Function for computing confusion matrix (weighted segment, overlap segment).

    Returns:
        float:
            Accuracy score between the ground truth and detected anomalies.
    """
    return _accuracy(expected, observed, data, start, end, method, contextual_confusion_matrix)


def contextual_precision(expected, observed, data=None, start=None, end=None, method=None):
    """Compute an precision score between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of tuples):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        observed (DataFrame or list of tuples):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.
        method (function):
            Function for computing confusion matrix (weighted segment, overlap segment).

    Returns:
        float:
            Precision score between the ground truth and detected anomalies.
    """
    return _precision(expected, observed, data, start, end, method, contextual_confusion_matrix)


def contextual_recall(expected, observed, data=None, start=None, end=None, method=None):
    """Compute an recall score between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of tuples):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        observed (DataFrame or list of tuples):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.
        method (function):
            Function for computing confusion matrix (weighted segment, overlap segment).

    Returns:
        float:
            Recall score between the ground truth and detected anomalies.
    """
    return _recall(expected, observed, data, start, end, method, contextual_confusion_matrix)


def contextual_f1_score(expected, observed, data=None, start=None, end=None, method=None):
    """Compute an f1 score between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of tuples):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        observed (DataFrame or list of tuples):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.
        method (function):
            Function for computing confusion matrix (weighted segment, overlap segment).

    Returns:
        float:
            F1 score between the ground truth and detected anomalies.
    """
    return _f1_score(expected, observed, data, start, end, method, contextual_confusion_matrix)


METRICS = {
    'accuracy': contextual_accuracy,
    'f1': contextual_f1_score,
    'recall': contextual_recall,
    'precision': contextual_precision,
}
