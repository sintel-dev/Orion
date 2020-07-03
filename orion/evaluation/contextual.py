from orion.evaluation.common import (
    _accuracy, _any_overlap, _f1_score, _overlap, _precision, _recall, _weighted_segment)


def _overlap_segment(expected, observed, start=None, end=None):
    tp, fp, fn = 0, 0, 0

    observed_copy = observed.copy()

    for expected_seq in expected:
        found = False
        for observed_seq in observed:
            if _overlap(expected_seq, observed_seq):
                if not found:
                    tp += 1
                    found = True
                if observed_seq in observed_copy:
                    observed_copy.remove(observed_seq)

        if not found:
            fn += 1

    fp += len(observed_copy)

    return None, fp, fn, tp


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


def contextual_confusion_matrix(expected, observed, data=None,
                                start=None, end=None, weighted=True):
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
        weighted (boolean):
            Flag to represent which algorithm to use.
            If true use weighted segment algorithm, else use overlap segment.

    Returns:
        tuple:
            number of true negative, false positive, false negative, true positive.
    """

    def _ws(x, y, z, w):
        return _weighted_segment(x, y, _contextual_partition, z, w)

    if weighted:
        function = _ws
    else:
        function = _overlap_segment

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


def contextual_accuracy(expected, observed, data=None, start=None, end=None, weighted=True):
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
        weighted (boolean):
            Flag to represent which algorithm to use.
            If true use weighted segment algorithm, else use overlap segment.

    Returns:
        float:
            Accuracy score between the ground truth and detected anomalies.
    """
    def _cm(x, y, z, w, f):
        return contextual_confusion_matrix(x, y, z, w, f, weighted)
    return _accuracy(expected, observed, data, start, end, _cm)


def contextual_precision(expected, observed, data=None, start=None, end=None, weighted=True):
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
        weighted (boolean):
            Flag to represent which algorithm to use.
            If true use weighted segment algorithm, else use overlap segment.

    Returns:
        float:
            Precision score between the ground truth and detected anomalies.
    """
    def _cm(x, y, z, w, f):
        return contextual_confusion_matrix(x, y, z, w, f, weighted)
    return _precision(expected, observed, data, start, end, _cm)


def contextual_recall(expected, observed, data=None, start=None, end=None, weighted=True):
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
        weighted (boolean):
            Flag to represent which algorithm to use.
            If true use weighted segment algorithm, else use overlap segment.

    Returns:
        float:
            Recall score between the ground truth and detected anomalies.
    """
    def _cm(x, y, z, w, f):
        return contextual_confusion_matrix(x, y, z, w, f, weighted)
    return _recall(expected, observed, data, start, end, _cm)


def contextual_f1_score(expected, observed, data=None, start=None, end=None, weighted=True):
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
        weighted (boolean):
            Flag to represent which algorithm to use.
            If true use weighted segment algorithm, else use overlap segment.

    Returns:
        float:
            F1 score between the ground truth and detected anomalies.
    """
    def _cm(x, y, z, w, f):
        return contextual_confusion_matrix(x, y, z, w, f, weighted)
    return _f1_score(expected, observed, data, start, end, _cm)
