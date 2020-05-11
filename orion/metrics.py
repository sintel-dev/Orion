from sklearn import metrics


def _overlap(expected, observed):
    first = expected[0] - observed[1]
    second = expected[1] - observed[0]
    return first * second < 0


def _any_overlap(part, intervals):
    for interval in intervals:
        if _overlap(part, interval):
            return 1

    return 0


def _partition(expected, observed, start=None, end=None):
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


def _score(scorer, expected, observed, data=None, start=None, end=None):
    """Compute a score between the ground truth and the detected anomalies.

    Args:
        * scorer (callable): scikit-learn style scorer function.
        * expected (pd.DataFrame): Ground truth.
        * observed (pd.DataFramne): Detected anomalies.
        * data (pd.DataFramne): Original data. Used to extract start and end.
        * start (int): Minimum timestamp of the original data.
        * end (int): Maximum timestamp of the original data.
    """
    if data is not None:
        start = data['timestamp'].min()
        end = data['timestamp'].max()

    expected = list(expected[['start', 'end']].itertuples(index=False))
    observed = list(observed[['start', 'end']].itertuples(index=False))

    expected, observed, weights = _partition(expected, observed, start, end)

    return scorer(expected, observed, sample_weight=weights)


def accuracy_score(expected, observed, data=None, start=None, end=None):
    """Compute an accuracy score between the ground truth and the detected anomalies.

    Args:
        * expected (pd.DataFrame): Ground truth
        * observed (pd.DataFramne): Detected anomalies
        * data (pd.DataFramne): Original data. Used to extract start and end.
        * start (int): Minimum timestamp of the original data.
        * end (int): Maximum timestamp of the original data.
    """
    return _score(metrics.accuracy_score, expected, observed, data, start, end)


def f1_score(expected, observed, data=None, start=None, end=None):
    """Compute an f1 score between the ground truth and the detected anomalies.

    Args:
        * expected (pd.DataFrame): Ground truth
        * observed (pd.DataFramne): Detected anomalies
        * data (pd.DataFramne): Original data. Used to extract start and end.
        * start (int): Minimum timestamp of the original data.
        * end (int): Maximum timestamp of the original data.
    """
    return _score(metrics.f1_score, expected, observed, data, start, end)


def recall_score(expected, observed, data=None, start=None, end=None):
    """Compute a recall score between the ground truth and the detected anomalies.

    Args:
        * expected (pd.DataFrame): Ground truth
        * observed (pd.DataFramne): Detected anomalies
        * data (pd.DataFramne): Original data. Used to extract start and end.
        * start (int): Minimum timestamp of the original data.
        * end (int): Maximum timestamp of the original data.
    """
    return _score(metrics.recall_score, expected, observed, data, start, end)


def precision_score(expected, observed, data=None, start=None, end=None):
    """Compute an precision score between the ground truth and the detected anomalies.

    Args:
        * expected (pd.DataFrame): Ground truth
        * observed (pd.DataFramne): Detected anomalies
        * data (pd.DataFramne): Original data. Used to extract start and end.
        * start (int): Minimum timestamp of the original data.
        * end (int): Maximum timestamp of the original data.
    """
    return _score(metrics.precision_score, expected, observed, data, start, end)


METRICS = {
    'accuracy': accuracy_score,
    'f1': f1_score,
    'recall': recall_score,
    'precision': precision_score,
}
