import numpy as np
from sklearn import metrics


class Evaluator:
    """Evaluator Class.

    The Evaluator Class provides a mechanic to assess the perfomance of
    anomaly detection methods. This class addresses the problem as
    contextual anomalies.

    anomaly = [(start, end)] inclusive interval.

    """

    def _overlap(self, expected, observed):
        first = expected[0] - observed[1]
        second = expected[1] - observed[0]
        return first * second < 0

    def _any_overlap(self, part, intervals):
        for interval in intervals:
            if self._overlap(part, interval):
                return 1

        return 0

    def _pad(self, lst):
        return [(part[0], part[1] + 1) for part in lst]

    def _partition(self, expected, observed, start=None, end=None):
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
            expected_parts.append(self._any_overlap(part, expected))
            observed_parts.append(self._any_overlap(part, observed))

        return expected_parts, observed_parts, weights

    def _weighted_segment(self, expected, observed, start=None, end=None):
        expected, observed, weights = self._partition(expected, observed, start, end)

        return metrics.confusion_matrix(
            expected, observed, sample_weight=weights, labels=[0, 1]).ravel()

    def _overlap_segment(self, expected, observed, start=None, end=None):
        tp, fp, fn = 0, 0, 0

        observed_copy = observed.copy()

        for expected_seq in expected:
            found = False
            for observed_seq in observed:
                if self._overlap(expected_seq, observed_seq):
                    if not found:
                        tp += 1
                        found = True
                    if observed_seq in observed_copy:
                        observed_copy.remove(observed_seq)
            if not found:
                fn += 1
        fp += len(observed_copy)

        return None, fp, fn, tp

    def get_confusion_matrix(self, expected, observed, method=None,
                             data=None, start=None, end=None):
        """Compute the confusion matrix between the ground truth and the detected anomalies.

        Args:
            * expected (pd.DataFrame): Ground truth.
            * observed (pd.DataFrame): Detected anomalies.
            * method (function): Approach for computing number of tp, fp, fn, tn.
            * data (pd.DataFrame): Original data. Used to extract start and end.
            * start (int): Minimum timestamp of the original data.
            * end (int): Maximum timestamp of the original data.
        """

        if data is not None:
            start = data['timestamp'].min()
            end = data['timestamp'].max()

        if not isinstance(expected, list):
            expected = list(expected[['start', 'end']].itertuples(index=False))
        if not isinstance(observed, list):
            observed = list(observed[['start', 'end']].itertuples(index=False))

        expected = self._pad(expected)
        observed = self._pad(observed)

        function = self._weighted_segment
        if method is not None:
            function = method

        return function(expected, observed, start, end)

    def accuracy_score(self, expected, observed, method=None, data=None, start=None, end=None):
        """Compute an accuracy score between the ground truth and the detected anomalies.
        Args:
            * expected (pd.DataFrame): Ground truth
            * observed (pd.DataFrame): Detected anomalies
            * data (pd.DataFrame): Original data. Used to extract start and end.
            * start (int): Minimum timestamp of the original data.
            * end (int): Maximum timestamp of the original data.
        """
        tn, fp, fn, tp = self.get_confusion_matrix(expected, observed, method, data, start, end)
        if tn is None:
            raise ValueError("Cannot obtain accuracy score for overlap segment method.")
        return (tp + tn) / (tn + fp + fn + tp)

    def recall_score(self, expected, observed, method=None, data=None, start=None, end=None):
        """Compute a recall score between the ground truth and the detected anomalies.
        Args:
            * expected (pd.DataFrame): Ground truth
            * observed (pd.DataFrame): Detected anomalies
            * data (pd.DataFrame): Original data. Used to extract start and end.
            * start (int): Minimum timestamp of the original data.
            * end (int): Maximum timestamp of the original data.
        """
        tn, fp, fn, tp = self.get_confusion_matrix(expected, observed, method, data, start, end)
        return tp / (tp + fn)

    def precision_score(self, expected, observed, method=None, data=None, start=None, end=None):
        """Compute an precision score between the ground truth and the detected anomalies.
        Args:
            * expected (pd.DataFrame): Ground truth
            * observed (pd.DataFrame): Detected anomalies
            * data (pd.DataFrame): Original data. Used to extract start and end.
            * start (int): Minimum timestamp of the original data.
            * end (int): Maximum timestamp of the original data.
        """
        tn, fp, fn, tp = self.get_confusion_matrix(expected, observed, method, data, start, end)
        return tp / (tp + fp)

    def f1_score(self, expected, observed, method=None, data=None, start=None, end=None):
        """Compute an f1 score between the ground truth and the detected anomalies.
        Args:
            * expected (pd.DataFrame): Ground truth
            * observed (pd.DataFrame): Detected anomalies
            * data (pd.DataFrame): Original data. Used to extract start and end.
            * start (int): Minimum timestamp of the original data.
            * end (int): Maximum timestamp of the original data.
        """
        precision = self.precision_score(expected, observed, method, data, start, end)
        recall = self.recall_score(expected, observed, method, data, start, end)

        try:
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            print('Invalid value for precision {}/ recall {}.'.format(precision, recall))
            return np.nan


class PointEvaluator(Evaluator):
    """PointEvaluator Class.

    The PointEvaluator Class is an extension of Evaluator in which it
    provides a mechanic to assess the perfomance of anomaly detection
    methods. This class addresses the problem as point anomalies.

    anomaly = [timestamp]

    """

    def _partition(self, expected, observed, start=None, end=None):
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

    def get_confusion_matrix(self, expected, observed, method=None,
                             data=None, start=None, end=None):
        """Compute the confusion matrix between the ground truth and the detected anomalies.

        Args:
            * expected (pd.DataFrame): Ground truth.
            * observed (pd.DataFrame): Detected anomalies.
            * method (function): Approach for computing number of tp, fp, fn, tn.
            * data (pd.DataFrame): Original data. Used to extract start and end.
            * start (int): Minimum timestamp of the original data.
            * end (int): Maximum timestamp of the original data.
        """

        if data is not None:
            start = data['timestamp'].min()
            end = data['timestamp'].max()

        if not isinstance(expected, list):
            expected = list(expected['timestamp'])
        if not isinstance(observed, list):
            observed = list(observed['timestamp'])

        return self._weighted_segment(expected, observed, start, end)


METRICS = {
    'accuracy': Evaluator().accuracy_score,
    'f1': Evaluator().f1_score,
    'recall': Evaluator().recall_score,
    'precision': Evaluator().precision_score,
}
