import numpy as np
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


def _weighted_segment(expected, observed, _partition, start=None, end=None):
    expected, observed, weights = _partition(expected, observed, start, end)

    return metrics.confusion_matrix(
        expected, observed, sample_weight=weights, labels=[0, 1]).ravel()


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


def _accuracy(expected, observed, data, start, end, method, get_confusion_matrix):
    tn, fp, fn, tp = get_confusion_matrix(expected,
                                          observed,
                                          data,
                                          start,
                                          end,
                                          method)
    if tn is None:
        raise ValueError("Cannot obtain accuracy score for overlap segment method.")
    return (tp + tn) / (tn + fp + fn + tp)


def _precision(expected, observed, data, start, end, method, get_confusion_matrix):
    tn, fp, fn, tp = get_confusion_matrix(expected,
                                          observed,
                                          data,
                                          start,
                                          end,
                                          method)
    return tp / (tp + fp)


def _recall(expected, observed, data, start, end, method, get_confusion_matrix):
    tn, fp, fn, tp = get_confusion_matrix(expected,
                                          observed,
                                          data,
                                          start,
                                          end,
                                          method)
    return tp / (tp + fn)


def _f1_score(expected, observed, data, start, end, method, get_confusion_matrix):
    precision = _precision(expected, observed, data, start, end, method, get_confusion_matrix)
    recall = _recall(expected, observed, data, start, end, method, get_confusion_matrix)

    try:
        return 2 * (precision * recall) / (precision + recall)

    except ZeroDivisionError:
        print('Invalid value encountered for precision {}/ recall {}.'.format(precision, recall))
        return np.nan
