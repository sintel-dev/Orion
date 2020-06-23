import numpy as np
import pandas as pd
import pytest

from orion.evaluation.contextual import (
    _contextual_partition, _overlap_segment, contextual_accuracy, contextual_confusion_matrix,
    contextual_f1_score, contextual_precision, contextual_recall)


@pytest.fixture()
def expected():
    return pd.DataFrame({'start': [1, 5, 7, 12, 16, 22], 'end': [2, 6, 10, 13, 17, 23]})


@pytest.fixture()
def observed():
    return pd.DataFrame({'start': [2, 4, 8, 11, 19, 22], 'end': [3, 5, 9, 14, 20, 23]})


@pytest.fixture()
def expected_point():
    return pd.DataFrame({'start': [2, 5, 8, 12, 15, 18], 'end': [2, 5, 8, 12, 17, 18]})


@pytest.fixture()
def observed_point():
    return pd.DataFrame({'start': [2, 3, 8, 11, 16, 20], 'end': [2, 5, 10, 13, 16, 20]})


def test__contextual_partition(expected, observed):
    expected = list(expected[['start', 'end']].itertuples(index=False))
    observed = list(observed[['start', 'end']].itertuples(index=False))
    expected_parts = [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
    observed_parts = [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    weights = [1] * len(observed_parts)
    weights[13] = weights[15] = weights[17] = 2
    length = expected[-1][-1] - expected[0][0]

    expected_return, observed_return, weights_return = _contextual_partition(expected, observed)

    np.testing.assert_array_equal(np.array(expected_return),
                                  np.array(expected_parts))

    np.testing.assert_array_equal(np.array(observed_return),
                                  np.array(observed_parts))

    np.testing.assert_array_equal(np.array(weights_return),
                                  np.array(weights))

    assert len(expected_return) == len(observed_return) == len(weights_return)
    assert sum(weights_return) == length


def test__overlap_segment(expected, observed):
    expected = list(expected[['start', 'end']].itertuples(index=False))
    observed = list(observed[['start', 'end']].itertuples(index=False))
    expected_return = (None, 3, 3, 3)
    returned = _overlap_segment(expected, observed)
    np.testing.assert_array_equal(np.array(returned),
                                  np.array(expected_return))


def test__overlap_segment_points(expected_point, observed_point):
    expected = list(expected_point[['start', 'end']].itertuples(index=False))
    observed = list(observed_point[['start', 'end']].itertuples(index=False))
    expected_return = (None, 4, 4, 2)
    returned = _overlap_segment(expected, observed)
    np.testing.assert_array_equal(np.array(returned),
                                  np.array(expected_return))


def test_contextual_confusion_matrix(expected, observed):
    expected_return = (3, 6, 6, 8)
    returned = contextual_confusion_matrix(expected, observed)
    np.testing.assert_array_equal(np.array(returned),
                                  np.array(expected_return))


def test_contextual_confusion_matrix_points(expected_point, observed_point):
    expected_return = (4, 7, 3, 5)
    returned = contextual_confusion_matrix(expected_point, observed_point)
    np.testing.assert_array_equal(np.array(returned),
                                  np.array(expected_return))


def test_contextual_confusion_matrix_overlap(expected, observed):
    expected_return = (None, 1, 1, 5)
    returned = contextual_confusion_matrix(expected, observed, weighted=False)
    np.testing.assert_array_equal(np.array(returned),
                                  np.array(expected_return))


def test_contextual_confusion_matrix_overlap_points(expected_point, observed_point):
    expected_return = (None, 1, 1, 5)
    returned = contextual_confusion_matrix(expected_point, observed_point, weighted=False)
    np.testing.assert_array_equal(np.array(returned),
                                  np.array(expected_return))


def test_contextual_accuracy(expected, observed):
    expected_return = float(11 / 23)
    returned = contextual_accuracy(expected, observed)
    assert returned == expected_return


def test_contextual_accuracy_overlap(expected, observed):
    with pytest.raises(ValueError):
        contextual_accuracy(expected, observed, weighted=False)


def test_contextual_precision(expected, observed):
    expected_return = float(8 / 14)
    returned = contextual_precision(expected, observed)
    assert returned == expected_return


def test_contextual_recall(expected, observed):
    expected_return = float(8 / 14)
    returned = contextual_recall(expected, observed)
    assert returned == expected_return


def test_contextual_f1_score(expected, observed):
    expected_return = float(4 / 7)
    returned = contextual_f1_score(expected, observed)
    assert returned == expected_return


def test_contextual_f1_score_nan():
    expected = pd.DataFrame({"start": [2], "end": [5]})
    observed = pd.DataFrame({"start": [6], "end": [8]})
    returned = contextual_f1_score(expected, observed)
    assert np.isnan(returned)
