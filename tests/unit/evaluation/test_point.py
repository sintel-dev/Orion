import numpy as np
import pandas as pd
import pytest

from orion.evaluation.point import (
    _point_partition, point_accuracy, point_confusion_matrix, point_f1_score, point_precision,
    point_recall)


@pytest.fixture()
def expected():
    return pd.DataFrame({'timestamp': [3, 4, 5]})


@pytest.fixture()
def observed():
    return pd.DataFrame({'timestamp': [4, 6, 7, 8, 12]})


def test__point_partiton(expected, observed):
    expected = list(expected['timestamp'])
    observed = list(observed['timestamp'])
    expected_parts = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    observed_parts = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1]

    expected_return, observed_return, _ = _point_partition(expected, observed)
    np.testing.assert_array_equal(np.array(expected_return),
                                  np.array(expected_parts))

    np.testing.assert_array_equal(np.array(observed_return),
                                  np.array(observed_parts))


def test_point_confusion_matrix(expected, observed):
    expected_return = (3, 4, 2, 1)
    returned = point_confusion_matrix(expected, observed)
    np.testing.assert_array_equal(np.array(returned),
                                  np.array(expected_return))


def test_point_accuracy(expected, observed):
    expected_return = float(4 / 10)
    returned = point_accuracy(expected, observed)
    assert returned == expected_return


def test_point_precision(expected, observed):
    expected_return = float(1 / 5)
    returned = point_precision(expected, observed)
    assert returned == expected_return


def test_point_recall(expected, observed):
    expected_return = float(1 / 3)
    returned = point_recall(expected, observed)
    assert returned == expected_return


def test_point_f1_score(expected, observed):
    expected_return = float(1 / 4)
    returned = point_f1_score(expected, observed)
    assert returned == expected_return


def test_point_f1_score_nan():
    expected = pd.DataFrame({"timestamp": [2, 3]})
    observed = pd.DataFrame({"timestamp": [4, 5]})
    returned = point_f1_score(expected, observed)
    assert np.isnan(returned)


def test_point_confusion_matrix_empty():
    empty = pd.DataFrame({"timestamp": []})

    returned = point_confusion_matrix(empty, empty)

    # There is no range to partition, so the number of true negatives is
    # unknown, but there are no positives and no false negatives.
    assert returned == (None, 0, 0, 0)


def test_point_confusion_matrix_empty_with_range():
    empty = pd.DataFrame({"timestamp": []})

    returned = point_confusion_matrix(empty, empty, start=0, end=9)

    # Every point in the range is correctly considered normal.
    np.testing.assert_array_equal(np.array(returned), np.array((10, 0, 0, 0)))


def test_point_scores_empty():
    empty = pd.DataFrame({"timestamp": []})

    assert np.isnan(point_precision(empty, empty))
    assert np.isnan(point_recall(empty, empty))
    assert np.isnan(point_f1_score(empty, empty))

    # Accuracy needs the true negatives, which are unknown without a range.
    with pytest.raises(ValueError):
        point_accuracy(empty, empty)

    assert point_accuracy(empty, empty, start=0, end=9) == 1.0
