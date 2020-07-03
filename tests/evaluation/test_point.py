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
