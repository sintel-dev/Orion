import numpy as np
import pandas as pd
import pytest

from orion.evaluation.point import (
    point_accuracy, point_confusion_matrix, point_f1_score, point_precision, point_recall)


@pytest.fixture()
def expected():
    return pd.DataFrame({'timestamp': [2, 3, 4, 5]})


@pytest.fixture()
def observed():
    return pd.DataFrame({'timestamp': [6, 7, 8]})


def test_point_confusion_matrix(expected, observed):
    expected_return = (0, 3, 4, 0)
    returned = point_confusion_matrix(expected, observed)
    np.testing.assert_array_equal(np.array(returned),
                                  np.array(expected_return))


def test_point_accuracy(expected, observed):
    expected_return = float(0)
    returned = point_accuracy(expected, observed)
    assert returned == expected_return


def test_point_precision(expected, observed):
    expected_return = float(0)
    returned = point_precision(expected, observed)
    assert returned == expected_return


def test_point_recall(expected, observed):
    expected_return = float(0)
    returned = point_recall(expected, observed)
    assert returned == expected_return


def test_point_f1_score(expected, observed):
    returned = point_f1_score(expected, observed)
    assert np.isnan(returned)
