import numpy as np
import pandas as pd
import pytest

from orion.evaluation.common import _overlap_segment
from orion.evaluation.contextual import (
    contextual_accuracy, contextual_confusion_matrix, contextual_f1_score, contextual_precision,
    contextual_recall)


@pytest.fixture()
def expected():
    return pd.DataFrame({'start': [2], 'end': [5]})


@pytest.fixture()
def observed():
    return pd.DataFrame({'start': [6], 'end': [8]})


def test_contextual_confusion_matrix(expected, observed):
    expected_return = (0, 3, 4, 0)
    returned = contextual_confusion_matrix(expected, observed)
    np.testing.assert_array_equal(np.array(returned),
                                  np.array(expected_return))


def test_contextual_accuracy(expected, observed):
    expected_return = float(0)
    returned = contextual_accuracy(expected, observed)
    assert returned == expected_return


def test_contextual_accuracy_overlap(expected, observed):
    with pytest.raises(ValueError):
        contextual_accuracy(expected, observed, method=_overlap_segment)


def test_contextual_precision(expected, observed):
    expected_return = float(0)
    returned = contextual_precision(expected, observed)
    assert returned == expected_return


def test_contextual_recall(expected, observed):
    expected_return = float(0)
    returned = contextual_recall(expected, observed)
    assert returned == expected_return


def test_contextual_f1_score(expected, observed):
    returned = contextual_f1_score(expected, observed)
    assert np.isnan(returned)
