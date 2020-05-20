import pandas as pd
import pytest

from orion.eval import Evaluator


@pytest.fixture()
def eval():
    return Evaluator()


def test__score_segments(eval):
    expected = pd.DataFrame({'start': [2], 'end': [5]})
    observed = pd.DataFrame({'start': [6], 'end': [8]})

    expected_return = float(0)
    returned = eval.accuracy_score(expected, observed)
    assert returned == expected_return


def test__score_points(eval):
    expected = observed = pd.DataFrame({'start': [2], 'end': [2]})

    expected_return = float(1)
    returned = eval.f1_score(expected, observed)
    assert returned == expected_return


def test__score_segments_segments(eval):
    expected = pd.DataFrame({'start': [2], 'end': [5]})
    observed = pd.DataFrame({'start': [6], 'end': [8]})

    expected_return = float(0)
    returned = eval.recall_score(expected, observed, eval._overlap_segment)
    assert returned == expected_return


def test_score_overlap_points(eval):
    expected = observed = pd.DataFrame({'start': [2], 'end': [2]})

    expected_return = float(1)
    returned = eval.f1_score(expected, observed, eval._overlap_segment)
    assert returned == expected_return
