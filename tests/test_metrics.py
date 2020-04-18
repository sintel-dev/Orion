import pandas as pd

from orion import metrics


def test__score_segments():
    expected = pd.DataFrame({'start': [2], 'end': [5]})
    observed = pd.DataFrame({'start': [6], 'end': [8]})

    expected_return = float(0)
    returned = metrics.f1_score(expected, observed)
    assert returned == expected_return


def test__score_points():
    expected = observed = pd.DataFrame({'start': [2], 'end': [2]})

    expected_return = float(1)
    returned = metrics.f1_score(expected, observed)
    assert returned == expected_return


def test__score_segments_segments():
    expected = pd.DataFrame({'start': [2], 'end': [5]})
    observed = pd.DataFrame({'start': [6], 'end': [8]})

    expected_return = (0, 1, 1)
    returned = metrics.score_overlap(expected, observed)
    assert returned == expected_return


def test_score_overlap_points():
    expected = observed = pd.DataFrame({'start': [2], 'end': [2]})

    expected_return = (1, 0, 0)
    returned = metrics.score_overlap(expected, observed)
    assert returned == expected_return
