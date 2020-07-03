import pandas as pd
import pytest

from orion.evaluation.utils import (
    from_list_points_labels, from_list_points_timestamps, from_pandas_contextual,
    from_pandas_points, from_pandas_points_labels)


def assert_list_tuples(returned, expected_return):
    assert len(returned) == len(expected_return)
    for ret, exp_ret in zip(returned, expected_return):
        assert tuple(ret) == exp_ret


def test_from_pandas_contextual():
    anomalies = pd.DataFrame({'start': [2, 8], 'end': [5, 9]})

    expected_return = [(2, 5), (8, 9)]
    returned = from_pandas_contextual(anomalies)
    assert_list_tuples(returned, expected_return)


def test_from_pandas_contextual_severity():
    anomalies = pd.DataFrame({'start': [2, 8], 'end': [5, 9],
                              'severity': [0.1, 0.2]})

    expected_return = [(2, 5, 0.1), (8, 9, 0.2)]
    returned = from_pandas_contextual(anomalies)
    assert_list_tuples(returned, expected_return)


def test_from_pandas_contextual_error():
    anomalies = pd.DataFrame({'start': [2, 8]})
    with pytest.raises(KeyError):
        from_pandas_contextual(anomalies)


def test_from_list_points_timestamps():
    anomalies = [2, 3, 4, 5, 8, 9]

    expected_return = [(2, 5), (8, 9)]
    returned = from_list_points_timestamps(anomalies)
    assert_list_tuples(returned, expected_return)


def test_from_pandas_points():
    anomalies = pd.DataFrame({'timestamp': [2, 3, 4, 5, 8, 9]})

    expected_return = [(2, 5), (8, 9)]
    returned = from_pandas_points(anomalies)
    assert_list_tuples(returned, expected_return)


def test_from_pandas_points_error():
    anomalies = pd.DataFrame({'label': [0, 1]})
    with pytest.raises(KeyError):
        from_pandas_points(anomalies)


def test_from_pandas_points_labels():
    anomalies = pd.DataFrame({'timestamp': [2, 3, 4, 5, 6, 7, 8, 9],
                              'label': [1, 1, 1, 1, 0, 0, 1, 1]})

    expected_return = [(2, 5), (8, 9)]
    returned = from_pandas_points_labels(anomalies)
    assert_list_tuples(returned, expected_return)


def test_from_pandas_points_labels_error():
    anomalies = pd.DataFrame({'timestamp': [2, 8]})
    with pytest.raises(KeyError):
        from_pandas_points_labels(anomalies)


def test_from_list_points_labels():
    anomalies = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]

    expected_return = [(2, 5), (8, 9)]
    returned = from_list_points_labels(anomalies)
    assert_list_tuples(returned, expected_return)
