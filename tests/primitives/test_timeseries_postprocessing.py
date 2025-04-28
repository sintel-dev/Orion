import pandas as pd
import pytest

from orion.primitives.timeseries_postprocessing import (
    diff_thresh, extract_dimension, get_intervals, rolling_std_thresh, thresholding)


@pytest.fixture
def data():
    return pd.DataFrame({
        "timestamp": list(range(1, 11)),
        "value": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "test": [1, 2] * 5,
    })


@pytest.fixture
def signal():
    return pd.DataFrame({
        "timestamp": list(range(1, 11)),
        "value": [0, 0, 1, 1, 5, 5, 10, 10, 2, 1],
    })


def test_extract_dimension(data):
    # Setup
    expected = data[['timestamp', 'value']]

    # Run
    output = extract_dimension(data)

    # Assert
    pd.testing.assert_frame_equal(output, expected)


def test_extract_dimension_dim(data):
    # Setup
    expected = data[['timestamp', 'test']]
    expected.columns = ['timestamp', 'value']

    # Run
    output = extract_dimension(data, dim='test')

    # Assert
    pd.testing.assert_frame_equal(output, expected)


def test_extract_dimension_empty():
    # Setup
    data = pd.DataFrame()

    # Run
    output = extract_dimension(data)

    # Assert
    output == []


def test_extract_dimension_timestamp_index(data):
    # Setup
    expected = data[['timestamp', 'value']]
    data = data.set_index('timestamp')

    # Run
    output = extract_dimension(data)

    # Assert
    pd.testing.assert_frame_equal(output, expected)


def test_extract_dimension_no_value(data):
    # Setup
    data = data[['timestamp', 'test']]

    # Run
    output = extract_dimension(data)

    # Assert
    output == []


def test_rolling_std_thresh(signal):
    # Setup
    expected = signal.iloc[[6, 7, 8, 9]]

    # Run
    output = rolling_std_thresh(signal, 3)

    # Assert
    output == expected


def test_rolling_std_thresh_one(signal):
    # Setup
    expected = signal.iloc[[9]]

    # Run
    output = rolling_std_thresh(signal, 4)

    # Assert
    output == expected


def test_rolling_std_thresh_empty(signal):
    # Run
    output = rolling_std_thresh(signal, 10)

    # Assert
    len(output) == 0


def test_diff_thresh(signal):
    # Setup
    expected = signal.iloc[[3, 5]]

    # Run
    output = diff_thresh(signal, 1)

    # Assert
    output == expected


def test_diff_thresh_equal(signal):
    # Setup
    expected = signal.iloc[[1, 3, 5]]

    # Run
    output = diff_thresh(signal, 1, ">=")

    # Assert
    output == expected


def test_thresholding(signal):
    # Setup
    expected = signal.iloc[[6, 7]]

    # Run
    output = thresholding(signal, 5, ">")

    # Assert
    output == expected


def test_thresholding_less(signal):
    # Setup
    expected = signal.iloc[[0, 1, 2, 3, 8, 9]]

    # Run
    output = thresholding(signal, 5, "<")

    # Assert
    output == expected


def test_get_intervals(signal):
    # Setup
    indicies = signal.iloc[[6, 7]]
    expected = (7, 8)

    # Run
    output = get_intervals(indicies, False)

    # Assert
    assert isinstance(output, list)
    assert len(output[0]) == 2

    assert output[0][0] == expected[0]
    assert output[0][1] == expected[1]


def test_get_intervals_severity(signal):
    # Setup
    indicies = signal.iloc[[6, 7]]
    expected = (7, 8, 10)

    # Run
    output = get_intervals(indicies)

    # Assert
    assert isinstance(output, list)
    assert len(output[0]) == 3

    assert output[0][0] == expected[0]
    assert output[0][1] == expected[1]
    assert output[0][2] == expected[2]
