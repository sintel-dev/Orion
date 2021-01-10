import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from orion.primitives.timeseries_preprocessing import fillna


def signal():
    return np.array([0, 3, np.nan, 4, 3, 2, 2, np.nan, np.nan]).reshape(-1, 1)


def multidimensional():
    return np.array([[0, 3, np.nan],
                     [np.nan, 2, 4],
                     [5, np.nan, 6],
                     [9, np.nan, 3],
                     [4, 7, np.nan]])


def series():
    X = signal()
    return pd.Series(X.flatten())


def test_fillna_series():
    df = series()
    expected_return = np.array([0, 3, 0, 4, 3, 2, 2, 0, 0])

    returned = fillna(df, value=0)

    assert_array_equal(returned, expected_return)


def test_fillna_single_dimension():
    X = signal().flatten()
    expected_return = np.array([0, 3, 0, 4, 3, 2, 2, 0, 0])

    returned = fillna(X, value=0)

    assert_array_equal(returned, expected_return)


def test_fillna_ffill():
    X = signal()
    expected_return = np.array([0, 3, 3, 4, 3, 2, 2, 2, 2]).reshape(-1, 1)

    returned = fillna(X, method='ffill')

    assert_array_equal(returned, expected_return)


def test_fillna_bfill():
    X = signal()
    expected_return = np.array([0, 3, 4, 4, 3, 2, 2, np.nan, np.nan]).reshape(-1, 1)

    returned = fillna(X, method='bfill')

    assert_array_equal(returned, expected_return)


def test_fillna_ffill_bfill():
    X = signal()
    expected_return = np.array([0, 3, 3, 4, 3, 2, 2, 2, 2]).reshape(-1, 1)

    returned = fillna(X, method=['ffill', 'bfill'])

    assert_array_equal(returned, expected_return)


def test_fillna_multi_dimensional():
    X = multidimensional()
    expected_return = np.array([[0, 3, 4],
                                [0, 2, 4],
                                [5, 2, 6],
                                [9, 2, 3],
                                [4, 7, 3]])

    returned = fillna(X, method=['ffill', 'bfill'])

    assert_array_equal(returned, expected_return)
