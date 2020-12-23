import numpy as np
from numpy.testing import assert_array_equal

from orion.primitives.timeseries_preprocessing import fillna


def single_dimensional():
    return np.array([0, 3, np.nan, 4, 3, 2, 2, np.nan, np.nan])


def multidimensional():
    return np.array([[0, 3, np.nan],
                     [np.nan, 2, 4],
                     [5, np.nan, 6],
                     [9, np.nan, 3],
                     [4, 7, np.nan]])


def test_fillna_ffill():
    X = single_dimensional()
    expected_return = np.array([0, 3, 3, 4, 3, 2, 2, 2, 2]).reshape(-1, 1)

    returned = fillna(X, method='ffill')

    assert_array_equal(returned, expected_return)


def test_fillna_bfill():
    X = single_dimensional()
    expected_return = np.array([0, 3, 4, 4, 3, 2, 2, np.nan, np.nan]).reshape(-1, 1)

    returned = fillna(X, method='bfill')

    assert_array_equal(returned, expected_return)


def test_fillna_ffill_bfill():
    X = single_dimensional()
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
