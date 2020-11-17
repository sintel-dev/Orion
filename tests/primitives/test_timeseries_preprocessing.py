import numpy as np
from numpy.testing import assert_array_equal

from orion.primitives.timeseries_preprocessing import bfill, ffill, fillna


def single_dimensional():
    return np.array([0, 3, np.nan, 4, 3, 2, 2, np.nan, np.nan])


def multidimensional():
    return np.array([[0, 3, 5],
                     [np.nan, 2, 4],
                     [5, np.nan, 6],
                     [9, np.nan, 3],
                     [4, 7, np.nan]])


def test_ffill():
    X = single_dimensional()
    expected_return = np.array([[0, 3, 3, 4, 3, 2, 2, 2, 2]])

    returned = ffill(X)

    assert_array_equal(returned, expected_return)


def test_ffill_multidimensional():
    X = multidimensional()
    expected_return = np.array([[0, 3, 5],
                                [np.nan, 2, 4],
                                [5, 5, 6],
                                [9, 9, 3],
                                [4, 7, 7]])

    returned = ffill(X)

    assert_array_equal(returned, expected_return)


def test_bfill():
    X = single_dimensional()
    expected_return = np.array([[0, 3, 4, 4, 3, 2, 2, np.nan, np.nan]])

    returned = bfill(X)

    assert_array_equal(returned, expected_return)


def test_bfill_multidimensional():
    X = multidimensional()
    expected_return = np.array([[0, 3, 5],
                                [2, 2, 4],
                                [5, 6, 6],
                                [9, 3, 3],
                                [4, 7, np.nan]])

    returned = bfill(X)

    assert_array_equal(returned, expected_return)


def test_fillna_bfill():
    X = single_dimensional()
    expected_return = np.array([0, 3, 4, 4, 3, 2, 2, np.nan, np.nan])

    returned = fillna(X, method='bfill')

    assert_array_equal(returned, expected_return)


def test_fillna_ffill_bfill():
    X = single_dimensional()
    expected_return = np.array([0, 3, 3, 4, 3, 2, 2, 2, 2])

    returned = fillna(X, method=['ffill', 'bfill'])

    assert_array_equal(returned, expected_return)
