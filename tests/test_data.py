#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the `data` module."""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd

import orion
from orion.data import download, load_signal

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(orion.__file__)),
    'data'
)


# ######## #
# download #
# ######## #
@patch('orion.data.pd.read_csv')
@patch('orion.data.os.path.exists')
def test_download_cached(exists_mock, read_csv_mock):
    # setup
    exists_mock.return_value = True

    # run
    returned = download('a_signal_name')

    # assert
    assert returned == read_csv_mock.return_value

    expected_filename = os.path.join(DATA_PATH, 'a_signal_name.csv')
    read_csv_mock.assert_called_once_with(expected_filename)


@patch('orion.data.pd.read_csv')
@patch('orion.data.os.path.exists')
def test_download_new(exists_mock, read_csv_mock):
    # setup
    exists_mock.return_value = False

    # run
    returned = download('a_signal_name')

    # assert
    assert returned == read_csv_mock.return_value

    expected_url = 'https://sintel-orion.s3.amazonaws.com/a_signal_name.csv'
    read_csv_mock.assert_called_once_with(expected_url)

    expected_filename = os.path.join(DATA_PATH, 'a_signal_name.csv')
    returned.to_csv.assert_called_once_with(expected_filename, index=False)


# ########### #
# load_signal #
# ########### #
@patch('orion.data.load_csv')
@patch('orion.data.os.path.isfile')
def test_load_signal_filename(isfile_mock, load_csv_mock):
    # setup
    data = pd.DataFrame({
        'timestamp': list(range(10)),
        'value': list(np.arange(10, 20, dtype=float))
    })
    load_csv_mock.return_value = data
    isfile_mock.return_value = True

    # run
    returned = load_signal('a/path/to/a.csv')

    # assert
    pd.testing.assert_frame_equal(returned, load_csv_mock.return_value)

    load_csv_mock.assert_called_once_with('a/path/to/a.csv', None, None)


@patch('orion.data.download')
@patch('orion.data.load_csv')
@patch('orion.data.os.path.isfile')
def test_load_signal_nasa_signal_name(isfile_mock, load_csv_mock, lns_mock):
    # setup
    data = pd.DataFrame({
        'timestamp': list(range(10)),
        'value': list(np.arange(10, 20, dtype=float))
    })
    lns_mock.return_value = data
    isfile_mock.return_value = False

    # run
    returned = load_signal('S-1')

    # assert
    pd.testing.assert_frame_equal(returned, data)

    load_csv_mock.assert_not_called()
    lns_mock.assert_called_once_with('S-1')


@patch('orion.data.download')
@patch('orion.data.load_csv')
@patch('orion.data.os.path.isfile')
def test_load_signal_nasa_signal_name_multivariate(isfile_mock, load_csv_mock, lns_mock):
    # setup
    data = pd.DataFrame({
        'timestamp': list(range(10)),
        '0': list(np.arange(10, 20, dtype=float)),
        '1': list(np.arange(20, 30, dtype=float)),
        '2': list(np.arange(30, 40, dtype=float))
    })
    lns_mock.return_value = data
    isfile_mock.return_value = False

    # run
    returned = load_signal('multivariate/S-1')

    # assert
    pd.testing.assert_frame_equal(returned, data)

    load_csv_mock.assert_not_called()
    lns_mock.assert_called_once_with('multivariate/S-1')


@patch('orion.data.load_csv')
@patch('orion.data.os.path.isfile')
def test_load_signal_test_size(isfile_mock, load_csv_mock):
    # setup
    isfile_mock.return_value = True

    data = pd.DataFrame({
        'timestamp': list(range(10)),
        'value': list(range(10, 20))
    })
    load_csv_mock.return_value = data

    # run
    returned = load_signal('a/path/to/a.csv', test_size=0.33)

    # assert
    assert isinstance(returned, tuple)
    assert len(returned) == 2

    train, test = returned

    expected_train = pd.DataFrame({
        'timestamp': list(range(7)),
        'value': list(np.arange(10, 17).astype(float))
    })

    pd.testing.assert_frame_equal(train, expected_train)

    expected_test = pd.DataFrame({
        'timestamp': list(range(7, 10)),
        'value': list(np.arange(17, 20).astype(float))
    })
    expected_test.index = range(7, 10)
    pd.testing.assert_frame_equal(test, expected_test)
