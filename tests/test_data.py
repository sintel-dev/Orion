#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the `data` module."""

import os
from unittest.mock import patch

import pandas as pd

import orion
from orion.data import load_nasa_signal

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(orion.__file__)),
    'data'
)


@patch('orion.data.pd.read_csv')
@patch('orion.data.os.path.exists')
def test_load_nasa_signal_cached(exists_mock, read_csv_mock):
    # setup
    exists_mock.return_value = True

    # run
    returned = load_nasa_signal('a_signal_name')

    # assert
    assert returned == read_csv_mock.return_value

    expected_filename = os.path.join(DATA_PATH, 'a_signal_name.csv')
    exists_mock.assert_called_once_with(expected_filename)

    read_csv_mock.assert_called_once_with(expected_filename)


@patch('orion.data.pd.read_csv')
@patch('orion.data.os.path.exists')
def test_load_nasa_signal_new(exists_mock, read_csv_mock):
    # setup
    exists_mock.return_value = False

    # run
    returned = load_nasa_signal('a_signal_name')

    # assert
    assert returned == read_csv_mock.return_value

    expected_filename = os.path.join(DATA_PATH, 'a_signal_name.csv')
    exists_mock.assert_called_once_with(expected_filename)

    expected_url = 'https://d3-ai-orion.s3.amazonaws.com/a_signal_name.csv'
    read_csv_mock.assert_called_once_with(expected_url)

    returned.to_csv.assert_called_once_with(expected_filename, index=False)


@patch('orion.data.pd.read_csv')
@patch('orion.data.os.path.exists')
def test_load_nasa_signal_test_size(exists_mock, read_csv_mock):
    # setup
    exists_mock.return_value = True

    data = pd.DataFrame({
        'timestamp': list(range(10)),
        'value': list(range(10, 20))
    })
    read_csv_mock.return_value = data

    # run
    returned = load_nasa_signal('a_signal_name', test_size=0.33)

    # assert
    assert isinstance(returned, tuple)
    assert len(returned) == 2

    train, test = returned

    expected_train = pd.DataFrame({
        'timestamp': list(range(7)),
        'value': list(range(10, 17))
    })
    pd.testing.assert_frame_equal(train, expected_train)

    expected_test = pd.DataFrame({
        'timestamp': list(range(7, 10)),
        'value': list(range(17, 20))
    })
    expected_test.index = range(7, 10)
    pd.testing.assert_frame_equal(test, expected_test)
