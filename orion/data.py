# -*- coding: utf-8 -*-

"""
Data Management module.

This module contains functions that allow downloading demo data from Amazon S3,
as well as load and work with other data stored locally.

The demo data is a modified version of the NASA data found here:

https://s3-us-west-2.amazonaws.com/telemanom/data.zip
"""

import logging
import os

import pandas as pd

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)
NASA_DATA_URL = 'https://d3-ai-orion.s3.amazonaws.com/{}.csv'


def load_nasa_signal(signal_name, test_size=None):
    """Load the nasa signal with the given name.

    If the signal has never been loaded before, it will be downloaded
    from the [d3-ai-orion bucket](https://d3-ai-orion.s3.amazonaws.com)
    and then cached inside the `data` folder, within the `orion` package
    directory, and then returned.

    Otherwise, if it has been downloaded and cached before, it will be directly
    loaded from the `orion/data` folder without contacting S3.

    If a `test_size` value is given, the data will be split in two parts
    without altering its order, making the second one proportionally as
    big as the given value.

    Args:
        signal_name (str): Name of the signal name to load.
        test_size (float): Value between 0 and 1 indicating the proportional
            size of the test split. If 0 or None (default), the data is not split.

    Returns:
        If no test_size is given, a single pandas.DataFrame is returned containing all
        the data. If test_size is given, a tuple containing one pandas.DataFrame for
        the train split and another one for the test split is returned.
    """

    filename = os.path.join(DATA_PATH, signal_name + '.csv')
    if os.path.exists(filename):
        data = pd.read_csv(filename)

    else:
        url = NASA_DATA_URL.format(signal_name)

        LOGGER.debug('Downloading signal %s from %s', signal_name, url)
        os.makedirs(DATA_PATH, exist_ok=True)
        data = pd.read_csv(url)
        data.to_csv(filename, index=False)

    return data


def load_csv(path, timestamp_column=None, value_column=None):
    header = timestamp_column is not None
    data = pd.read_csv(path, header=header)

    if timestamp_column is None:
        if value_column is not None:
            raise ValueError("If value_column is provided, timestamp_column must be as well")

        return data

    elif value_column is None:
        raise ValueError("If timestamp_column is provided, value_column must be as well")
    elif timestamp_column == value_column:
        raise ValueError("timestamp_column cannot be the same as value_column")

    timestamp_column_name = data.columns[timestamp_column]
    value_column_name = data.columns[value_column]
    columns = {
        'timestamp': data[timestamp_column_name].values,
        'value': data[value_column_name].values,
    }

    return pd.DataFrame(columns)[['timestamp', 'value']]


def load_signal(signal, test_size=None, timestamp_column=None, value_column=None):
    if os.path.isfile(signal):
        data = load_csv(signal, timestamp_column, value_column)
    else:
        data = load_nasa_signal(signal)

    if test_size is None:
        return data

    test_length = round(len(data) * test_size)
    train = data.iloc[:-test_length]
    test = data.iloc[-test_length:]

    return train, test
