# -*- coding: utf-8 -*-

"""
Data Management module.

This module contains functions that allow downloading demo data from Amazon S3,
as well as load and work with other data stored locally.

The demo data is a modified version of the NASA data found here:

https://s3-us-west-2.amazonaws.com/telemanom/data.zip
"""

import json
import logging
import os

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)
BUCKET = 'd3-ai-orion'
S3_URL = 'https://{}.s3.amazonaws.com/{}.csv'


def download(name, test_size=None):
    """Load the CSV with the given name from S3.

    If the CSV has never been loaded before, it will be downloaded
    from the [d3-ai-orion bucket](https://d3-ai-orion.s3.amazonaws.com)
    and then cached inside the `data` folder, within the `orion` package
    directory, and then returned.

    Otherwise, if it has been downloaded and cached before, it will be directly
    loaded from the `orion/data` folder without contacting S3.

    If a `test_size` value is given, the data will be split in two parts
    without altering its order, making the second one proportionally as
    big as the given value.

    Args:
        name (str): Name of the CSV to load.
        test_size (float): Value between 0 and 1 indicating the proportional
            size of the test split. If 0 or None (default), the data is not split.

    Returns:
        If no test_size is given, a single pandas.DataFrame is returned containing all
        the data. If test_size is given, a tuple containing one pandas.DataFrame for
        the train split and another one for the test split is returned.
    """

    filename = os.path.join(DATA_PATH, name + '.csv')
    if os.path.exists(filename):
        data = pd.read_csv(filename)

    else:
        url = S3_URL.format(BUCKET, name)

        LOGGER.debug('Downloading CSV %s from %s', name, url)
        os.makedirs(DATA_PATH, exist_ok=True)

        try:
            data = pd.read_csv(url)
        except Exception:
            LOGGER.exception('File not found: %s', url)
            data = None
        else:
            data.to_csv(filename, index=False)

    return data


def load_csv(path, timestamp_column=None, value_column=None):
    header = None if timestamp_column is not None else 'infer'
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
        data = download(signal)

    data['timestamp'] = data['timestamp'].astype(int)
    data['value'] = data['value'].astype(float)

    if test_size is None:
        return data

    test_length = round(len(data) * test_size)
    train = data.iloc[:-test_length]
    test = data.iloc[-test_length:]

    return train, test


def load_anomalies(signal, edges=False):
    anomalies = download('anomalies')

    anomalies = anomalies.set_index('signal').loc[signal].values[0]
    anomalies = pd.DataFrame(json.loads(anomalies), columns=['start', 'end'])

    if edges:
        data = download(signal)
        start = data.timestamp.min()
        end = data.timestamp.max()

        anomalies['score'] = 1
        parts = np.concatenate([
            [[start, anomalies.start.min(), 0]],
            anomalies.values,
            [[anomalies.end.max(), end, 0]]
        ], axis=0)
        anomalies = pd.DataFrame(parts, columns=['start', 'end', 'score'])

    return anomalies
