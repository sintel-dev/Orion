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
BUCKET = 'sintel-orion'
S3_URL = 'https://{}.s3.amazonaws.com/{}'

NASA_SIGNALS = (
    'P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7',
    'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-3',
    'D-2', 'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2', 'D-5',
    'D-6', 'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9',
    'F-2', 'G-4', 'T-3', 'D-11', 'D-12', 'B-1', 'G-6', 'G-7', 'P-7',
    'R-1', 'A-5', 'A-6', 'A-7', 'D-13', 'A-8', 'A-9', 'F-3', 'M-6',
    'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4',
    'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14',
    'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8'
)


def download(name, test_size=None, data_path=DATA_PATH):
    """Load the CSV with the given name from S3.

    If the CSV has never been loaded before, it will be downloaded
    from the [d3-ai-orion bucket](https://d3-ai-orion.s3.amazonaws.com) or
    the S3 bucket specified following the `s3://{bucket}/path/to/the.csv` format,
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

    url = None
    if name.startswith('s3://'):
        parts = name[5:].split('/', 1)
        bucket = parts[0]
        path = parts[1]
        url = S3_URL.format(bucket, path)

        filename = os.path.join(data_path, path.split('/')[-1])
    else:
        filename = os.path.join(data_path, name + '.csv')
        data_path = os.path.join(data_path, os.path.dirname(name))

    if os.path.exists(filename):
        data = pd.read_csv(filename)
    else:
        url = url or S3_URL.format(BUCKET, '{}.csv'.format(name))

        LOGGER.info('Downloading CSV %s from %s', name, url)
        os.makedirs(data_path, exist_ok=True)
        data = pd.read_csv(url)
        data.to_csv(filename, index=False)

    return data


def download_demo(path='orion-data', split=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    LOGGER.info('Downloading Orion Demo Data to folder %s', path)
    for signal in NASA_SIGNALS[0:3]:
        if split:
            download(signal + '-train', data_path=path)
            download(signal + '-test', data_path=path)
        else:
            download(signal, data_path=path)


def format_csv(df, timestamp_column=None, value_columns=None):
    timestamp_column_name = df.columns[timestamp_column] if timestamp_column else df.columns[0]
    value_column_names = df.columns[value_columns] if value_columns else df.columns[1:]

    data = dict()
    data['timestamp'] = df[timestamp_column_name].astype('int64').values
    for column in value_column_names:
        data[column] = df[column].astype(float).values

    return pd.DataFrame(data)


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

    return format_csv(data, timestamp_column, value_column)


def load_signal(signal, test_size=None, timestamp_column=None, value_column=None):
    if os.path.isfile(signal):
        data = load_csv(signal, timestamp_column, value_column)
    else:
        data = download(signal)

    data = format_csv(data)

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
