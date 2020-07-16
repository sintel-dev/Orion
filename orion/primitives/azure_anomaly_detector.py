import logging
from datetime import datetime

import numpy as np
from azure.cognitiveservices.anomalydetector import AnomalyDetectorClient
from azure.cognitiveservices.anomalydetector.models import APIErrorException, Point, Request
from msrest.authentication import CognitiveServicesCredentials

LOGGER = logging.getLogger(__name__)


def split_sequence(X, index, target_column, sequence_size, pad_size):
    """Split sequences of time series data.

    The function creates a list of input sequences by splitting the input sequence
    into partitions with a specified size and pad it be a spceificied size.

    Args:
        X (ndarray):
            N-dimensional value sequence to iterate over.
        index (ndarray):
            N-dimensional index sequence to iterate over.
        target_column (int):
            Indicating which column of X is the target.
        sequence_size (int):
            Length of the input sequences.
        pad_size (int):
            Length of the values from previous window.

    Returns:
        list:
            * A list of sliced value ndarray.
            * A list of sliced index ndarray.
    """
    X_ = list()
    index_ = list()

    min_pad = 0
    start = 0
    max_start = len(X) - 1

    target = X[:, target_column]

    while start < max_start:
        end = start + sequence_size

        X_.append(target[start - min_pad:end])
        index_.append(index[start - min_pad:end])

        start = end
        min_pad = pad_size

    return np.asarray(X_), np.asarray(index_)


def detect_anomalies(X, index, interval, pad_size, subscription_key, endpoint, granularity,
                     custom_interval=None, period=None, max_anomaly_ratio=None, sensitivity=None):
    """Microsoft's Azure Anomaly Detection tool.

    Args:
        X (ndarray):
            N-dimensional array containing the input value sequences.
        index (ndarray):
            N-dimensional array containing the input index sequences.
        interval (int):
            Integer denoting time span frequency of the data.
        pad_size (int):
            Length of the values from previous sequence.
        subscription_key (str):
            Resource key for authenticating your requests.
        endpoint (str):
            Resource endpoint for sending API requests.
        granularity (str or Granularity):
            Can only be one of yearly, monthly, weekly, daily, hourly or minutely.
            Granularity is used for verify whether input series is valid. Possible values
            include: 'yearly', 'monthly', 'weekly', 'daily', 'hourly', 'minutely'.
        custom_interval (int):
            Integer used to set non-standard time interval, for example, if the series
            is 5 minutes, request can be set as {"granularity":"minutely", "custom_interval":5}.
            If not given, `None` is used.
        period (int):
            Periodic value of a time series. If not given, `None` is used, and the API
            will determine the period automatically. 
        max_anomaly_ratio (float):
            Advanced model parameter, max anomaly ratio in a time series. If not given, 
            `None` is used.
        sensitivity (int):
            Advanced model parameter, between 0-99, the lower the value is, the larger
            the margin value will be which means less anomalies will be accepted. If not given, 
            `None` is used.

        Returns:
            list:
                Array containing start-index, end-index, score for each anomalous sequence. 
                Note that the API does not have an anomaly score, and so score is set to `None`.
    """

    def _convert_date(x):
        return datetime.fromtimestamp(x).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    SUBSCRIPTION_KEY = subscription_key
    ANOMALY_DETECTOR_ENDPOINT = endpoint

    client = AnomalyDetectorClient(
        ANOMALY_DETECTOR_ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY))

    min_pad = 0
    result = list()

    for x, idx in zip(X, index):
        series = []
        for i in range(len(x)):
            idx_ = _convert_date(idx[i])
            series.append(Point(timestamp=idx_, value=x[i]))

        request = Request(
            series=series, granularity=granularity, custom_interval=custom_interval, period=period,
            max_anomaly_ratio=max_anomaly_ratio, sensitivity=sensitivity)

        try:
            response = client.entire_detect(request)
            if response.is_anomaly:
                anomalous = response.is_anomaly[min_pad:]
                index_ = idx[min_pad:]
                result.extend(index_[anomalous])

        except Exception as e:
            if isinstance(e, APIErrorException):
                print('Error code: {}'.format(e.error.code),
                      'Error message: {}'.format(e.error.message))

            else:
                print(e)

        min_pad = pad_size

    return _convert_anomalies_to_contextual(result, interval)


def _convert_anomalies_to_contextual(X, interval=1):
    """ Convert list of timestamps to list of tuples.

    Convert a list of anomalies identified by timestamps,
    to a list of tuples marking the start and end interval
    of anomalies; make it contextually defined.

    Args:
        X (list): contains timestamp of anomalies.
        interval (int): allowed gap between anomalies.

    Returns:
        list:
            tuple (start, end, `None`) timestamp.
    """
    if len(X) == 0:
        return []

    X = sorted(X)

    start_ts = 0
    max_ts = len(X) - 1

    anomalies = list()
    break_point = start_ts
    while break_point < max_ts:
        if X[break_point + 1] - X[break_point] <= interval:
            break_point += 1
            continue

        anomalies.append((X[start_ts], X[break_point], None))
        break_point += 1
        start_ts = break_point

    anomalies.append((X[start_ts], X[break_point], None))
    return anomalies
