# -*- coding: utf-8 -*-

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0-dev'

import os

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLPRIMITIVES_JSONS_PATH = os.path.join(_BASE_PATH, 'primitives', 'jsons')

_PIPELINES_FOLDER = os.path.join(_BASE_PATH, 'pipelines')
PIPELINES = {
    'Dummy': os.path.join(_PIPELINES_FOLDER, 'dummy.json'),
    'LSTM Dynamic Thresholding': os.path.join(
        _PIPELINES_FOLDER,
        'lstm_dynamic_threshold.json'
    ),
    'Sum 24h LSTM': os.path.join(_PIPELINES_FOLDER, 'mean_24h_lstm.json'),
    'Mean 24h LSTM': os.path.join(_PIPELINES_FOLDER, 'median_24h_lstm.json'),
    'Median 24h LSTM': os.path.join(_PIPELINES_FOLDER, 'sum_24h_lstm.json'),
    'Skew 24h LSTM': os.path.join(_PIPELINES_FOLDER, 'skew_24h_lstm.json'),
}
