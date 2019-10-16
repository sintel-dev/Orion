# -*- coding: utf-8 -*-

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0-dev'

import os

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLBLOCKS_PRIMITIVES = os.path.join(_BASE_PATH, 'primitives', 'jsons')
MLBLOCKS_PIPELINES = os.path.join(_BASE_PATH, 'pipelines')


PIPELINES = {
    'Dummy': os.path.join(MLBLOCKS_PIPELINES, 'dummy.json'),
    'LSTM Dynamic Thresholding': os.path.join(MLBLOCKS_PIPELINES, 'lstm_dynamic_threshold.json'),
    'CycleGAN': os.path.join(MLBLOCKS_PIPELINES, 'cyclegan.json'),
    'ARIMA': os.path.join(MLBLOCKS_PIPELINES, 'arima.json'),
    'Sum 24h LSTM': os.path.join(MLBLOCKS_PIPELINES, 'mean_24h_lstm.json'),
    'Mean 24h LSTM': os.path.join(MLBLOCKS_PIPELINES, 'median_24h_lstm.json'),
    'Median 24h LSTM': os.path.join(MLBLOCKS_PIPELINES, 'sum_24h_lstm.json'),
    'Skew 24h LSTM': os.path.join(MLBLOCKS_PIPELINES, 'skew_24h_lstm.json'),
}
