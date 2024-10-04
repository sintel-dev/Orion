# -*- coding: utf-8 -*-

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.6.1'

import os

from orion.core import Orion
from orion.functional import detect_anomalies, evaluate_pipeline, fit_pipeline

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLBLOCKS_PRIMITIVES = os.path.join(_BASE_PATH, 'primitives', 'jsons')
MLBLOCKS_PIPELINES = tuple(
    dirname
    for dirname, _, _ in os.walk(os.path.join(_BASE_PATH, 'pipelines'))
)

__all__ = (
    'Orion',
    'detect_anomalies',
    'evaluate_pipeline',
    'fit_pipeline'
)
