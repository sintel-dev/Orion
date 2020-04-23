# -*- coding: utf-8 -*-

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0'

import os

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLBLOCKS_PRIMITIVES = os.path.join(_BASE_PATH, 'primitives', 'jsons')
MLBLOCKS_PIPELINES = os.path.join(_BASE_PATH, 'pipelines')


def get_available_templates():
    return [
        filename[:-5]
        for filename in os.listdir(MLBLOCKS_PIPELINES)
    ]
