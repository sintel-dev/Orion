# -*- coding: utf-8 -*-

import logging

from mlblocks import MLPipeline


def clone_pipeline(pipeline):
    return MLPipeline.from_dict(pipeline.to_dict())


def walk(document, transform):
    if not isinstance(document, dict):
        return document

    new_doc = dict()
    for key, value in document.items():
        if isinstance(value, dict):
            value = walk(value, transform)
        elif isinstance(value, list):
            value = [walk(v, transform) for v in value]

        new_key, new_value = transform(key, value)
        new_doc[new_key] = new_value

    return new_doc


def remove_dots(document):
    return walk(document, lambda key, value: (key.replace('.', '-'), value))


def restore_dots(document):
    return walk(document, lambda key, value: (key.replace('-', '.'), value))


def logging_setup(verbosity=1, logfile=None, logger_name=None):
    logger = logging.getLogger(logger_name)
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
