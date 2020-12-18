"""Orion Functional API.

This module provides a collection of simple python functions that
allow using Orion performing as little steps as possible, hidding
away all the complexity related to loading data, creating class
instances or serializing and de-serializing previously fitted
pipelines.

Currently implemented functions:
- `fit_pipeline`: Learn an Orion pipeline and save it for later usage.
- `detect_anomalies`: Analyze a signal to detect anomalies. Optionally learn
  a pipeline on the way.
- `evaluate_pipeline`: Evaluate the performance of a pipeline against a list
  of known anomalies.
"""

import json
import os
from pickle import UnpicklingError
from typing import List, Union

import pandas as pd
from mlblocks import MLPipeline

from orion.core import Orion


def _load_data(data):
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, str):
        return pd.read_csv(data)


def _load_dict(path_or_dict):
    if isinstance(path_or_dict, dict):
        return path_or_dict
    elif isinstance(path_or_dict, str) and os.path.exists(path_or_dict):
        with open(path_or_dict) as json_file:
            return json.load(json_file)


def _load_orion(pipeline, hyperparameters=None):
    if pipeline is None:
        return Orion()
    elif isinstance(pipeline, Orion):
        return pipeline
    else:
        hyperparameters = _load_dict(hyperparameters)
        try:
            return Orion(pipeline, hyperparameters)
        except ValueError:
            try:
                return Orion.load(pipeline)
            except (FileNotFoundError, UnpicklingError):
                raise ValueError('Invalid pipeline: {}'.format(pipeline))


def fit_pipeline(data: Union[str, pd.DataFrame],
                 pipeline: Union[str, MLPipeline, dict] = None,
                 hyperparameters: Union[str, pd.DataFrame] = None,
                 save_path: str = None) -> Orion:
    """Fit an Orion pipeline to the data.

    The pipeine can be passed as:
        * An ``str`` with a path to a JSON file.
        * An ``str`` with the name of a registered Orion pipeline.
        * An ``MLPipeline`` instance.
        * A ``dict`` with an ``MLPipeline`` specification.

    If no pipeline is passed, the default Orion pipeline is used.

    Args:
        data (str or DataFrame):
            Data to which the pipeline should be fitted.
            It can be passed as a path to a CSV file or as a DataFrame.
        pipeline (str, Pipeline or dict):
            Pipeline to use. It can be passed as:
                * An ``str`` with a path to a JSON file.
                * An ``str`` with the name of a registered pipeline.
                * An ``MLPipeline`` instance.
                * A ``dict`` with an ``MLPipeline`` specification.
        hyperparameters (str or dict):
            Hyperparameters to set to the pipeline. It can be passed as a
            hyperparameters ``dict`` in the ``mlblocks`` format or as a
            path to the corresponding JSON file. Defaults to
            ``None``.
        save_path (str):
            Path to the file where the fitted Orion instance will be stored
            using ``pickle``. If not given, the Orion instance is returned.
            Defaults to ``None``.

    Returns:
        Orion:
            If no save_path is provided, the fitted Orion instance is returned.
    """
    data = _load_data(data)
    hyperparameters = _load_dict(hyperparameters)

    if pipeline is None:
        pipeline = Orion.DEFAULT_PIPELINE

    orion = Orion(pipeline, hyperparameters)

    orion.fit(data)

    if save_path:
        orion.save(save_path)
    else:
        return orion


def detect_anomalies(data: Union[str, pd.DataFrame] = None,
                     pipeline: Union[Orion, str, MLPipeline, dict] = None,
                     hyperparameters: Union[str, pd.DataFrame] = None,
                     train_data: Union[str, pd.DataFrame] = None) -> pd.DataFrame:
    """Detect anomalies on timeseries data.

    The anomalies are detected using an Orion pipeline which can
    be passed as:

        * An ``Orion`` instance.
        * An ``str`` with the path to an Orion pickle file.
        * An ``str`` with a path to a JSON file.
        * An ``str`` with the name of a registered Orion pipeline.
        * An ``MLPipeline`` instance.
        * A ``dict`` with an ``MLPipeline`` specification.

    If no pipeline is passed, the default Orion pipeline is used.

    Optionally, separated learning data can be passed to fit
    the pipeline to it before using it to detect anomalies.

    Args:
        data (str or DataFrame):
            Data to analyze searching for anomalies.
            It can be passed as a path to a CSV file or as a DataFrame.
        pipeline (str or Pipeline or dict):
            Pipeline to use. It can be passed as:
                * An ``Orion`` instance.
                * An ``str`` with the path to an Orion pickle file.
                * An ``str`` with a path to a JSON file.
                * An ``str`` with the name of a registered pipeline.
                * An ``MLPipeline`` instance.
                * A ``dict`` with an ``MLPipeline`` specification.
        hyperparameters (str or dict):
            Hyperparameters to set to the pipeline. It can be passed as a
            hyperparameters ``dict`` in the ``mlblocks`` format or as a
            path to the corresponding JSON file. Ignored if being passed a
            previously serialized ``Orion`` instance. Defaults to ``None``.
        train_data (str or DataFrame):
            Data to which the pipeline should be fitted.
            It can be passed as a path to a CSV file or as a DataFrame.
            If not given, the pipeline is used without fitting it first.

    Returns:
        DataFrame:
            ``pandas.DataFrame`` containing the detected anomalies.
    """
    data = _load_data(data)
    orion = _load_orion(pipeline, hyperparameters)

    if train_data is not None:
        train_data = _load_data(train_data)
        orion.fit(train_data)

    return orion.detect(data)


def evaluate_pipeline(data: Union[str, pd.DataFrame],
                      truth: Union[str, pd.DataFrame],
                      pipeline: Union[str, dict, MLPipeline],
                      hyperparameters: Union[str, pd.DataFrame] = None,
                      metrics: List[Union[callable, str]] = None,
                      train_data: Union[str, pd.DataFrame] = None) -> pd.DataFrame:
    """Evaluate the performance of a pipeline.

    The pipeline is evaluated by executing it on a signal
    for which anomalies are known and then applying one or
    more metrics to it to compute scores.

    The pipeline can be passed as:
        * An ``str`` with a path to a JSON file.
        * An ``str`` with the path to a pickle file.
        * An ``str`` with the name of a registered Orion pipeline.
        * An ``MLPipeline`` instance.
        * A ``dict`` with an ``MLPipeline`` specification.

    If the pipeline is not fitted, it is possible to pass separated
    learning data can be passed to fit the pipeline to it before using
    it to detect anomalies.

    Args:
        data (str or DataFrame):
            Data to analyze searching for anomalies.
            It can be passed as a path to a CSV file or as a DataFrame.
        truth (str or DataFrame):
            Table of known anomalies to use as the ground truth for
            scoring. It can be passed as a path to a CSV file or as a
            DataFrame.
        pipeline (str or Pipeline or dict):
            Pipeline to use. It can be passed as:
                * An ``str`` with a path to a JSON file.
                * An ``str`` with the name of a registered pipeline.
                * An ``str`` with the path to a pickle file.
                * An ``MLPipeline`` instance.
                * A ``dict`` with an ``MLPipeline`` specification.
        hyperparameters (str or dict):
            Hyperparameters to set to the pipeline. It can be passed as
            a hyperparameters ``dict`` in the ``mlblocks`` format or as
            a path to the corresponding JSON file. Defaults to ``None``.
        metrics (list[str]):
            List of metrics to use. If not passed, all the Orion metrics
            are applied.
        train_data (str or DataFrame):
            Data to which the pipeline should be fitted.
            It can be passed as a path to a CSV file or as a DataFrame.
            If not given, the pipeline is used without fitting it first.
    """
    data = _load_data(data)
    truth = _load_data(truth)
    fit = train_data is not None
    if fit:
        train_data = _load_data(train_data)

    orion = _load_orion(pipeline, hyperparameters)

    return orion.detect(data, truth, fit, train_data, metrics)
