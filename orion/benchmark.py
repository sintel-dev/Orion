# -*- coding: utf-8 -*-

import ast
import csv
import json
import logging
import os
import warnings
from datetime import datetime
from functools import partial

import dask
import pandas as pd
from scipy import signal as scipy_signal

from orion.analysis import _load_pipeline, analyze
from orion.data import NASA_SIGNALS, load_anomalies, load_signal
from orion.evaluation import CONTEXTUAL_METRICS as METRICS
from orion.evaluation import contextual_confusion_matrix

LOGGER = logging.getLogger(__name__)

BENCHMARK_PATH = os.path.join(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'),
    'benchmarking'
)

with open(os.path.join(BENCHMARK_PATH, 'benchmark_data.csv'), newline='') as f:
    reader = csv.reader(f)
    BENCHMARK_DATA = {row[0]: ast.literal_eval(row[1]) for row in reader}

with open(os.path.join(BENCHMARK_PATH, 'benchmark_parameters.csv'), newline='') as f:
    reader = csv.reader(f)
    BENCHMARK_PARAMS = {row[0]: ast.literal_eval(row[1]) for row in reader}

BENCHMARK_HYPER = pd.read_csv(
    os.path.join(BENCHMARK_PATH, 'benchmark_hyperparameters.csv'), index_col=0).to_dict()

VERIFIED_PATH = os.path.join(os.path.dirname(__file__), 'pipelines', 'verified')

VERIFIED_PIPELINES = [
    os.path.basename(dirname)
    for dirname in os.listdir(VERIFIED_PATH)
]

VERIFIED_PIPELINES_GPU = {
    pipeline: pipeline + '_gpu'
    if os.path.exists(os.path.join(VERIFIED_PATH, pipeline, pipeline + '_gpu.json'))
    else pipeline
    for pipeline in VERIFIED_PIPELINES
}


def _load_signal(signal, holdout):
    if holdout:
        train = load_signal(signal + '-train')
        test = load_signal(signal + '-test')
    else:
        train = test = load_signal(signal)

    return train, test


def _detrend_signal(df, value_column):
    df[value_column] = scipy_signal.detrend(df[value_column])
    return df


def _get_parameter(parameters, name):
    if isinstance(parameters, dict) and name in parameters.keys():
        return parameters[name]

    return None


def _sort_leaderboard(df, rank, metrics):
    if rank not in df.columns:
        rank_ = list(metrics.keys())[0]
        LOGGER.exception("Rank %s is not in %s, using %s instead.",
                         rank, df.columns, rank_)
        rank = rank_

    df.sort_values(rank, ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'rank'
    df.reset_index(drop=False, inplace=True)
    df['rank'] += 1

    return df.set_index('pipeline').reset_index()


@dask.delayed
def _evaluate_on_signal(pipeline, name, dataset, signal, hyperparameter, metrics,
                        holdout=True, detrend=False):

    train, test = _load_signal(signal, holdout)

    if detrend:
        train = _detrend_signal(train, 'value')
        test = _detrend_signal(test, 'value')

    try:
        LOGGER.info("Scoring pipeline %s on signal %s (Holdout: %s)",
                    name, signal, holdout)

        pipeline = _load_pipeline(pipeline, hyperparameter)
        start = datetime.utcnow()
        anomalies = analyze(pipeline, train, test)
        elapsed = datetime.utcnow() - start

        truth = load_anomalies(signal)

        scores = {
            name: scorer(truth, anomalies, test)
            for name, scorer in metrics.items()
        }
        scores['elapsed'] = elapsed.total_seconds()

    except Exception as ex:
        LOGGER.exception("Exception scoring pipeline %s on signal %s (Holdout: %s), error %s.",
                         name, signal, holdout, ex)

        scores = {
            name: 0 for name in metrics.keys()
        }
        scores['elapsed'] = 0

    scores['pipeline'] = name
    scores['holdout'] = holdout
    scores['dataset'] = dataset
    scores['signal'] = signal

    return scores


def _evaluate_pipeline(pipeline, name, dataset, signals, hyperparameter, metrics, holdout,
                       detrend):
    if holdout is None:
        holdout = (True, False)
    elif not isinstance(holdout, tuple):
        holdout = (holdout, )

    if isinstance(hyperparameter, str) and os.path.exists(hyperparameter):
        LOGGER.info("Loading hyperparameter %s", hyperparameter)
        with open(hyperparameter) as f:
            hyperparameter = json.load(f)

    scores = list()

    for signal in signals:
        for holdout_ in holdout:
            score = _evaluate_on_signal(
                pipeline, name, dataset, signal, hyperparameter, metrics, holdout_, detrend)

            scores.append(score)

    return scores


def evaluate_pipeline(pipeline, pipeline_name=None, dataset="NASA", signals=NASA_SIGNALS,
                      hyperparameter=None, metrics=METRICS, holdout=None, detrend=False):
    """Evaluate a pipeline on multiple signals with multiple metrics.

    The pipeline is used to analyze the given signals and later on the
    detected anomalies are scored against the known anomalies using the
    indicated metrics.

    Args:
        pipeline (str): Path to the pipeline JSON.
        pipeline_name (str): Name of the pipeline. If not given, use JSON path as name.
        dataset (str): Name of the dataset which the signals belong to. If not
            given, use NASA as default.
        signals (list): List of signals. If not given, all the NASA signals
            are used.
        hyperparameter (str or dict): Path to or dictionary of hyperparameters.
        metrics (dict): Dictionary with metric names as keys and scoring functions as
            values. If not given, all the available metrics willbe used.

    Returns:
        pandas.Series: Series object containing the average of the scores obtained with
            each scoring function accross all the signals.
    """
    pipeline_name = pipeline_name or pipeline
    scores = _evaluate_pipeline(
        pipeline, pipeline_name, dataset, signals, hyperparameter, metrics, holdout, detrend)

    scores = dask.compute(*scores)
    scores = pd.DataFrame.from_records(scores).groupby('holdout').mean().reset_index()

    # Move holdout and elapsed column to the last position
    scores['elapsed'] = scores.pop('elapsed')

    return scores


def _evaluate_pipelines(pipelines, dataset, signals, hyperparameters, metrics, holdout, detrend):
    if isinstance(pipelines, list):
        pipelines = {pipeline: pipeline for pipeline in pipelines}

    if isinstance(hyperparameters, list):
        hyperparameters = {pipeline: hyperparameter for pipeline, hyperparameter in
                           zip(pipelines.keys(), hyperparameters)}

    if isinstance(metrics, list):
        metrics_ = dict()
        for metric in metrics:
            if callable(metric):
                metrics_[metric.__name__] = metric
            elif metric in METRICS:
                metrics_[metric] = METRICS[metric]
            else:
                raise ValueError('Unknown metric: {}'.format(metric))

        metrics = metrics_

    scores = list()
    for name, pipeline in pipelines.items():
        hyperparameter = _get_parameter(hyperparameters, name)
        score = _evaluate_pipeline(
            pipeline, name, dataset, signals, hyperparameter, metrics, holdout, detrend)

        scores.extend(score)

    return scores


def evaluate_pipelines(pipelines, dataset=None, signals=None, hyperparameters=None, metrics=None,
                       rank=None, holdout=(True, False), detrend=False):
    """Evaluate a list of pipelines on multiple signals with multiple metrics.

    The pipelines are used to analyze the given signals and later on the
    detected anomalies are scored against the known anomalies using the
    indicated metrics.

    Finally, the scores obtained with each metric are averaged accross all the signals,
    ranked by the indicated metric and returned on a pandas.DataFrame.

    Args:
        pipelines (dict or list): dictionary with pipeline names as keys and their
            JSON paths as values. If a list is given, it should be of JSON paths,
            and the paths themselves will be used as names.
        dataset (str, optional): name of the dataset which the signals belong to. If not
            given, use NASA as default.
        signals (list, optional): list of signals. If not given, all the NASA signals
            are used.
        hyperparameters (dict or list, optional): dictionary with pipeline names as keys
            and their hyperparameter JSON paths or dictionaries as values. If a list is
            given, it should be of corresponding order to pipelines.
        metrics (dict or list, optional): dictionary with metric names as keys and
            scoring functions as values. If a list is given, it should be of scoring
            functions, and they `__name__` value will be used as the metric name.
            If not given, all the available metrics will be used.
        rank (str, optional): Sort and rank the pipelines based on the given metric.
            If not given, rank using the first metric.

    Returns:
        pandas.DataFrame: Table containing the average of the scores obtained with
            each scoring function accross all the signals for each pipeline, ranked
            by the indicated metric.
    """
    dataset = dataset or "NASA"
    signals = signals or NASA_SIGNALS
    metrics = metrics or METRICS

    scores = _evaluate_pipelines(
        pipelines, dataset, signals, hyperparameters, metrics, holdout, detrend)

    scores = dask.compute(*scores)
    scores = pd.DataFrame.from_records(scores)

    return _sort_leaderboard(scores, rank, metrics)


def benchmark(pipelines=None, datasets=None, hyperparameters=None, metrics=METRICS):
    delayed = []

    for dataset, signals in datasets.items():
        LOGGER.info("Starting dataset {} with {} signals..".format(
            dataset, len(signals)))

        # set dataset configuration
        hyperparameters_ = _get_parameter(hyperparameters, dataset)
        parameters = _get_parameter(BENCHMARK_PARAMS, dataset) or dict()

        result = _evaluate_pipelines(
            pipelines, dataset, signals, hyperparameters_, metrics, **parameters)

        delayed.extend(result)

    persisted = dask.persist(*delayed)
    results = dask.compute(*persisted)
    df = pd.DataFrame.from_records(results)

    return df


def run_benchmark(pipelines=None, datasets=None, hyperparameters=None, metrics=METRICS, rank='f1',
                  output_path=None):
    """Benchmark a list of pipelines on multiple signals with multiple metrics.

    The pipelines are used to analyze the given signals and later on the
    detected anomalies are scored against the known anomalies using the
    indicated metrics.

    Finally, the scores obtained with each metric are averaged accross all the signals,
    ranked by the indicated metric and returned on a `pandas.DataFrame`.

    Args:
        pipelines (dict or list, optional): dictionary with pipeline names as keys and
            their JSON paths as values. If a list is given, it should be of JSON paths,
            and the paths themselves will be used as names. If `None` is given, all
            available pipelines under `orion/pipelines` will be used.
        datasets (dict or list, optional): list of signals or dictionary of dataset name
            as keys and list of signals as values. If not given, all the datasets defined
            in `BENCHMARK_DATA` will be used.
        hyperparameters (dict, optional): dictionary with dataset name, pipeline name
            as keys and hyperparameter JSON settings path or dictionary as values.
            If not given, use default hyperparameters.
        metrics (dict or list, optional): dictionary with metric names as keys and
            scoring functions as values. If a list is given, it should be of scoring
            functions, and their `__name__` value will be used as the metric name.
            If not given, all the available metrics will be used.
        rank (str, optional): Sort and rank the pipelines based on the given metric.
            If not given, rank using the first metric.
        output_path (str, optional):
            location to save the intermediatry results. If not given, intermediatry
            results will not be saved.

    Returns:
        pandas.DataFrame: Table containing the average of the scores obtained with
            each scoring function accross all the signals for each pipeline, ranked
            by the indicated metric.
    """
    pipelines = pipelines or VERIFIED_PIPELINES
    datasets = datasets or BENCHMARK_DATA
    hyperparameters = hyperparameters or BENCHMARK_HYPER

    results = benchmark(pipelines, datasets, hyperparameters, metrics)

    if output_path:
        LOGGER.info('Saving benchmark report to %s', output_path)
        results.to_csv(output_path)

    return results


def summarize_results(df, metrics):
    """ Summarize the result of benchmark.

    The table is summarized for according to the number of wins each pipeline
    had over ARIMA pipeline per dataset, the number of anomalies detected, and
    the average f1 score acheived by that pipeline.

    Args:
        df (`pandas.DataFrame`): detailed dataframe containing the score of each
            pipeline on each signal.
        metrics (dict): dictionary with metric names as keys and scoring functions
            as values.

    Returns:
        pandas.DataFrame: Table containing the the number of wins each pipeline
            had over ARIMA pipeline per dataset, the number of anomalies detected, and
            the average f1 score acheived by that pipeline. The pipelines are ranked
            by the average f1 score metric.
    """
    def return_cm(x):
        if isinstance(x, int):
            return (0, 0, 0)

        elif len(x) > 3:
            return x[1:]

        return x

    df['confusion_matrix'] = df['confusion_matrix'].apply(return_cm)
    df[['fp', 'fn', 'tp']] = pd.DataFrame(df['confusion_matrix'].tolist(), index=df.index)

    # calculate f1 score
    df = df.groupby(['dataset', 'pipeline'])[['fp', 'fn', 'tp']].sum().reset_index()

    precision = df['tp'] / (df['tp'] + df['fp'])
    recall = df['tp'] / (df['tp'] + df['fn'])
    df['f1'] = 2 * (precision * recall) / (precision + recall)

    result = dict()

    # number of wins over ARIMA
    arima_pipeline = 'arima'
    intermediate = df.set_index(['pipeline', 'dataset'])['f1'].unstack().T
    arima = intermediate.pop(arima_pipeline)

    result['# Wins'] = (intermediate.T > arima).sum(axis=1)
    result['# Wins'][arima_pipeline] = None

    # number of anomalies detected
    result['# Anomalies'] = df.groupby('pipeline')[['tp', 'fp']].sum().sum(axis=1).to_dict()

    # average f1 score
    result['Average F1 Score'] = df.groupby('pipeline')['f1'].mean().to_dict()

    result = pd.DataFrame(result)
    result.index.name = 'pipeline'
    result.reset_index(inplace=True)

    rank = 'Average F1 Score'
    result = _sort_leaderboard(result, rank, metrics)
    result = result.drop('rank', axis=1).set_index('pipeline')

    return result


def main(gpu=False):
    warnings.filterwarnings("ignore")

    # output path
    version = "example.csv"
    output_path = os.path.join(BENCHMARK_PATH, 'results', version)

    # metrics
    del METRICS['accuracy']
    METRICS['confusion_matrix'] = contextual_confusion_matrix
    metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

    # pipelines
    pipelines = VERIFIED_PIPELINES
    if gpu:
        pipelines = VERIFIED_PIPELINES_GPU

    results = run_benchmark(pipelines=pipelines, metrics=metrics, output_path=output_path)

    leaderboard = summarize_results(results, metrics)
    output_path = os.path.join(BENCHMARK_PATH, 'leaderboard.csv')
    leaderboard.to_csv(output_path)


if __name__ == "__main__":
    main()
