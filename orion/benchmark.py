# -*- coding: utf-8 -*-

import ast
import json
import logging
import os
from datetime import datetime
from functools import partial

import dask
import pandas as pd
from scipy import signal as scipy_signal

from orion.analysis import _load_pipeline, analyze
from orion.data import load_anomalies, load_signal
from orion.evaluation import CONTEXTUAL_METRICS as METRICS
from orion.evaluation import contextual_confusion_matrix

LOGGER = logging.getLogger(__name__)

BUCKET = 'd3-ai-orion'
S3_URL = 'https://{}.s3.amazonaws.com/{}'

BENCHMARK_PATH = os.path.join(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'),
    'benchmark'
)

BENCHMARK_DATA = pd.read_csv(S3_URL.format(
    BUCKET, 'datasets.csv'), index_col=0, header=None).applymap(ast.literal_eval).to_dict()[1]
BENCHMARK_PARAMS = pd.read_csv(S3_URL.format(
    BUCKET, 'parameters.csv'), index_col=0, header=None).applymap(ast.literal_eval).to_dict()[1]

PIPELINE_DIR = os.path.join(os.path.dirname(__file__), 'pipelines', 'verified')

VERIFIED_PIPELINES = [
    'arima', 'lstm_dynamic_threshold'
]

VERIFIED_PIPELINES_GPU = {
    'arima': 'arima',
    'lstm_dynamic_threshold': 'lstm_dynamic_threshold_gpu'
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


def _evaluate_signal(pipeline, name, dataset, signal, hyperparameter, metrics,
                     holdout=True, detrend=False):

    train, test = _load_signal(signal, holdout)
    truth = load_anomalies(signal)

    if detrend:
        train = _detrend_signal(train, 'value')
        test = _detrend_signal(test, 'value')

    try:
        LOGGER.info("Scoring pipeline %s on signal %s (Holdout: %s)",
                    name, signal, holdout)

        start = datetime.utcnow()
        pipeline = _load_pipeline(pipeline, hyperparameter)
        anomalies = analyze(pipeline, train, test)
        elapsed = datetime.utcnow() - start

        scores = {
            name: scorer(truth, anomalies, test)
            for name, scorer in metrics.items()
        }
        scores['status'] = 'OK'

    except Exception as ex:
        LOGGER.exception("Exception scoring pipeline %s on signal %s (Holdout: %s), error %s.",
                         name, signal, holdout, ex)

        elapsed = datetime.utcnow() - start
        scores = {
            name: 0 for name in metrics.keys()
        }

        metric_ = 'confusion_matrix'
        if metric_ in metrics.keys():
            fn = len(truth)
            scores[metric_] = (None, 0, fn, 0)  # (tn, fp, fn, tp)

        scores['status'] = 'ERROR'

    scores['elapsed'] = elapsed.total_seconds()
    scores['pipeline'] = name
    scores['holdout'] = holdout
    scores['dataset'] = dataset
    scores['signal'] = signal

    return scores


def _evaluate_pipeline(pipeline, pipeline_name, dataset, signals, hyperparameter, metrics,
                       distributed, holdout, detrend):
    if holdout is None:
        holdout = (True, False)
    elif not isinstance(holdout, tuple):
        holdout = (holdout, )

    # hyperparameter settings
    if hyperparameter is None:
        file_path = os.path.join(
            PIPELINE_DIR, pipeline_name, pipeline_name + '_' + dataset.lower() + '.json')
        if os.path.exists(file_path):
            hyperparameter = file_path

    if isinstance(hyperparameter, str) and os.path.exists(hyperparameter):
        LOGGER.info("Loading hyperparameter %s", hyperparameter)
        with open(hyperparameter) as f:
            hyperparameter = json.load(f)

    if distributed:
        function = dask.delayed(_evaluate_signal)
    else:
        function = _evaluate_signal

    scores = list()

    for signal in signals:
        for holdout_ in holdout:
            score = function(pipeline, pipeline_name, dataset, signal, hyperparameter,
                             metrics, holdout_, detrend)

            scores.append(score)

    return scores


def _evaluate_pipelines(pipelines, dataset, signals, hyperparameters, metrics, distributed,
                        holdout, detrend):

    scores = list()
    for name, pipeline in pipelines.items():
        hyperparameter = _get_parameter(hyperparameters, name)
        score = _evaluate_pipeline(pipeline, name, dataset, signals, hyperparameter,
                                   metrics, distributed, holdout, detrend)
        scores.extend(score)

    return scores


def _evaluate_datasets(pipelines, datasets, hyperparameters, metrics, distributed, holdout,
                       detrend):
    delayed = []
    for dataset, signals in datasets.items():
        LOGGER.info("Starting dataset {} with {} signals..".format(
            dataset, len(signals)))

        # dataset configuration
        hyperparameters_ = _get_parameter(hyperparameters, dataset)
        parameters = _get_parameter(BENCHMARK_PARAMS, dataset)
        if parameters is not None:
            detrend, holdout = parameters.values()

        result = _evaluate_pipelines(
            pipelines, dataset, signals, hyperparameters_, metrics, distributed, holdout, detrend)

        delayed.extend(result)

    if distributed:
        persisted = dask.persist(*delayed)
        results = dask.compute(*persisted)

    else:
        results = delayed

    df = pd.DataFrame.from_records(results)
    return df


def _summarize_results(df, metrics):
    """ Summarize the result of benchmark.

    The table is summarized for according to the number of wins each pipeline
    had over ARIMA pipeline per dataset, the number of anomalies detected, and
    the average f1 score acheived by that pipeline.
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


def benchmark(pipelines=None, datasets=None, hyperparameters=None, metrics=METRICS, rank='f1',
              distributed=False, holdout=False, detrend=False, output_path=None):
    """Evaluate pipelines on the given datasets and evaluate the performance.

    The pipelines are used to analyze the given signals and later on the
    detected anomalies are scored against the known anomalies using the
    indicated metrics.

    Finally, the scores obtained with each metric are averaged accross all the signals,
    ranked by the indicated metric and returned on a ``pandas.DataFrame``.

    Args:
        pipelines (dict or list): dictionary with pipeline names as keys and their
            JSON paths as values. If a list is given, it should be of JSON paths,
            and the paths themselves will be used as names. If not give, all verified
            pipelines will be used for evaluation.
        datasets (dict or list): dictionary of dataset name as keys and list of signals as
            values. If a list is given then it will be under a generic name ``dataset``.
            If not given, all benchmark datasets will be used used.
        hyperparameters (dict or list): dictionary with pipeline names as keys
            and their hyperparameter JSON paths or dictionaries as values. If a list is
            given, it should be of corresponding order to pipelines.
        metrics (dict or list): dictionary with metric names as keys and
            scoring functions as values. If a list is given, it should be of scoring
            functions, and they ``__name__`` value will be used as the metric name.
            If not given, all the available metrics will be used.
        rank (str): Sort and rank the pipelines based on the given metric.
            If not given, rank using the first metric.
        distributed (bool): Whether to use dask for distributed computing. If not given,
            use ``False``.
        holdout (bool): Whether to use the prespecified train-test split. If not given,
            use ``False``.
        detrend (bool): Whether to use ``scipy.detrend``. If not given, use ``False``.
        output_path (str): Location to save the intermediatry results. If not given,
            intermediatry results will not be saved.

    Returns:
        pandas.DataFrame: Table containing the average of the scores obtained with
            each scoring function accross all the signals for each pipeline, ranked
            by the indicated metric.
    """
    pipelines = pipelines or VERIFIED_PIPELINES
    datasets = datasets or BENCHMARK_DATA

    if isinstance(pipelines, list):
        pipelines = {pipeline: pipeline for pipeline in pipelines}

    if isinstance(datasets, list):
        datasets = {'dataset': datasets}

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

    results = _evaluate_datasets(
        pipelines, datasets, hyperparameters, metrics, distributed, holdout, detrend)

    if output_path:
        LOGGER.info('Saving benchmark report to %s', output_path)
        results.to_csv(output_path)

    return _sort_leaderboard(results, rank, metrics)


def main(cuda=False, distributed=False):
    # output path
    version = "results.csv"
    output_path = os.path.join(BENCHMARK_PATH, 'results', version)

    # metrics
    del METRICS['accuracy']
    METRICS['confusion_matrix'] = contextual_confusion_matrix
    metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

    # pipelines
    pipelines = VERIFIED_PIPELINES
    if cuda:
        pipelines = VERIFIED_PIPELINES_GPU

    results = benchmark(
        pipelines=pipelines, metrics=metrics, output_path=output_path, distributed=distributed)

    leaderboard = _summarize_results(results, metrics)
    output_path = os.path.join(BENCHMARK_PATH, 'leaderboard.csv')
    leaderboard.to_csv(output_path)


if __name__ == "__main__":
    main()
