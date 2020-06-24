import ast
import csv
import json
import logging
import os
import warnings
from datetime import datetime

import pandas as pd
from scipy import signal as scipy_signal

from orion.analysis import analyze
from orion.data import NASA_SIGNALS, load_anomalies, load_signal
from orion.evaluation import CONTEXTUAL_METRICS as METRICS

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)

BENCHMARK_PATH = os.path.join(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'),
    'benchmarking'
)

BENCHMARK_PIPELINES = os.path.join(os.path.dirname(__file__), 'pipelines')

with open('{}/benchmark_data.csv'.format(BENCHMARK_PATH), newline='') as f:
    reader = csv.reader(f)
    BENCHMARK_DATA = {row[0]: ast.literal_eval(row[1]) for row in reader}

with open('{}/benchmark_parameters.csv'.format(BENCHMARK_PATH), newline='') as f:
    reader = csv.reader(f)
    BENCHMARK_PARAMS = {row[0]: ast.literal_eval(row[1]) for row in reader}

BENCHMARK_HYPER = pd.read_csv('{}/benchmark_hyperparameters.csv'.format(
    BENCHMARK_PATH), index_col=0).to_dict()


def _get_pipelines(with_gpu=False):
    pipeline_path = os.path.join(os.path.dirname(__file__), 'pipelines')
    pipelines = (f for f in os.listdir(pipeline_path) if f.endswith('.json'))
    pipelines = sorted(pipelines)

    pipelines_ = dict()
    for pipeline in pipelines:
        name = pipeline.split('/', 1)[-1].replace('.json', '')
        if '_gpu' in name:
            if with_gpu:
                name = name.replace('_gpu', '')
            else:
                continue

        pipelines_[name] = os.path.join(pipeline_path, pipeline)

    return pipelines_


def _get_data(datasets=None):
    if isinstance(datasets, dict):
        return datasets

    elif isinstance(datasets, list):
        if set(datasets).issubset(BENCHMARK_DATA.keys()):
            return {k: v for k, v in BENCHMARK_DATA.items() if k in datasets}
        else:
            return datasets

    return BENCHMARK_DATA


def _get_hyperparameter(hyperparameters, name):
    if isinstance(hyperparameters, dict) and name in hyperparameters.keys():
        return hyperparameters[name]

    return None


def _detrend_signal(df, value_column):
    df[value_column] = scipy_signal.detrend(df[value_column])
    return df


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


def _evaluate_on_signal(pipeline, signal, hyperparameter, metrics, detrend=False, holdout=True):
    if holdout:
        train = load_signal(signal + '-train')
        test = load_signal(signal + '-test')
    else:
        train = test = load_signal(signal)

    if detrend:
        train = _detrend_signal(train, 'value')
        test = _detrend_signal(test, 'value')

    start = datetime.utcnow()
    anomalies = analyze(pipeline, train, test, hyperparameter)
    elapsed = datetime.utcnow() - start

    truth = load_anomalies(signal)

    scores = {
        name: scorer(truth, anomalies, test)
        for name, scorer in metrics.items()
    }
    scores['elapsed'] = elapsed.total_seconds()

    return scores


def evaluate_pipeline(pipeline, signals=NASA_SIGNALS, hyperparameter=None, metrics=METRICS,
                      detrend=False, holdout=None):
    """Evaluate a pipeline on multiple signals with multiple metrics.

    The pipeline is used to analyze the given signals and later on the
    detected anomalies are scored against the known anomalies using the
    indicated metrics.

    Args:
        pipeline (str): Path to the pipeline JSON.
        hyperparameter (str or dict, optional): Path to or dictionary of hyperparameters.
        signals (list, optional): list of signals. If not given, all the NASA signals
            are used.
        metrics (dict, optional): dictionary with metric names as keys and
            scoring functions as values. If not given, all the available metrics will
            be used.

    Returns:
        pandas.Series: Series object containing the average of the scores obtained with
            each scoring function accross all the signals.
    """
    if holdout is None:
        holdout = (True, False)
    elif not isinstance(holdout, tuple):
        holdout = (holdout, )

    if isinstance(hyperparameter, str) and os.path.isfile(hyperparameter):
        LOGGER.info("Using pipeline %s with hyperparameter in %s",
                    pipeline, hyperparameter)
        with open(hyperparameter) as f:
            hyperparameter = json.load(f)

    scores = list()
    for signal in signals:
        for holdout_ in holdout:
            try:
                LOGGER.info("Scoring pipeline %s on signal %s (Holdout: %s)",
                            pipeline, signal, holdout_)
                score = _evaluate_on_signal(
                    pipeline, signal, hyperparameter, metrics, detrend, holdout_)

            except Exception:
                LOGGER.exception("Exception scoring pipeline %s on signal %s (Holdout: %s)",
                                 pipeline, signal, holdout_)
                score = (0, 0)
                score = {name: 0 for name in metrics.keys()}

            score['holdout'] = holdout_
            scores.append(score)

    scores = pd.DataFrame(scores).groupby('holdout').mean().reset_index()

    # Move holdout and elapsed column to the last position
    scores['elapsed'] = scores.pop('elapsed')

    return scores


def evaluate_pipelines(pipelines, signals=None, hyperparameters=None, metrics=None, rank=None,
                       detrend=False, holdout=(True, False)):
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
        hyperparameters (dict or list, optional): dictionary with pipeline names as keys
            and their hyperparameter JSON paths or dictionaries as values. If a list is
            given, it should be of corresponding order to pipelines.
        signals (list, optional): list of signals. If not given, all the NASA signals
            are used.
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
    signals = signals or NASA_SIGNALS
    metrics = metrics or METRICS

    scores = list()
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

    for name, pipeline in pipelines.items():
        hyperparameter = _get_hyperparameter(hyperparameters, name)

        LOGGER.info("Evaluating pipeline: %s", name)
        score = evaluate_pipeline(
            pipeline, signals, hyperparameter, metrics, detrend, holdout)

        score['pipeline'] = name
        score['hyperparameter'] = hyperparameter
        scores.append(score)

    scores = pd.concat(scores)

    return _sort_leaderboard(scores, rank, metrics)


def run_benchmark(pipelines=None, datasets=None, hyperparameters=None, metrics=METRICS, rank='f1',
                  output_path=None, with_gpu=False, **kwargs):
    """Benchmark a list of pipelines on multiple signals with multiple metrics.

    The pipelines are used to analyze the given signals and later on the
    detected anomalies are scored against the known anomalies using the
    indicated metrics.

    Finally, the scores obtained with each metric are averaged accross all the signals,
    ranked by the indicated metric and returned on a pandas.DataFrame.

    Args:
        pipelines (dict or list): dictionary with pipeline names as keys and their
            JSON paths as values. If a list is given, it should be of JSON paths,
            and the paths themselves will be used as names.
        datasets (list, optional): list of signals. If not given, all the NASA, Yahoo,
            and NAB signals are used.
        hyperparameters (dict, optional): dictionary with (dataset name, pipeline name)
            as keys and hyperparameter JSON settings path or dictionary as values.
            If not given, use default hyperparameters.
        metrics (dict or list, optional): dictionary with metric names as keys and
            scoring functions as values. If a list is given, it should be of scoring
            functions, and their `__name__` value will be used as the metric name.
            If not given, all the available metrics will be used.
        rank (str, optional): Sort and rank the pipelines based on the given metric.
            If not given, rank using the first metric.
        output_path (str, optional):
            location to save the final results. If not given, use default location.
        with_gpu (boolean, optional):
            use pipeline with gpu config. If not given, use False.

    Returns:
        pandas.DataFrame: Table containing the average of the scores obtained with
            each scoring function accross all the signals for each pipeline, ranked
            by the indicated metric.
    """
    pipelines = pipelines or _get_pipelines(with_gpu=with_gpu)
    datasets = _get_data(datasets)
    hyperparameters = hyperparameters or BENCHMARK_HYPER

    results = list()
    for name, signals in datasets.items():
        hyper = _get_hyperparameter(hyperparameters, name)
        kwarg = kwargs[name] if name in kwargs.keys() else {}

        result = evaluate_pipelines(
            pipelines, signals, hyper, metrics, rank, kwarg)
        result['dataset'] = name
        results.append(result)

        if output_path:
            LOGGER.info('Saving benchmark report to %s', output_path)
            result.to_csv(os.path.join(output_path, name + '.csv'))

    results = pd.concat(results).drop(['rank', 'holdout'], axis=1)
    results = results.groupby('pipeline').mean().reset_index()

    return _sort_leaderboard(results, rank, metrics)


if __name__ == "__main__":
    leaderboard = run_benchmark(output_path='results/', **BENCHMARK_PARAMS)
    output_path = os.path.join(BENCHMARK_PATH, 'leaderboard.csv')
    leaderboard.to_csv(output_path)
    print(leaderboard)
