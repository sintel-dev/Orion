import ast
import csv
import os
import json
import logging
import warnings
from datetime import datetime

import pandas as pd

from orion.analysis import analyze
from orion.data import NASA_SIGNALS, load_anomalies, load_signal
from orion.metrics import METRICS

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)

BENCHMARK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'orion_benchmark'
)

with open('{}/benchmark_data.csv'.format(BENCHMARK_PATH), newline='') as f:
    reader = csv.reader(f)
    BENCHMARK_DATA = {row[0]:ast.literal_eval(row[1]) for row in reader}

BENCHMARK_HYPER = pd.read_csv('{}/benchmark_hyperparameters.csv'.format(
    BENCHMARK_PATH), index_col=0).to_dict()

def _get_hyperparameter(pipeline, signal):
    file = os.path.join(os.path.abspath(pipeline), '_' + signal + '.csv')
    print(file)
    if os.path.isfile(file):
        return file
    return None 

def _evaluate_on_signal(pipeline, hyperparameter, signal, metrics, holdout=True):
    if holdout:
        train = load_signal(signal + '-train')
        test = load_signal(signal + '-test')
    else:
        train = test = load_signal(signal)

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


def evaluate_pipeline(pipeline, hyperparameter=None, signals=NASA_SIGNALS, metrics=METRICS, holdout=None):
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
                score = _evaluate_on_signal(pipeline, hyperparameter, signal, metrics, holdout_)
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


def evaluate_pipelines(pipelines, hyperparameters=None, signals=None, metrics=None, rank=None, holdout=(True, False)):
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

    hyperparameters_ = hyperparameters.keys() if hyperparameters is not None else []
    for name, pipeline in pipelines.items():
        hyperparameter = None
        if name in hyperparameters_:
            hyperparameter = hyperparameters[name]

        LOGGER.info("Evaluating pipeline: %s", name)
        score = evaluate_pipeline(pipeline, hyperparameter, signals, metrics, holdout)
        score['pipeline'] = name
        score['hyperparameter'] = hyperparameter
        scores.append(score)

    scores = pd.concat(scores)

    rank = rank or list(metrics.keys())[0]
    scores.sort_values(rank, ascending=False, inplace=True)
    scores.reset_index(drop=True, inplace=True)
    scores.index.name = 'rank'
    scores.reset_index(drop=False, inplace=True)
    scores['rank'] += 1

    return scores.set_index('pipeline').reset_index()


def run_benchmark(pipelines, datasets=None, hyperparameters=None, metrics=None, rank=None, output_path=None):    
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
        hyperparameters (dict, optional):
            dictionary with (dataset name, pipeline name) as keys and hyperparameter 
            JSON settings path or dictionary as values. If not given, use default 
            hyperparameters.
        metrics (dict or list, optional): dictionary with metric names as keys and
            scoring functions as values. If a list is given, it should be of scoring
            functions, and their `__name__` value will be used as the metric name.
            If not given, all the available metrics will be used.
        rank (str, optional): Sort and rank the pipelines based on the given metric.
            If not given, rank using the first metric.
        output_path (str, optional):
            location to save the final results. If not given, use default location.

    Returns:
        pandas.DataFrame: Table containing the average of the scores obtained with
            each scoring function accross all the signals for each pipeline, ranked
            by the indicated metric.
    """
    datasets = datasets or BENCHMARK_DATA
    hyperparameters = hyperparameters or BENCHMARK_HYPER
    metrics = metrics or METRICS

    print(hyperparameters)

    results = list()
    for name, signals in datasets.items():
        if isinstance(hyperparameters, dict) and name in hyperparameters.keys():
            hyper =  hyperparameters[name]
        
        result = evaluate_pipelines(pipelines, hyper, signals, metrics, rank)
        results.append(result)

        if output_path:
            LOGGER.info('Saving benchmark report to %s', output_path)
            result.to_csv(output_path + name + '.csv')

    return pd.concat(results)


if __name__ == "__main__":
    pipelines = {
    "dummy": 'orion/pipelines/dummy.json', 
    "arima": 'orion/pipelines/arima.json',
    "lstm": 'orion/pipelines/lstm_dynamic_threshold.json'}

    print(pipelines)
    dataset = {
        "MSL": BENCHMARK_DATA["MSL"][:2],
        "SMAP": BENCHMARK_DATA["SMAP"][:2],
        "YAHOOA2": BENCHMARK_DATA["YAHOOA2"][:2],
        "realTweets": BENCHMARK_DATA["realTweets"][:2]
    }
    print(dataset)
    run_benchmark(pipelines, dataset, output_path='results/')
