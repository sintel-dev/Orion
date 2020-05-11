import logging
import warnings
from datetime import datetime

import pandas as pd

from orion.analysis import analyze
from orion.data import NASA_SIGNALS, load_anomalies, load_signal
from orion.metrics import METRICS

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


def _evaluate_on_signal(pipeline, signal, metrics, holdout=True):
    if holdout:
        train = load_signal(signal + '-train')
    else:
        train = load_signal(signal)

    test = load_signal(signal + '-test')
    start = datetime.utcnow()
    anomalies = analyze(pipeline, train, test)
    elapsed = datetime.utcnow() - start

    truth = load_anomalies(signal)

    scores = {
        name: scorer(truth, anomalies, test)
        for name, scorer in metrics.items()
    }
    scores['elapsed'] = elapsed.total_seconds()

    return scores


def evaluate_pipeline(pipeline, signals=NASA_SIGNALS, metrics=METRICS, holdout=None):
    """Evaluate a pipeline on multiple signals with multiple metrics.

    The pipeline is used to analyze the given signals and later on the
    detected anomalies are scored against the known anomalies using the
    indicated metrics.

    Args:
        pipeline (str): Path to the pipeline JSON.
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

    scores = list()
    for signal in signals:
        for holdout_ in holdout:
            try:
                LOGGER.info("Scoring pipeline %s on signal %s (Holdout: %s)",
                            pipeline, signal, holdout_)
                score = _evaluate_on_signal(pipeline, signal, metrics, holdout_)
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


def evaluate_pipelines(pipelines, signals=None, metrics=None, rank=None, holdout=(True, False)):
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

    if isinstance(pipelines, list):
        pipelines = dict(zip(pipelines, pipelines))

    for name, pipeline in pipelines.items():
        LOGGER.info("Evaluating pipeline: %s", name)
        score = evaluate_pipeline(pipeline, signals, metrics, holdout)
        score['pipeline'] = name
        scores.append(score)

    scores = pd.concat(scores)

    rank = rank or list(metrics.keys())[0]
    scores.sort_values(rank, ascending=False, inplace=True)
    scores.reset_index(drop=True, inplace=True)
    scores.index.name = 'rank'
    scores.reset_index(drop=False, inplace=True)
    scores['rank'] += 1

    return scores.set_index('pipeline').reset_index()
