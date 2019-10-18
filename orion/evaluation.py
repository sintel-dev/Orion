import logging
import warnings
from datetime import datetime

import pandas as pd

from orion import metrics
from orion.analysis import analyze
from orion.data import load_anomalies, load_signal

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


METRICS = {
    'accuracy': metrics.accuracy_score,
    'precision': metrics.precision_score,
    'recall': metrics.recall_score,
    'f1': metrics.f1_score,
}
NASA_SIGNALS = (
    'P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7',
    'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-3',
    'D-2', 'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2', 'D-5',
    'D-6', 'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9',
    'F-2', 'G-4', 'T-3', 'D-11', 'D-12', 'B-1', 'G-6', 'G-7', 'P-7',
    'R-1', 'A-5', 'A-6', 'A-7', 'D-13', 'A-8', 'A-9', 'F-3', 'M-6',
    'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4',
    'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14',
    'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8'
)


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
