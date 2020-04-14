import logging
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import signal as scipy_signal

from orion import metrics
from orion.metrics import score_overlap
from orion.analysis import analyze3
from orion.data import load_anomalies, load_signal

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


METRICS = {
    'accuracy': metrics.accuracy_score,
    'precision': metrics.precision_score,
    'recall': metrics.recall_score,
    'f1': metrics.f1_score,
}

def _evaluate_on_signal(pipeline, signal, metrics, holdout=True, split=None, detrend=False):
    if holdout:
        train = load_signal(signal + '-train')
        test = load_signal(signal + '-test')
    else:
        if split:
            train, test = load_signal(signal, test_size=split)
            if split == 1:
                train = test
        else:
            train = test = load_signal(signal)

    if detrend:
        train['value'] = scipy_signal.detrend(train['value'])
        test['value'] = scipy_signal.detrend(test['value'])

    truth = load_anomalies(signal)

    anomalies_set = analyze3(pipeline, train, test, truth)

    truth = load_anomalies(signal)

    scores_set = list()
    tps = list()
    fps = list()
    fns = list()
    for anomalies in anomalies_set:
        scores = {
            name: scorer(truth, anomalies, test)
            for name, scorer in metrics.items()
        }
        scores_set.append(scores)
        tp, fp, fn = score_overlap(truth, anomalies)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

    return scores_set, tps, fps, fns

def evaluate_pipeline(pipeline, signals, metrics=METRICS, holdout=None, split=None, detrend=False):
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

    comb_num = 10
        
    scores = list()
    tp_sums, fp_sums, fn_sums = [0]*comb_num, [0]*comb_num, [0]*comb_num
    signal_num = len(signals)
    for idx, signal in enumerate(signals):
        print('{}/{} {} using {}'.format(idx+1, signal_num, signal, pipeline))
        for holdout_ in holdout:
            try:
                LOGGER.info("Scoring pipeline %s on signal %s (Holdout: %s)",
                            pipeline, signal, holdout_)
                score_set, tps, fps, fns = _evaluate_on_signal(pipeline, signal, metrics, holdout_, split, detrend)
            except Exception:
                LOGGER.exception("Exception scoring pipeline %s on signal %s (Holdout: %s)",
                                 pipeline, signal, holdout_)
                score_set, tps, fps, fns = list(), [0]*comb_num, [0]*comb_num, [0]*comb_num
                score = {name: 0 for name in metrics.keys()}
                for ni in range(comb_num):  # comb_num combinations
                    score_set.append(score)

            scores.append(score_set)
            
            for ni in range(comb_num):
                tp_sums[ni] += tps[ni]
                fp_sums[ni] += fps[ni]
                fn_sums[ni] += fns[ni]
    
    final_scores = []
    for ni in range(comb_num):
        final_score = dict()
        for name in metrics.keys():
            ele = list()
            for nj in range(signal_num):
                ele.append(scores[nj][ni][name])
            final_score[name] = np.array(ele).mean()
        final_scores.append(pd.Series(final_score))
    
    return final_scores, tp_sums, fp_sums, fn_sums


def evaluate_pipelines(pipelines, signals=None, metrics=None, rank=None, holdout=(True, False), split=None, detrend=False):
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
    signals = signals
    metrics = metrics or METRICS

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

    results = list()
    for name, pipeline in pipelines.items():
        LOGGER.info("Evaluating pipeline: %s", name)
        scores, tps, fps, fns = evaluate_pipeline(pipeline, signals, metrics, holdout, split, detrend)
        for i in range(len(scores)):
            scores[i]['pipeline'] = name 
            scores[i]['tp'] = tps[i] 
            scores[i]['fp'] = fps[i]
            scores[i]['fn'] = fns[i]
        results.append(scores)

    return results

