import os
import sys
from functools import partial

from orion.benchmark import benchmark, BENCHMARK_DATA, METRICS
from orion.evaluation import contextual_confusion_matrix
from orion.evaluation.contextual import record_observed, record_expected

# Datasets
NAB = ['artificialWithAnomaly', 'realAdExchange', 'realAWSCloudwatch', 'realTraffic', 'realTweets']
NASA = ['MSL', 'SMAP']
YAHOO = ['YAHOOA1', 'YAHOOA2', 'YAHOOA3', 'YAHOOA4']
UNIVARIATE_DATASETS = NAB + NASA + YAHOO

MULTIVARIATE_NASA = ['MULTIVARIATE_SMAP', 'MULTIVARIATE_MSL']
SWAT = ['SWaT']
WADI = ['WADI']
SMD = ['SMD']
MULTIVARIATE_DATASETS = MULTIVARIATE_NASA + SMD + SWAT + WADI

RESULTS_DIRECTORY = os.path.join(os.getcwd(), 'results')

# Additional Metrics
del METRICS['accuracy']
METRICS['confusion_matrix'] = contextual_confusion_matrix
METRICS['observed'] = record_observed
METRICS['expected'] = record_expected
METRICS = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}


def run_experiment(experiment_name: str, pipelines: dict, datasets: list, metrics: dict,
                   results_directory: str = RESULTS_DIRECTORY, workers: int = 1):
    datasets = {key: BENCHMARK_DATA[key] for key in datasets}
    scores = benchmark(
        pipelines=pipelines,
        datasets=datasets,
        metrics=metrics,
        rank='f1',
        show_progress=True,
        workers=workers,
        cache_dir=os.path.join(results_directory, experiment_name, 'cache'),
        pipeline_dir=os.path.join(results_directory, experiment_name, 'pipeline'),
    )
    return scores


if __name__ == "__main__":
    experiment_name, datasets, pipelines = sys.argv[1:4]
    if datasets == 'univariate_datasets':
        datasets = UNIVARIATE_DATASETS
    elif datasets == 'multivariate_datasets':
        datasets = MULTIVARIATE_DATASETS
    else:
        raise Exception('Unknown dataset.')

    try:
        pipelines = eval(pipelines)
    except Exception:
        raise

    results = run_experiment(
        experiment_name=experiment_name,
        pipelines=pipelines,
        datasets=datasets,
        metrics=METRICS,
        results_directory=RESULTS_DIRECTORY,
        workers=1
    )
