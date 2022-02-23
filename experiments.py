import os
from functools import partial

from orion.benchmark import benchmark, BENCHMARK_DATA, METRICS
from orion.evaluation import contextual_confusion_matrix
from orion.evaluation.contextual import record_observed, record_expected

NOTEBOOKS_DIRECTORY = os.path.join(os.getcwd(), 'notebooks')

pipelines = [
    # 'arima'
    'tadgan',
    # 'tadgan_tensorflow',
]

# metrics
del METRICS['accuracy']
METRICS['confusion_matrix'] = contextual_confusion_matrix
METRICS['observed'] = record_observed
METRICS['expected'] = record_expected
metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

# datasets = ['artificialWithAnomaly', 'realAdExchange', 'realAWSCloudwatch', 'realTraffic', 'realTweets', 'MSL', 'SMAP']
# datasets = ['YAHOOA1', 'YAHOOA2', 'YAHOOA3', 'YAHOOA4']
# datasets = ['MULTIVARIATE_SMAP', 'MULTIVARIATE_MSL']
datasets = ['SMAP']
datasets = {key: BENCHMARK_DATA[key] for key in datasets}

EXPERIMENT_NAME = 'tadgan_tensorflow_2.0'

scores = benchmark(
    pipelines=pipelines,
    datasets=datasets,
    metrics=metrics,
    rank='f1',
    show_progress=True,
    workers=1,
    cache_dir=os.path.join(NOTEBOOKS_DIRECTORY, EXPERIMENT_NAME, 'cache'),
    # pipeline_dir=os.path.join(NOTEBOOKS_DIRECTORY, EXPERIMENT_NAME, 'pipeline'),
)
