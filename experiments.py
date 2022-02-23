import os
from functools import partial

from orion.benchmark import benchmark, BENCHMARK_DATA, METRICS
from orion.evaluation import contextual_confusion_matrix
from orion.evaluation.contextual import record_observed, record_expected


def main(workers=1):
    NOTEBOOKS_DIRECTORY = os.path.join(os.getcwd(), 'notebooks')

    pipelines = {
        # 'arima': 'arima',
        # 'lstm_dynamic_threshold': 'lstm_dynamic_threshold_gpu',
        # 'azure': 'azure',
        # 'tadgan': 'tadgan_gpu',
        # 'tadgan': 'tadgan',
        'tadgan': 'tadgan_encoder_downsample'
    }

    # metrics
    del METRICS['accuracy']
    METRICS['confusion_matrix'] = contextual_confusion_matrix
    METRICS['observed'] = record_observed
    METRICS['expected'] = record_expected
    metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

    NAB = ['artificialWithAnomaly', 'realAdExchange', 'realAWSCloudwatch', 'realTraffic', 'realTweets']
    YAHOO = ['YAHOOA1', 'YAHOOA2', 'YAHOOA3', 'YAHOOA4']
    NASA = ['MSL', 'SMAP']
    # MULTIVARIATE_NASA = ['MULTIVARIATE_SMAP', 'MULTIVARIATE_MSL']
    # datasets = ['artificialWithAnomaly']
    datasets = NAB + YAHOO + NASA
    datasets = {key: BENCHMARK_DATA[key] for key in datasets}

    EXPERIMENT_NAME = 'tadgan_tensorflow_2.0_encoder_downsample'
    scores = benchmark(
        pipelines=pipelines,
        datasets=datasets,
        metrics=metrics,
        rank='f1',
        show_progress=True,
        workers=workers,
        cache_dir=os.path.join(NOTEBOOKS_DIRECTORY, EXPERIMENT_NAME, 'cache'),
        # pipeline_dir=os.path.join(NOTEBOOKS_DIRECTORY, EXPERIMENT_NAME, 'pipeline'),
    )


if __name__ == '__main__':
    main()
