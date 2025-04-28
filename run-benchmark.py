from orion.benchmark import BENCHMARK_DATA, main

datasets = ['MSL', 'SMAP',
            'YAHOOA1', 'YAHOOA1', 'YAHOOA1', 'YAHOOA1',
            'artificialWithAnomaly', 'realAWSCloudwatch', 'realAdExchange',
            'realTraffic', 'realTweets']

pipelines = ['arima', 'tadgan', 'aer',
             'lstm_dynamic_threshold', 'lstm_autoencoder', 'dense_autoencoder',
             'vae', 'anomaly_transformer', 'timesfm', 'units']

if __name__ == '__main__':
    if any([dataset in BENCHMARK_DATA for dataset in datasets]):
        datasets = dict((dataset, BENCHMARK_DATA[dataset]) for dataset in datasets)

    pipelines = [p + '_viz' for p in pipelines]

    anomaly_directory = 'anomalies'
    main(pipelines=pipelines,
         datasets=['S-1'],
         resume=False,
         workers=1,
         output_path='results.csv',
         cache_dir='cache',
         pipeline_dir=None,
         anomaly_dir=anomaly_directory
    )