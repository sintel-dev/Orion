# Benchmark

This document explains the benchmarking procedure we develop in Orion in order to evaluate how good a pipeline is detecting anomalies.

## Evaluating the Pipelines

Using the [Evaluation sub-package](orion/evaluation), we can compute a score given a set of known anomalies and another one of detected anomalies. 

We can evaluate the overall performance of 
our pipelines in order to know which one has the best performance.

For this we:

1. Use each pipeline to detect anomalies on all datasets and their signals.
2. Retreive the list of known anomalies for each of these signals.
3. Compute the scores for each signal using multiple metrics (e.g. accuraccy and f1).
4. Average the score obtained for each metric and pipeline accross all the signals.
5. Finally, we rank our pipelines sorting them by one of the computed scores.

## Benchmark Function

### Pipelines
The first item required to benchmark is the set of pipelines you want to evaluate, this can be done by supplying a list of pipelines or a dictionary of pipelines.

```python3
pipelines = [ 
    'arima',
    'lstm_dynamic_threshold'
]
```
You can use pipelines from the suite of existing pipelines in Orion or, alternatively, you can use your own implemented pipeline. 

### Datasets
An important item in benchmarking is the requirement to know the ground truth anomalies for proper comparison. 
An example of such dataset is the demo signals and a list of their known anomalies, obtained from
the [telemanom repository](https://github.com/khundman/telemanom/blob/master/labeled_anomalies.csv).
Using these signals, we can compute how similar the anomalies detected by the pipeline are with the prevoiusly known
anomalies.

Similarly, we define the datasets as either a single dataset compose of a list of signals, or a dictionary of datasets.
```python3
signals = [
    'S-1',
    'P-1'
]

# or defined by a dataset name
datasets = {
    'demo': signals
}
```

### Hyperparameters
In most cases, pipelines need to be modified based on the dataset currently being evaluated. To provide this functionality, you can use hyperparameters, which are nested dictionaries, to change a particular hyperparameter setting within the pipeline.
For example, to consider changing the number of ``epochs``  within ``lstm_dynamic_threshold`` pipeline for the ``demo`` dataset, we use:
```python3
hyperparameters = {
    'demo': {
        'lstm_dynamic_threshold': {
            'keras.Sequential.LSTMTimeSeriesRegressor#1': {
                'epochs': 5
            }
        }
    }
}
```

### Arguments
To use ``orion.benchmark.benchmark`` function, it expects the following inputs:

* pipelines (dict or list): dictionary with pipeline names as keys and their
 JSON paths as values. If a list is given, it should be of JSON paths,
 and the paths themselves will be used as names. If not given, all verified
 pipelines will be used for evaluation.
* datasets (dict or list): dictionary of dataset name as keys and list of signals as 
 values. If a list is given then it will be under a generic name `dataset`.
 If not given, all benchmark datasets will be used used.
* hyperparameters (dict or list): dictionary with pipeline names as keys
 and their hyperparameter JSON paths or dictionaries as values. If a list is
 given, it should be of corresponding order to pipelines.
* metrics (dict or list): dictionary with metric names as keys and
 scoring functions as values. If a list is given, it should be of scoring
 functions, and they ``__name__`` value will be used as the metric name.
 If not given, all the available metrics will be used.
* rank (str): Sort and rank the pipelines based on the given metric.
 If not given, rank using the first metric.
* test_split (bool or float): if bool, it indicates whether to use the prespecified train-test split. If float, then it 
 should be between 0.0 and 1.0 and represent the proportion of the signal to include in the test split. 
 If not given, use ``False``.
* detrend (bool): Whether to use ``scipy.detrend``. If not given, use ``False``.
* iterations (int): Number of iterations to perform over each signal and pipeline. Defaults to 1.
* workers (int or str): If ``workers`` is given as an integer value other than 0 or 1, a multiprocessing 
 Pool is used to distribute the computation across the indicated number of workers.
 If the string ``dask`` is given, the computation is distributed using ``dask``.
 In this case, setting up the ``dask`` cluster and client is expected to be handled outside of this function.
* show_progress (bool): Whether to use tqdm to keep track of the progress. Defaults to ``True``.
* cache_dir (str): If a ``cache_dir`` is given, intermediate results are stored in the indicated directory
 as CSV files as they get computted. This allows inspecting results while the benchmark
 is still running and also recovering results in case the process does not finish
 properly. Defaults to ``None``.
* output_path (str): Location to save the intermediatry results. If not given,
 intermediatry results will not be saved.
* pipeline_dir (str): If a ``pipeline_dir`` is given, pipelines will get dumped in the specificed directory as pickle files.
 Defaults to ``None``.

And returns a ``pandas.DataFrame`` which contains the scores obtained with each scoring function accross all the signals for each pipeline.

This is an example of how to call this function:

```python3
from orion.benchmark import benchmark

pipelines = [
    'arima',
    'lstm_dynamic_threshold'
]

signals = ['S-1', 'P-1']

metrics = ['f1', 'accuracy', 'recall', 'precision']

scores = benchmark(pipelines=pipelines, datasets=signals, metrics=metrics, rank='f1')

```
> :warning: Benchmarking might require a long time to compute.

The output of the benchmark will be a ``pandas.DataFrame`` containing the results obtained by each pipeline on evaluating each signal.
```
                 pipeline  rank  accuracy  elapsed     f1         precision   recall   
0  lstm_dynamic_threshold     1  0.986993  915.958132  0.846868   0.879518     0.816555
1                   arima     2  0.962160  319.968949  0.382637   0.680000     0.266219 
```

## Releases
In every release, we run Orion benchmark. We maintain an up-to-date leaderboard with the current scoring of the verified pipelines according to the benchmarking procedure explained above.

We run the benchmark on **11** datasets with their known grounth truth. We record the score of the pipelines on each dataset. 
Results obtained during the benchmarking process as well as previous benchmarks can be found 
within [benchmark/results](benchmark/results) folder as CSV files. In addition, you can find it in the [details Google Sheets document](https://docs.google.com/spreadsheets/d/1HaYDjY-BEXEObbi65fwG0om5d8kbRarhpK4mvOZVmqU/edit?usp=sharing).

### Leaderboard
We summarize the results in the [leaderboard](benchmark/leaderboard.md) table. We showcase the number of wins each pipeline has over the ARIMA pipeline.

The summarized results can also be browsed in the following [summary Google Sheets document](https://docs.google.com/spreadsheets/d/1ZPUwYH8LhDovVeuJhKYGXYny7472HXVCzhX6D6PObmg/edit?usp=sharing).
