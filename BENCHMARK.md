# Benchmark

This document explains the benchmarking procedure we develop in Orion in order to evaluate how good
a pipeline is detecting anomalies.

## Releases
In every release, we run Orion benchmark and maintain an upto-date [leaderboard](README.md#leaderboard).
Results obtained during the benchmarking process as well as previous benchmarks can be found 
within [benchmark/results](benchmark/results) folder as CSV files. 

## Evaluating the Pipelines

Using the [Evaluation sub-package](orion/evaluation), we can compute a score given a set of known 
anomalies and another one of detected anomalies. 
The entire process can be summarized in the following diagram:

![Scoring](docs/images/scoring-300.png?raw=true "Scoring")

We can then evaluate the overall performance of 
our pipelines in order to know which one has the best performance.

For this we:

1. Use each pipeline to detect anomalies on all datasets and their signals.
2. Retreive the list of known anomalies for each of these signals.
3. Compute the scores for each signal using multiple metrics (e.g. accuraccy and f1).
4. Average the score obtained for each metric and pipeline accross all the signals.
5. Finally, we rank our pipelines sorting them by one of the computed scores.

The output of this process is the [leaderboard](README.md#leaderboard).

## Benchmark function

For the scoring, we will be using the demo signals and a list of their known anomalies, obtained from
the [telemanom repository](https://github.com/khundman/telemanom/blob/master/labeled_anomalies.csv)
and computing how similar the anomalies detected by the pipeline are with these prevoiusly known
anomalies.

The complete evaluation process described above is directly available using the
`orion.benchmark.benchmark` function.

This function expects the following inputs:

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
 functions, and they `__name__` value will be used as the metric name.
 If not given, all the available metrics will be used.
* rank (str): Sort and rank the pipelines based on the given metric.
 If not given, rank using the first metric.
* distributed (bool): Whether to use dask for distributed computing. If not given,
 use `False`.
* holdout (bool): Whether to use the prespecified train-test split. If not given,
 use `False`.
* detrend (bool): Whether to use `scipy.detrend`. If not given, use `False`.
* output_path (str): Location to save the results. If not given, results will not be saved.

And returns a `pandas.DataFrame` which contains the scores obtained with each scoring function 
accross all the signals for each pipeline, optionally, you can follow that output with a 
`summarize_results` to average the scores and produce the leaderboard.

This is an example of how to call this function:

```
In [1]: from orion.benchmark import benchmark

In [2]: pipelines = [
   ...:     'arima',
   ...:     'lstm_dynamic_threshold'
   ...: ]

In [3]: metrics = ['f1', 'accuracy', 'recall', 'precision']

In [4]: signals = ['S-1', 'P-1']

In [5]: scores = benchmark(pipelines=pipelines, datasets=signals, metrics=metrics, rank='f1')

In [6]: scores
Out[6]:
                 pipeline  rank  accuracy  elapsed     f1         precision   recall   
0  lstm_dynamic_threshold     1  0.986993  915.958132  0.846868   0.879518     0.816555
1                   arima     2  0.962160  319.968949  0.382637   0.680000     0.266219 
```
