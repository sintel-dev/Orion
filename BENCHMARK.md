# Benchmark

This document explains the scoring system being used in Orion in order to evaluate how good
a pipeline is detecting anomalies.

## Data used for the scoring

For the scoring, we will be using the demo signals and a list of their known anomalies, obtained from
the [telemanom repository](https://github.com/khundman/telemanom/blob/master/labeled_anomalies.csv)
and computing how similar the anomalies detected by the pipeline are with these prevoiusly known
anomalies.

## Calculating a Score

Here we describe how we compute a score about how similar a set of previously known anomalies
and a set of detected anomalies are.

### Scoring Input

The information that we have is:

* The time series start (min) and end (max) timestamps.
* A list of start/stop pairs of timestamps for the known anomalies.
* A list of start/stop pairs of timestamps for the detected anomalies.

An example of this would be:

* Timeseries start, end

```
data_span = (1222819200, 1442016000)
```

* Known anomalies (in this case only one):

```
ground_truth = [
    (1392768000, 1402423200)
]
```

* Detected anomalies (in this case only one):

```
anomalies = [
    (1398729600, 1399356000)
]
```

### Scoring process: Reformat as labels with weights

The solution implemented in Orion has been to use all the previous information to compute a
list of labels, 1s and 0s, and then use the scikit-learn metrics passing a weights array.

For this we do the following:

1. Make a sorted set of all the timestamps and compute consecutive intervals:

```
intervals = [
    (1222819200, 1392768000),
    (1392768000, 1398729600),
    (1398729600, 1399356000)
    (1399356000, 1402423200),
    (1402423200, 1442016000)
]
```

2. For both the known and detected anomalies sequences, compute a label for each interval
using 1 if the interval intersects with one of the anomaly intervals in the sequence:

```
truth = [0, 1, 1, 1, 0]
detected = [0, 0, 1, 0, 0]
```

3. Compute a vector of weights using the lengths of the intervals:

```
weights = [169948800, 5961600, 626400, 3067200, 39592800]
```

4. Compute a score using the labels:

```
accuracy_score(truth, detected, sample_weight=weights) = 0.9588096176586519
f1_score(truth, detected, sample_weight=weights) = 0.1218487394957983
```

The whole process can be summarized in this diagram:

![Scoring](docs/images/Scoring.png?raw=true "Scoring")

## Evaluating the Pipelines

Once we know how to compute a score given a set of known anomalies and another one of detected
anomalies, we can evaluate the overall performance of our pipelines in order to know which
one has the best performance.

For this we:

1. Use each pipeline to detect anomalies on all our demo signals.
2. Retreive the list of known anomalies for each of these signals.
3. Compute the scores for each signal using multiple metrics (e.g. accuraccy and f1).
4. Average the score obtained for each metric and pipeline accross all the signals.
5. Finally, we rank our pipelines sorting them by one of the computed scores.

The output of this process is the [Leaderboard table from the README.md](README.md#leaderboard).

## Evaluate function

The complete evaluation process described above is directly available using the
`orion.evaluation.evaluate_pipelines` function.

This function expects the following inputs:

* `pipelines (dict or list)`: dictionary with pipeline names as keys and their
JSON paths as values. If a list is given, it should be of JSON paths,
and the paths themselves will be used as names.
* `signals (list, optional)`: list of signals. If not given, all the NASA signals are used.
* `metrics (dict or list, optional)`: dictionary with metric names as keys and
scoring functions as values. If a list is given, it should be of scoring
functions, and they `__name__` value will be used as the metric name.
If not given, all the available metrics will be used.
* `rank (str, optional)`: Sort and rank the pipelines based on the given metric.
If not given, rank using the first metric.

And returns a `pandas.DataFrame` which contains the average of the scores obtained with
each scoring function accross all the signals for each pipeline, ranked by the indicated metric.

This is an example of how to call this function:

```
In [1]: from orion.evaluation import evaluate_pipelines

In [2]: pipelines = [
   ...:     'orion/pipelines/skew_24h_lstm.json',
   ...:     'orion/pipelines/lstm_dynamic_threshold.json'
   ...: ]

In [3]: metrics = ['f1', 'accuracy', 'recall', 'precision']

In [4]: signals = ['S-1', 'P-1']

In [5]: scores = evaluate_pipelines(pipelines, signals, metrics, rank='f1')

In [6]: scores
Out[6]:
                                      pipeline  rank  accuracy        f1  precision    recall
0  orion/pipelines/lstm_dynamic_threshold.json     1  0.946529  0.060924     0.5000  0.032438
1           orion/pipelines/skew_24h_lstm.json     2  0.939685  0.015370     0.0625  0.008762
```
