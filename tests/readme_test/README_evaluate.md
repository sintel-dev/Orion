# Evaluation

This document explains the evaluation subpackage accompanied with Orion. It is used in order to evaluate how good a pipeline is at detecting anomalies.
In order to use this framework, we require two main arguments: known anomalies, and detected anomalies.

## Anomaly types

There are two approaches to defined anomalies:
- _Point anomalies_ which are identified by a single value in the time series.
- _Contextual anomalies_ which are identified by an anomalous interval, specifically the start/end timestamps.

```python3
# Example

point_anomaly = [1222819200, 1222828100, 1223881200]

contextual_anomaly = [(1222819200, 1392768000), 
                      (1392768000, 1398729600), 
                      (1398729600, 1399356000)]
```

We have created an evaluator for both types. 
We also provide a suite of transformation functions in `utils.py` to help with converting one type to another.


## Calculating a Score

Here we describe how we compute a score of how close a set of previously known anomalies and a set of detected anomalies are.

### Point Scoring

In point anomalies, we perform a point-wise comparison at each timestamp; this is done on a second (s) based frequency.

#### Scoring Input

The information that we have is:

* The time series start (min) and end (max) timestamps.
* A list of timestamps for the known anomalies.
* A list of timestamps for the detected anomalies.

An example of this would be:

* Timeseries start, end

```python3
data_span = (1222819200, 1222819205)
```

* Known anomalies:

```python3
ground_truth = [
    1222819200, 
    1222819201, 
    1222819202
]
```

* Detected anomalies:

```python3
anomalies = [
    1222819201, 
    1222819202, 
    1222819203
]
```

#### Scoring process: Reformat as labels

The solution implemented for point anomalies is to compute a list of labels, 1s and 0s, and then use the scikit-learn confusion matrix function as an intermediate to finding the accuracy, precision, recall, and f1 scores.

For this we generate a sequence of the same length as `data_span` and fill the corresponding anomalies within the correct placement.

Continuing on the previous example, we obtain the following:

```python3
truth = [1, 1, 1, 0, 0, 0]
detected = [0, 1, 1, 1, 0, 0]
```

This results with the following true negative (tn), false positive (fp), false negative (fn), true positive (tp):

```python3
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(truth, detected).ravel()
```

Since we have the result of the confusion matrix, we can now compute the accuracy, precision, recall, and f1 score to evaluate the performance of the model.

```python3
# accuracy score
tn + tp / (tn + fp + fn + tp) # 0.667
```

This entire process is implemented within the point metrics
```python3
from orion.evaluation.point import point_accuracy, point_f1_score 

start, end = data_span

point_accuracy(ground_truth, anomalies, start=start, end=end) # 0.667
point_f1_score(ground_truth, anomalies, start=start, end=end) # 0.667
```

### Contextual Scoring

In contextual anomalies, we can compare the detected anomalies to the ground truth in two approaches: weighted segment, and overlap segment.

#### Scoring Input

The information that we have is:

* The time series start (min) and end (max) timestamps.
* A list of start/stop pairs of timestamps for the known anomalies.
* A list of start/stop pairs of timestamps for the detected anomalies.

An example of this would be:

* Timeseries start, end

```python3
data_span = (1222819200, 1442016000)
```

* Known anomalies (in this case only one):

```python3
ground_truth = [
    (1392768000, 1402423200)
]
```

* Detected anomalies (in this case only one):

```python3
anomalies = [
    (1398729600, 1399356000)
]
```

#### Scoring process: Reformat as labels with weights (weighted segment)

The solution implemented in Orion has been to use all the previous information to compute a list of labels, 1s and 0s, and then use the scikit-learn confusion matrix function passing a weights array as an intermediate to finding the accuracy, precision, recall, and f1 scores.

Continuing on the previous example, we do the following:

1. Make a sorted set of all the timestamps and compute consecutive intervals:

```python3
intervals = [
    (1222819200, 1392768000),
    (1392768000, 1398729600),
    (1398729600, 1399356001),
    (1399356001, 1402423201),
    (1402423201, 1442016000)
]
```

2. For both the known and detected anomalies sequences, compute a label for each interval using 1 if the interval intersects with one of the anomaly intervals in the sequence:

```python3
truth = [0, 1, 1, 1, 0]
detected = [0, 0, 1, 0, 0]
```

3. Compute a vector of weights using the lengths of the intervals:

```python3
weights = [169948800, 5961600, 626401, 3067200, 39592799]
```

4. Compute the confusion matrix using labels and weights:

```python3
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(
  truth, detected, sample_weight=weights, labels=[0, 1]).ravel()
```

5. Compute a score:

```python3
# accuracy score
tn + tp / (tn + fp + fn + tp) # 0.959
```

This entire process is implemented within the contextual metrics
```python3
from orion.evaluation import contextual_accuracy, contextual_f1_score


start, end = data_span

contextual_accuracy(ground_truth, anomalies, start=start, end=end) # 0.959
contextual_f1_score(ground_truth, anomalies, start=start, end=end) # 0.122
```

#### Scoring process: Look for overlap between anomalies (overlap segment)

In this methodology, we are more concerned with whether or not we were able to find an anomaly; even just a part of it. It records:

* a true positive if a known anomalous window overlaps any detected windows.
* a false negative if a known anomalous window does not overlap any detected windows.
* a false positive if a detected window does not overlap any known anomalous region.

To use this objective, we pass ``weighted=False`` in the metric method of choice.

```python3
start, end = data_span

contextual_f1_score(ground_truth, anomalies, start=start, end=end, weighted=False) # 1.0
```
