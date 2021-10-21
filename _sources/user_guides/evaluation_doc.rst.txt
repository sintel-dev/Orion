.. _evaluation_doc:

==========
Evaluation
==========

When evaluating a pipeline, we rely on two main arguments: *known anomalies*, and *detected anomalies*.

Types of anomalies
------------------

There are two approaches to defined anomalies:

* *point anomalies* which are identified by a single value in the time series.
* *contextual anomalies* which are identified by an anomalous interval, specifically the start/end timestamps.

For example:

.. code-block:: python

    # defined by a single timestamp
    point_anomaly = [1222819200, 1222828100, 1223881200]

    # defined by an interval
    contextual_anomaly = [(1222819200, 1392768000), 
                          (1392768000, 1398729600), 
                          (1398729600, 1399356000)]


We have created an evaluator for both types. 
We also provide a suite of transformation functions in ``utils.py`` to help with converting one type to another.

View the `Evaluation sub-package <https://github.com/sintel-dev/Orion/tree/master/orion/evaluation>`__ to see the metrics provided in **Orion**.

How do we score anomalies?
--------------------------
We use two main approaches to compare detected anomalies with the ground truth:

1. Assessing every segment in the detected anomalies with its counterpart in the ground truth, we refer to this approach as a **weighted segment**.
2. Assess the detected anomaly segment by seeing if we caught an overlap with the correct anomalies, we refer to this approach as an **overlapping segment**.

Let us use the following example to walk through the differences between both approaches:

The information that we have is:

* The time series start (min) and end (max) timestamps.
* A list of start/stop pairs of timestamps for the *known anomalies*.
* A list of start/stop pairs of timestamps for the *detected anomalies*.

An example of this would be:

* timeseries start, end timestamps

.. ipython:: python
    :okwarning:

    data_span = (1222819200, 1442016000)

* known anomalies (in this case only one)

.. ipython:: python
    :okwarning:

    ground_truth = [
        (1392768000, 1402423200)
    ]

* detected anomalies (in this case only one)

.. ipython:: python
    :okwarning:

    anomalies = [
        (1398729600, 1399356000)
    ]

So, what is the score?

Weighted segment
~~~~~~~~~~~~~~~~

Weighted segment based evaluation is a strict approach which weighs each segment by its actual time duration. It is valuable to use when you want to detect the exact segment of the anomaly, without any slackness. It first segments the signal into partitions based on the ground truth and detected sequences. Then it makes a segment to segment comparison and records true positive, true negative, false positive, and false negative accordingly. The overall score is weighted by the duration of each segment.

.. ipython:: python
    :okwarning:

    from orion.evaluation.contextual import contextual_f1_score

    start, end = data_span

    contextual_f1_score(ground_truth, anomalies, start=start, end=end, weighted=True)

Overlapping segment
~~~~~~~~~~~~~~~~~~~

We look for overlap between detected anomalies and ground truth anomalies.

In this methodology, we are more concerned with whether or not we were able to find an anomaly; even just a part of it. It records:

* a *true positive* if a known anomalous window overlaps any detected windows.
* a *false negative* if a known anomalous window does not overlap any detected windows.
* a *false positive* if a detected window does not overlap any known anomalous region.

To use this objective, we pass ``weighted=False`` in the metric method of choice.

.. ipython:: python
    :okwarning:

    contextual_f1_score(ground_truth, anomalies, start=start, end=end, weighted=False)

You can read more about our step-by-step process in our evaluation by visiting the `Evaluation sub-package <https://github.com/sintel-dev/Orion/tree/master/orion/evaluation>`__


Evaluate the performance of your pipeline
-----------------------------------------
We can use the same dataset we saw in the :ref:`quickstart`

.. ipython:: python
    :okwarning:

    from orion.data import load_signal
    data = load_signal('S-1')

We set up the pipeline (``lstm_dynamic_threshold``) as well as some hyperparameters.

.. ipython:: python
    :okwarning:

    from orion import Orion

    hyperparameters = {
        'keras.Sequential.LSTMTimeSeriesRegressor#1': {
            'epochs': 5
        }
    }

    orion = Orion(
        pipeline='lstm_dynamic_threshold',
        hyperparameters=hyperparameters
    )

In this next step we will load some already known anomalous intervals and evaluate how good our anomaly detection was by comparing those with our detected intervals.

For this, we will first load the known anomalies for the signal that we are using:

.. ipython:: python
    :okwarning:

    from orion.data import load_anomalies

    ground_truth = load_anomalies('S-1')
    ground_truth

The output will be a table in the same format as the ``anomalies`` one.

Afterwards, we can call the `orion.evaluate` method, passing both the data and the ground truth:

.. ipython:: python
    :okwarning:

    scores = orion.evaluate(data, ground_truth, fit=True)

.. note::

    since the pipeline has not been trained yet, we set ``fit=True`` to fit it first before detecting anomalies.

The output will be a ``pandas.Series`` containing a collection of scores indicating how the predictions were:

.. ipython:: python
    :okwarning:

    scores
