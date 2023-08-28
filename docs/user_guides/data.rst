.. _data:

====
Data
====

Orion takes a time series signal and produces an interval of expected anomalies. The input to the framework is a univariate time series and the output is a table denoting the start and end timestamp of the anomalies.

Data Format
-----------

All Orion pipelines follow the same format in terms of what it is expecting as input (a signal represented in a ``pandas`` dataframe) and output (a ``list`` of detected anomalous intervals that later is rendered by Orion into a dataframe). 

Input
~~~~~

Orion Pipelines work on univariate or multivariate time series that are provided as a single table of telemetry observations with at least two columns:

* ``timestamp``: an ``int`` column with the time of the observation in Unix Time format.
* ``value``: an ``int`` or ``float`` column with the observed value at the indicated timestamp. Each value is represented as a different column.

Here is an example of signal:

+------------+-----------+------------+-----------+
|  timestamp |   value 0 |    value 1 |   value 2 |
+------------+-----------+------------+-----------+
| 1222819200 | -0.366358 |          0 |         1 |
+------------+-----------+------------+-----------+
| 1222840800 | -0.394107 |          0 |         0 |
+------------+-----------+------------+-----------+
| 1222862400 |  0.403624 |          1 |         0 |
+------------+-----------+------------+-----------+
| 1222884000 | -0.362759 |          0 |         0 |
+------------+-----------+------------+-----------+
| 1222905600 | -0.370746 |          0 |         0 |
+------------+-----------+------------+-----------+

The table above contains an observation column value 0, and two status values 1 and 2.

Output
~~~~~~

There can be many steps (primitives) in a pipeline, however the end result must be a list of intervals. Each interval contains at least two entries: 

* ``start``: timestamp where the anomalous interval starts
* ``end``: timestamp where the anomalous interval ends

Optionally, there can be a third entry that contains a proxy to the likelihood that the interval is anomalous.

.. code-block:: python

    [
        (1222840800, 1222884000, 0.12572),
        (1402012800, 1403870400, 0.64339),
    ]


For example, the list above shows that the pipeline detected two intervals as anomalous, the second one being more likely than the first.

Then Orion will present this result as a ``pandas`` dataframe:

+------------+------------+----------+
|      start |        end | severity |
+------------+------------+----------+
| 1222840800 | 1222884000 |  0.12572 |
+------------+------------+----------+
| 1402012800 | 1403870400 |  0.64339 |
+------------+------------+----------+


Dataset we use in this library
------------------------------

For development, evaluation of pipelines, we include a dataset which includes several signals already formatted as expected by the Orion Pipelines.

This formatted dataset can be browsed and downloaded directly from the `sintel-orion AWS S3 Bucket`_.

This dataset is adapted from the one used for the experiments in the `Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding paper`_. Original source data is available for download `here`_.

We thank NASA for making this data available for public use.

.. _sintel-orion AWS S3 Bucket: https://sintel-orion.s3.amazonaws.com/index.html
.. _Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding paper: https://arxiv.org/abs/1802.04431
.. _here: https://s3-us-west-2.amazonaws.com/telemanom/data.zip

