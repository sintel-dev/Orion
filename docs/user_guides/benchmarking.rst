.. _benchmarking:

============
Benchmarking
============

We provide a benchmarking framework to enable users to compare multiple pipelines against each other. The evaluation metrics are documented within pipeline evaluation, please visit :ref:`evaluation_doc` to read more about it.

Releases
--------

In every release, we run Orion benchmark. We maintain an up-to-date leaderboard with the current scoring of the verified pipelines according to the benchmarking procedure.

Results obtained during benchmarking as well as previous releases can be found within `benchmark/results`_ folder as CSV files. Summarized results can also be browsed in the following `summary Google Sheets document`_ as well as the `details Google Sheets document`_.


Leaderboard
~~~~~~~~~~~

We run the benchmark on **12** datasets with their known grounth truth. We record the score of the pipelines on each datasets. To compute the leaderboard table, we showcase the number of wins each pipeline has over the ARIMA pipeline. 

.. mdinclude:: ../../benchmark/leaderboard.md

To view a list of all available pipelines, visit :ref:`pipelines` page.

Process
-------

We evaluate the performance of pipelines by following a series of executions. From a high level, we can view the process as:

1. Use each pipeline to detect anomalies on all datasets and their signals.
2. Retrieve the list of known anomalies for each of these signals.
3. Compute the scores for each signal using multiple metrics (e.g. accuracy and f1).
4. Average the score obtained for each metric and pipeline across all the signals.
5. Finally, we rank our pipelines sorting them by one of the computed scores.

Benchmark function
~~~~~~~~~~~~~~~~~~

The complete evaluation process is directly available using the
``orion.benchmark.benchmark`` function.

.. code-block:: python

    from orion.benchmark import benchmark

    pipelines = [
        'arima',
        'lstm_dynamic_threshold'
    ]

    metrics = ['f1', 'accuracy', 'recall', 'precision']

    signals = ['S-1', 'P-1']

    scores = benchmark(pipelines=pipelines, datasets=datasets, metrics=metrics, rank='f1')

For further details about all the arguments and possibilities that the ``benchmark`` function offers please refer to the `Orion benchmark
documentation <https://github.com/sintel-dev/Orion/blob/master/BENCHMARK.md>`__


.. _benchmark/results: https://github.com/sintel-dev/Orion/tree/master/benchmark/results
.. _summary Google Sheets document: https://docs.google.com/spreadsheets/d/1ZPUwYH8LhDovVeuJhKYGXYny7472HXVCzhX6D6PObmg/edit?usp=sharing
.. _details Google Sheets document: https://docs.google.com/spreadsheets/d/1HaYDjY-BEXEObbi65fwG0om5d8kbRarhpK4mvOZVmqU/edit?usp=sharing