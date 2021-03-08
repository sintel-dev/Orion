.. _quickstart:

Quickstart
==========

In the following steps we will show a short guide about how to run one of the **Orion Pipelines**
on one of the signals from the **Demo Dataset**.

1. Load the data
----------------

In the first step we will load the **S-1** signal from the **Demo Dataset**.

We will do so in two parts, we will use the first part to fit the
pipeline and the second one to detect anomalies.

To do so, we need to import the `orion.data.load_signal` function and call it passing the `'S-1-train'` as signal name.

.. ipython:: python
    :okwarning:

	from orion.data import load_signal

	train_data = load_signal('S-1-train')

	train_data.head()

The output will be a table that contains two columns `timestamp` and `value`.

2. Detect anomalies using Orion
-------------------------------

Once we have the data, let us try to use an Orion pipeline to analyze it and search for anomalies.

In order to do so, we will have to create an instance of the `orion.Orion` class.

.. ipython:: python
    :okwarning:

	from orion import Orion

	orion = Orion()

Optionally, we might want to select a pipeline other than the default one or alter the hyperparameters by the underlying MLBlocks pipeline.

For example, let's select the ``lstm_dynamic_threshold`` pipeline and set some hyperparameters (in this case training epochs as 5).

.. ipython:: python
    :okwarning:

    hyperparameters = {
        'keras.Sequential.LSTMTimeSeriesRegressor#1': {
            'epochs': 5
        }
    }

Then, we simply pass the ``hyperparameters`` alongside the pipeline.

.. ipython:: python
    :okwarning:

    orion = Orion(
        pipeline='lstm_dynamic_threshold',
        hyperparameters=hyperparameters
    )

Once we the pipeline is ready, we can proceed to fit it to our data:

.. ipython:: python
    :okwarning:

	orion.fit(train_data)


Once it is fitted, we are ready to use it to detect anomalies in incoming data:

.. ipython:: python
    :okwarning:

	new_data = load_signal('S-1-new')
	anomalies = orion.detect(new_data)

.. warning::

	Depending on your system and the exact versions that you might have installed some *WARNINGS* may be printed. These can be safely ignored as they do not interfere with the proper behavior of the pipeline.

The output of the previous command will be a ``pandas.DataFrame`` containing a table in the detected anomalies
Output format described above:

.. ipython:: python
    :okwarning:

    anomalies


3. Evaluate the performance of your pipeline
--------------------------------------------

In this next step we will load some already known anomalous intervals and evaluate how
good our anomaly detection was by comparing those with our detected intervals.

For this, we will first load the known anomalies for the signal that we are using:

.. ipython:: python
    :okwarning:

	from orion.data import load_anomalies

	ground_truth = load_anomalies('S-1')

	ground_truth

The output will be a table in the same format as the `anomalies` one.

Afterwards, we can call the ``orion.evaluate`` method, passing both the data to detect anomalies and the ground truth:

.. ipython:: python
    :okwarning:

	scores = orion.evaluate(new_data, ground_truth)
	scores

The output will be a ``pandas.Series`` containing a collection of scores indicating how the predictions were.
