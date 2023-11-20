.. _pipelines:

=========
Pipelines
=========

The main component in the Orion project are the **Orion Pipelines**, which consist of `MLBlocks Pipelines <https://mlbazaar.github.io/MLBlocks/advanced_usage/pipelines.html>`__ specialized in detecting anomalies in time series.

As ``MLPipeline`` instances, **Orion Pipelines**:

* consist of a list of one or more `mlstars <https://sintel-dev.github.io/ml-stars/>`__
* can be *fitted* on some data and later on used to *predict* anomalies on more data
* can be *scored* by comparing their predictions with some known anomalies
* have *hyperparameters* that can be *tuned* to improve their anomaly detection performance
* can be stored as a JSON file that includes all the primitives that compose them, as well as other required configuration options.

Pipeline Representation
-----------------------

As previously mentioned, a pipeline is composed of a sequence of :ref:`primitives`. In Orion, we store pipelines as annotated **JSON** files.
Let's view the structure of a pipeline JSON for anomaly detection in `orion/pipelines <https://github.com/sintel-dev/Orion/tree/master/orion/pipelines>`__. For example let's consider the `ARIMA <https://github.com/sintel-dev/Orion/blob/master/orion/pipelines/verified/arima/arima.json>`__ pipeline. There are four main categories defined in the JSON:

.. code-block:: python

    {
        "primitives": [
            ...
        ],
        "init_params": {
            ...
        },
        "input_names": {
            ...
        },
        "output_names": {
            ...
        }
    }

* **primitives**: this is where we list the necessary primitives for the pipeline, where we can notice a series of preprocessing, modeling, and postprocessing primitives.
* **init_params**: this is where we initialize our pipeline parameters. If the parameters are not specified, then they will use default values. Notice how to change a value, we first specify the primitive as key then we change the parameter value.
* **input_names**: this is where we map the input name of a parameter in a primitive to a variable within the `Context <https://mlbazaar.github.io/MLBlocks/advanced_usage/pipelines.html#context>`__.
* **output_names**: this is where we assign the output of the primitive a variable name other than the default one within the primitive JSON.


Optionally there is an addition category `outputs`. In the case where you would like to view *intermediatry* outputs from the pipeline we can use this category to define it. Continuing on the previous example, ARIMA. To view the output of ``orion.primitives.timeseries_anomalies.regression_errors`` primitive, we include an additional ``visualization`` key with the output of interest:


.. code-block:: python

    {
        "outputs": {
    	    "default": [
                {
                    "name": "events",
                    "variable": "orion.primitives.timeseries_anomalies.find_anomalies#1.y"
                }
            ],
            "visualization": [
                {
                    "name": "errors",
                    "variable": "orion.primitives.timeseries_anomalies.regression_errors#1.errors"
                },
            ]
        }
    }

Then ``orion.detect(.., visualization=True)`` will return two outputs: the first output being the detected anomalies, and the second output is a dictionary containing the specified outputs in ``visualization``. You can read more about the specifications of :ref:`orion` API.

To read more about the intricacies of composing a pipeline, please refer to `MLBlocks Pipelines <https://mlbazaar.github.io/MLBlocks/advanced_usage/pipelines.html>`__.


Current Available Pipelines
---------------------------

In the **Orion Project**, the pipelines are included as **JSON** files, which can be found
in the subdirectories inside the ``orion/pipelines`` folder.

This is the list of pipelines available so far, which will grow over time:

+----------+------------------------------------------------------+
| name     | description                                          |
+----------+------------------------------------------------------+
| ARIMA    | ARIMA based pipeline                                 |
+----------+------------------------------------------------------+
| LSTM     | LSTM based pipeline inspired by the `NASA`_          |
+----------+------------------------------------------------------+
| Dummy    | Dummy pipeline for testing                           |
+----------+------------------------------------------------------+
| TadGAN   | GAN based pipeline with reconstruction based errors  |
+----------+------------------------------------------------------+
| LSTM AE  | Autoencoder based pipeline with LSTM layers          |
+----------+------------------------------------------------------+
| Dense AE | Autoencoder based pipeline with Dense layers         |
+----------+------------------------------------------------------+
| VAE      | Variational autoencoder                              |
+----------+------------------------------------------------------+
| AER      | Autoencoder with Regression based pipeline           |
+----------+------------------------------------------------------+
| Azure    | Azure API for `Anomaly Detector`_                    |
+----------+------------------------------------------------------+

Pipeline Storage
----------------

For each pipeline, there is a dedicated folder that stores:
* the pipeline itself
* the hyperparameter settings used for this pipeline to produce the results in the benchmark. To Learn more about it, we detail the process of benchmarking here.

We store a pipeline ``json`` within the pipeline subfolder. In addition, the hyperparameters would be called ``pipeline_dataset.json`` within the same folder. For example::

	├── tadgan/
	    ├── tadgan.json
	    └── tadgan_dataset.json
	└── arima/
	    ├── arima.json
	    └── arima_dataset.json

.. note::
	the pipeline name must follow the subfolder name.

Verified Pipelines
------------------

In **Orion**, we organize pipelines into *verified* and *sandbox*. The distinction between verified and sandbox is kept until several tests and verifications are made. We consider two cases when pipelines are inspected before transferring:

* When a new pipeline is proposed.
* When a new set of hyperparameters are suggested.

In both of these cases, the user is expected to open a new *PR* and pass tests before considering its merge and storage in sandbox.
Next, we test the new pipeline/hyperparameters in the benchmark and verify that they perform as expected and indicated by the user. Once these checks have passed, we make the transfer.

To know more about our process in contributing and testing, read our :ref:`contributing` guidelines and :ref:`benchmarking`.


Pipelines for Custom Data
-------------------------

To use a pipeline on your own data, first you need to make sure that it follows the data format described in :ref:`data`. Additionally, some hyperparameter changes in the pipeline might be necessary.

Since pipelines are composed of :ref:`primitives`, you can discover the interpretation of each hyperparameter by visiting the primitive's documentation. One of the most used primitives is ``time_segments_aggregate`` which makes your signal equi-spaced. You need to set the ``interval`` hyperparameter to the frequency of your data. For example, if your data has a frequency of 5 minutes then ``interval=300`` would be most appropriate since we are dealing with second intervals. A hands on example is shown here:

.. ipython:: python
    :okwarning:

    import numpy as np
    import pandas as pd

    from orion import Orion

    np.random.seed(0)
    custom_data = pd.DataFrame({"timestamp": np.arange(0, 150000, 300),
                                "value": np.random.randint(0, 10, 500)})

    hyperparameters = {
        "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
            "interval": 300
        },
        'keras.Sequential.LSTMTimeSeriesRegressor#1': {
            'epochs': 5,
            'verbose': True
        }
    }

    orion = Orion(
        pipeline='lstm_dynamic_threshold',
        hyperparameters=hyperparameters
    )

    orion.fit(custom_data)


Creating Pipelines
------------------

A **Pipeline** is a sequence of primitives that perform a number of operations to the given input to produce a desired output. In Orion, the pipeline takes a signal as input and produces a list of detected anomalies as output.

To create an Orion pipeline, we need to assemble a JSON representation of the pipeline. The representation contains four blocks mentioned above:

1. *primitives*
2. *init_params*
3. *input_names*
4. *output_names*

Let’s visit one of the existing Orion pipelines to help understand how it is working, as an example we will work with the ``lstm_dynamic_threshold`` `pipeline <https://github.com/sintel-dev/Orion/blob/master/orion/pipelines/verified/lstm_dynamic_threshold/lstm_dynamic_threshold.json>`__.


Block 1: Primitives
~~~~~~~~~~~~~~~~~~~

The first block in the pipeline JSON is a list containing the sequence of primitives applied to the signal in order to produce a list of detected anomalies.

.. code-block:: json

    "primitives": [
        "mlstars.custom.timeseries_preprocessing.time_segments_aggregate",
        "sklearn.impute.SimpleImputer",
        "sklearn.preprocessing.MinMaxScaler",
        "mlstars.custom.timeseries_preprocessing.rolling_window_sequences",
        "keras.Sequential.LSTMTimeSeriesRegressor",
        "orion.primitives.timeseries_errors.regression_errors",
        "orion.primitives.timeseries_anomalies.find_anomalies"
    ]

The primitive ``time_segments_aggregate`` takes a raw signal to produce another signal where
the time intervals between each sample and another is equal. Then the pipeline uses ``SimpleImputer`` to impute missing values in the signal with the mean value. Next, the pipeline scale the data into the range ``[-1, 1]`` using ``MinMaxScaler``. After that, ``rolling_window_sequences`` create the training window sequences for the model. The
processed signal is now ready to train the double-stacked LSTM network through ``LSTMTimeSeriesRegressor`` primitive. Once the network is trained, we generate the predicted signal and compute the discrepancies (error) using ``regression_errors``, which
is an absolute point-wise difference. Lastly, we use ``find_anomalies`` on error values to find anomalous regions. 

Customizing pipelines and swapping one primitive with another can be done by changing and adding primitives in this block.


Block 2: Initialization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second block in the pipeline JSON is a dictionary of primitive names and their associated initialized parameters.

.. code-block:: json

    "init_params": {
        "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
            "time_column": "timestamp",
            "interval": 21600,
            "method": "mean"
        },
        "sklearn.preprocessing.MinMaxScaler#1": {
            "feature_range": [
                -1,
                1
            ]
        },
        "mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1": {
            "target_column": 0,
            "window_size": 250
        },
        "keras.Sequential.LSTMTimeSeriesRegressor": {
            "epochs": 35
        },
        "orion.primitives.timeseries_anomalies.find_anomalies#1": {
            "window_size_portion": 0.33,
            "window_step_size_portion": 0.1,
            "fixed_threshold": true
        }
    }

For example, in ``time_segments_aggregate`` we are marking that the interval between one sample and another in the signal is 6 hours which is 21600. Another example is that the ``LSTMTimeSeriesRegressor`` model is trained for 35 epochs.

If the parameters are not present in this block, the default value in the original primitive JSON file will be used. For example if the interval was not specified, then it will use the default value (which is 3600) that is defined in the original `primitive <https://github.com/sintel-dev/ml-stars/blob/master/mlstars/primitives/mlstars.custom.timeseries_preprocessing.time_segments_aggregate.json>`__.


Block 3: Input Names
~~~~~~~~~~~~~~~~~~~~

Block three is used for mapping name of argument to existing variables. Recall that primitives define input arguments and outputs. These variables names are how the pipeline knows which primitive uses which variables.

Sometimes we need to change the names of these input arguments to find the correct variable.

.. code-block:: json

    "input_names": {
        "orion.primitives.timeseries_anomalies.find_anomalies#1": {
            "index": "target_index"
        }
    }


Here the pipeline changes the input argument ``index`` to ``target_index`` which is basically denoting that the input ``index`` will now be the output ``target_index`` of a previous primitive which is the `rolling_window_sequences <https://github.com/sintel-dev/ml-stars/blob/master/mlstars/primitives/mlstars.custom.timeseries_preprocessing.rolling_window_sequences.json>`__ primitive.


Block 4: Output Names
~~~~~~~~~~~~~~~~~~~~~

Similarly to input names, sometimes we need to change output names, which is what the pipeline specifies in block four.

.. code-block:: json

    "output_names": {
        "keras.Sequential.LSTMTimeSeriesRegressor#1": {
            "y": "y_hat"
        }
    }

Here we change the output of the primitive ``LSTMTimeSeriesRegressor`` from ``y`` to ``y_hat`` in order not to override the existing value of ``y`` and also to prepare it as input for the next primitive `regression_errors <https://github.com/sintel-dev/Orion/blob/master/orion/primitives/jsons/orion.primitives.timeseries_errors.regression_errors.json>`__.


Contributing a Pipeline
-----------------------

To add a new pipeline to Orion, follow the same steps mentioned in :ref:`int_benchmark` page.


.. _NASA: https://arxiv.org/abs/1802.04431
.. _Anomaly Detector: https://azure.microsoft.com/en-us/services/cognitive-services/anomaly-detector/
