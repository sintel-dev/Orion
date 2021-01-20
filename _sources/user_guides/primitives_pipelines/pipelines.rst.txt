.. _pipelines:

=========
Pipelines
=========

The main component in the Orion project are the **Orion Pipelines**, which consist of `MLBlocks Pipelines <https://mlbazaar.github.io/MLBlocks/advanced_usage/pipelines.html>`__ specialized in detecting anomalies in time series.

As ``MLPipeline`` instances, **Orion Pipelines**:

* consist of a list of one or more `MLPrimitives <https://mlbazaar.github.io/MLPrimitives/>`__
* can be *fitted* on some data and later on used to *predict* anomalies on more data
* can be *scored* by comparing their predictions with some known anomalies
* have *hyperparameters* that can be *tuned* to improve their anomaly detection performance
* can be stored as a JSON file that includes all the primitives that compose them, as well as other required configuration options.

Current Available Pipelines
---------------------------

In the **Orion Project**, the pipelines are included as **JSON** files, which can be found
in the subdirectories inside the ``orion/pipelines`` folder.

This is the list of pipelines available so far, which will grow over time:

+--------+------------------------------------------------------+
| name   | description                                          |
+--------+------------------------------------------------------+
| ARIMA  | ARIMA based pipeline                                 |
+--------+------------------------------------------------------+
| LSTM   | LSTM based pipeline inspired by the `NASA`_          |
+--------+------------------------------------------------------+
| Dummy  | Dummy pipeline for testing                           |
+--------+------------------------------------------------------+
| TadGAN | GAN based pipeline with reconstruction based errors  |
+--------+------------------------------------------------------+
| Azure  | Azure API for `Anomaly Detector`_                    |
+--------+------------------------------------------------------+

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

Verified pipelines
------------------

In **Orion**, we organize pipelines into *verified* and *sandbox*. The distinction between verified and sandbox is kept until several tests and verifications are made. We consider two cases when pipelines are inspected before transferring:

* When a new pipeline is proposed.
* When a new set of hyperparameters are suggested.

In both of these cases, the user is expected to open a new *PR* and pass tests before considering its merge and storage in sandbox.
Next, we test the new pipeline/hyperparameters in the benchmark and verify that they perform as expected and indicated by the user. Once these checks have passed, we make the transfer.

To know more about our process in contributing and testing, read our :ref:`contributing` guidelines and :ref:`benchmarking`.

.. _NASA: https://arxiv.org/abs/1802.04431
.. _Anomaly Detector: https://azure.microsoft.com/en-us/services/cognitive-services/anomaly-detector/
