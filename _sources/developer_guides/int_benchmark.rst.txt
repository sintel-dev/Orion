.. _int_benchmark:

===============================
Contributing to Orion Benchmark
===============================

Below we provide a guide on how to integrate new pipelines, datasets, and metrics.

Adding a Pipeline to Orion Benchmark
------------------------------------

To add a new pipeline to the benchmark, we need to make it part of the Orion pipelines available for all users. Before starting, please review some of the following concepts to help you understand pipelines. 

- **Standardized Orion I/O Format:** please visit :ref:`data` page.
- **Primitives and Pipelines:**
    - **Creating Primitives:** please visit :ref:`primitives` page.
    - **Creating Pipelines:** please visit :ref:`pipelines` page.

Once you have a pipeline that you have been using that you would like to add to Orion, please follow these steps in order to make an official Orion pipeline!

1. **Open an issue:** first task is to `open an issue`_ on Orion github detailing your pipeline and its primitives. This will help maintainers understand what the pipeline is doing and suggest potential primitives that can be used rather than creating one from scratch.
2. **Open a pull request:** after describing your pipeline clearly in an issue, you can open a pull request on Orion github to make your contributions. This pull request should have the following files added or modified:
    1. **primitive files:** this includes any python code and primitive JSON files needed by the pipeline. These files should be stored in ``orion/primitives`` and ``orion/primitives/jsons`` respectively. Moreover, for any newly added classes and functions, please include unit tests to make sure they are performing as intended in ``tests/primitives``.
    2. **pipeline files:** this includes the main pipeline JSON file. Create a new directory under ``orion/pipelines/sandbox`` with the given pipeline name and store the pipeline JSON file inside it. Imagine the model name is ``new_model`` then the pipeline should have the following path ``orion/pipelines/sandbox/new_model/new_model.json``.
    3. **hyperparameter files:** in addition to the pipeline JSON file, create hyperparameter JSON files for each dataset used in the benchmark and store it in the same location as the pipeline. As an example, for dataset SMAP, create ``orion/pipelines/sandbox/new_model/new_model_smap.json``. The hyperparameter file should only modify the data specific hyperparameters such as the interval level of the signals in the dataset.
    4. **documentation:** lastly there is documentation files. Create a new ``docs/user_guides/primitives_pipelines/primitives/new_primitive.rst`` that describes the primitive for any newly added primitive. Moreover, include modify ``docs/user_guides/primitives_pipelines/pipelines.rst`` and ``docs/user_guides/primitives_pipelines/primitives.rst`` as needed.
3. **Reviewing:** once the pull request is made, the Orion team will take a look at your contributions and make any necessary suggestions. When the pull request passes unit and integration tests and the code is approved by the reviewers, it will be merged. The pipeline will remain in sandbox until it passes the verification testing phase.
4. **Verification Testing:** to ensure that pipelines are robust, reproducible, and can be maintained in the long run, several tests and validations are made. This includes testing the pipeline in our benchmarks.
5. **Transferring from Sandbox to Verified:** once the pipeline passes verification testing, it becomes an Orion verified pipeline and will be included in all future releases of the benchmark.


Adding a Dataset to Orion Benchmark
-----------------------------------

To add a new dataset to the benchmark, it needs to follow the Orion format. Visit the :ref:`data` to familiarize yourself with it.

Once the data follows the expected input format, please follow these steps to introduce the dataset to the benchmark:

1. **Open an issue:** first task is to `open an issue`_ on Orion github pointing to the source of the data. If the data needs formatting, please attach a link to a notebook (either in your fork or a colab notebook) to make the integration faster.
2. **Adding the dataset to S3:** if the data is publicly available, we would like to make it available to all Orion users by adding it to our ``sintel-orion`` public s3 bucket. This way, users can load any signal directly from Orion using ``load_signal`` functionality. This task will be performed by the Orion team.
3. **Adding dataset/pipeline hyperparameters:** open a new pull request that will feature adding the respective hyperparameters of the dataset for each pipeline. For example, for a new dataset named ``new_data`` we will have ``aer_new_data.json``, ``tadgan_new_data.json``, etc. While this might feel redundant, it is crucial to maintain transparency of hyperparameter settings in our benchmarks.
4. **Adding the dataset to leaderboard:** in the same pull request, kindly add the name of the dataset to the dictionaries included in this `file <https://github.com/sintel-dev/Orion/blob/master/orion/results.py>`__.

Once the pull request is merged, the dataset will then be featured in subsequent releases of the benchmark!


Adding an Evaluation Metric to Orion Benchmark
----------------------------------------------

Orion has an evaluation sub-package for evaluating anomaly detection performance. Before adding a new evaluation metric, please visit :ref:`evaluation_doc`.

To make an evaluation function it needs to accept at least two arguments:

- ``expected``: which is a list of known ground truth anomalies.
- ``observed``: which is a list of detected anomalies.

If you are working with *point* anomalies, you can add your metric to ``point.py`` in the evaluation sub-package, and if you are working with *contextual* anomalies, please include it in ``contextual.py``.

Once you have created your metric, you can start the process of integrating it to the benchmark:

1. **Open an issue:** first task is to `open an issue`_ on Orion github detailing the specifications of the new evaluation metric and how it will be useful to users.
2. **Open a pull request:** after describing your new metric clearly in an issue, you can open a pull request on Orion github to make your contributions. Below we describe what the PR should include:
    1. **metric files:** this includes the python functions that implement the evaluation metric. Note that these files should be stored in the evaluation sub-package.
    2. **benchmark file:** to add the new metric to the benchmark, simply add it to the dictionary ``METRICS``.  Once included, it will be added to the benchmark detailed sheet results.
    3. **documentation:** lastly there is documentation. Add a description of the new metric in ``docs/user_guides/evaluation_doc.rst``. It is also valuable to add an example usage of the metric with an expected result.
3. **Reviewing:** once the pull request is made, the Orion team will make any necessary suggestions. Please include docstrings in your code to help the team in the reviewing process.

Once the PR is merged, the new evaluation metric will be available to all users. Moreover, subsequent benchmark released will contain the new metric in their benchmark results.


Resources
---------

- **Data Format:** :ref:`data` page.
- **Primitives:** :ref:`primitives` page.
- **Pipelines:** :ref:`pipelines` page.

.. _open an issue: https://github.com/sintel-dev/Orion/issues/new
