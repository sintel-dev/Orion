.. _primitives:

==========
Primitives
==========

Primitives are data processing units. They are defined by the code that performs the actual processing and an annotated ``json`` file. To read more about primitives and their composition, visit `mlstars <https://sintel-dev.github.io/ml-stars/>`__.

Preprocessing
-------------

.. toctree::
    :maxdepth: 1

    primitives/time_segments_aggregate.rst
    primitives/intervals_to_mask.rst
    primitives/rolling_window_sequences.rst
    primitives/fillna.rst
    primitives/SimpleImputer.rst
    primitives/MinMaxScaler.rst

Modeling
--------

.. toctree::
    :maxdepth: 1

    primitives/arima.rst
    primitives/LSTMTimeSeriesRegressor.rst
    primitives/LSTMSeq2Seq.rst
    primitives/DenseSeq2Seq.rst
    primitives/TadGAN.rst
    primitives/AER.rst
    primitives/VAE.rst

Postprocessing
--------------

.. toctree::
    :maxdepth: 1

    primitives/score_anomalies.rst
    primitives/reconstruction_errors.rst
    primitives/regression_errors.rst
    primitives/find_anomalies.rst


Creating Primitives
-------------------

**Primitives** are basic operations and are essentially functions. There are currently a number of already available pipelines in `orion primtives`_ and `mlstars primitives`_.

A primitive consists of two items:

- python code that performs the operation/process on the data.
- an annotated JSON file that represents that primitives.

For easier interpretation, we will walk through an example to demonstrate how you can create a primitive.


Scaling Primitive Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Let's take a simple example of wanting to scale a one dimensional ``numpy`` array into be in the range ``[0, 1]``. Then a python function that will perform this operation would look something like this:

.. code-block:: python

    # file orion/primitives/preprocessing.py

    def scale(X):
    """Scale data into the range [0, 1].

    Args:
            X (array):
                A one dimensional numpy array of integers or floats.

    Returns:
            array:
                A transformed array with values between [0, 1].
    """
    minimum = X.min()
    maximum = X.max()

    return (X - minimum) / (maximum - minimum)


The function ``scale`` now will return a transformed version of `X`. So what would the JSON file for this primitive look like?

.. code-block:: json

    {
        "name": "orion.primitives.preprocessing.scale",
            "description": "Scale data to a [0, 1] range.",
        "classifiers": {
                    "type": "preprocessor",
            "subtype": "scaler"
        },
        "primitive": "orion.primitives.preprocessing.scale",
        "produce": {
            "args": [
                {
                    "name": "X",
                    "type": "ndarray"
                }
            ],
            "output": [
                {
                    "name": "y",
                    "type": "ndarray"
                }
            ]
        } 
    }

The above JSON file represents the primitive ``orion.primitives.preprocessing.scale``. It mentions the needed information to perform the operation as well as some valuable metadata such as the type of this primitive.

The most important section of the JSON is that it defines the input of the primitives, which is the variable ``X``, and the output, which is the variable ``y``. Essentially this means that the primitive is expecting a one dimensional array named ``X`` to produce a new transformed array ``y``.

Now that we have defined our function and JSON, we can use the primitive in pipelines.

Components of a Primitive
~~~~~~~~~~~~~~~~~~~~~~~~~

The previous example showed a simple case of a primitive with a single input and output and no hyperparameters. Below we show a comprehensive JSON representation with additional components.

The first section of the JSON describes metadata about the primitive, the second part contains functional information including the names of the methods and their arguments, the third part defines the hyperparameters.

.. code-block:: json

    {
        "name": "orion.primitives.primitive.PrimitiveName",
        "contributors": ["Author <email>"],
        "documentation": "reference to documentation or paper if available.",
        "description": "short description.",
        "classifiers": {
            "type": "postprocessor",
            "subtype": "anomaly_detector"
        },
        "modalities": [],
        "primitive": "orion.primitives.primitive.PrimitiveName",
        "fit": {
            "method": "fit",
            "args": [
                {
                    "name": "X",
                    "type": "ndarray"
                },
                {
                    "name": "y",
                    "type": "ndarray"
                }
            ]
        },
        "produce": {
            "method": "detect",
            "args": [
                {
                    "name": "X",
                    "type": "ndarray"
                },
                {
                    "name": "y",
                    "type": "ndarray"
                }
            ],
            "output": [
                {
                    "name": "y",
                    "type": "ndarray"
                }
            ]
        },
        "hyperparameters": {
            "fixed": {
                "hyper_name": {
                    "type": "str",
                    "default": "value"
                }
            }
        }
    }


The components of the JSON include:

- The name of the file and the name of the primitive should be following the name of the corresponding function and module it belongs to. For example, the function implemented in the previous example belongs in ``preprocessing.py`` in the ``orion.primitives`` module and thus the name of the JSON file is ``orion.primitives.preprocessing.scale.json`` and the name of the primitive would be ``orion.primitives.preprocessing.scale``.
- The description and metadata section of the primitive which includes: contributions, description, documentation, classifiers and modalities.
- The arguments section which includes the name of the function that will be called during the ``fit`` process and ``produce`` process. In addition, what are the expected arguments that it will receive and produce in both processes.
- Lastly, the hyperparameters section which includes a list of *fixed* hyperparameters. 


Adding a Primitive to Orion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have created your primitive and would like to make it part of the Orion library, follow these steps:

1. Create a new issue on github detailing the functionality of the primitive.
2. Open a new pull request associated with the issue that includes:
    1.  The code that implements the functionality of the primitive. Make sure it follows our code style and has documentation. This code should be located in python file in ``orion/primitives/``
    2. The JSON file of the primitive. This file should stored in ``orion/primitives/jsons/``
    3. Unit tests that ensures your code is covered.
3. The maintainers will review your code help improve your implementation and test it.
4. Once it has passed all test and is approved it will be merged!


Resources
---------

To read more about primitives, please refer to the original `MLBazaar paper <https://arxiv.org/pdf/1905.08942.pdf>`__ and `ml-stars documentation <https://sintel.dev/ml-stars/getting_started/concepts.html>`__.

If you have any questions please open an `issue <https://github.com/sintel-dev/Orion/issues/new>`__!

.. _orion primtives: https://github.com/sintel-dev/Orion/tree/master/orion/primitives/jsons
.. _mlstars primitives: https://github.com/sintel-dev/ml-stars/tree/master/mlstars/