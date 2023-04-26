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
