.. _primitives:

================
Orion Primitives
================

Primitives are data processing units. They are defined by the code that performs the actual processing and an annotated ``json`` file. To read more about primitives and their composition, visit `MLPrimitives <https://mlbazaar.github.io/MLPrimitives/>`__.

Preprocessing
-------------

.. include:: primitives/time_segments_aggregate.rst
.. include:: primitives/intervals_to_mask.rst
.. include:: primitives/rolling_window_sequences.rst
.. include:: primitives/SimpleImputer.rst
.. include:: primitives/MinMaxScaler.rst

Modeling
--------

.. include:: primitives/arima.rst
.. include:: primitives/LSTMTimeSeriesRegressor.rst
.. include:: primitives/TadGAN.rst

Postprocessing
--------------

.. include:: primitives/score_anomalies.rst
.. include:: primitives/regression_errors.rst
.. include:: primitives/find_anomalies.rst