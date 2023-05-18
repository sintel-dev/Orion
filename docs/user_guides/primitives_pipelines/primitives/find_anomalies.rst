.. highlight:: shell

find anomalies
~~~~~~~~~~~~~~

**path**: ``orion.primitives.timeseries_anomalies.find_anomalies``

**description**: this primitive extracts anomalies from sequences of errors following the approach explained in the `related paper <https://arxiv.org/pdf/1802.04431.pdf>`__.

see `json <https://github.com/sintel-dev/Orion/tree/master/orion/primitives/jsons/orion.primitives.timeseries_anomalies.find_anomalies.json>`__.

========================== ==================== ===================================================================================================================================
argument                    type                 description  

**parameters**
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 ``errors``                 ``numpy.ndarray``    array of errors
 ``index``                  ``numpy.ndarray``    array of indices of errors

**hyperparameters**
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 ``z_range``                ``tuple``            list of two values denoting the range out of which the start points for the ``scipy.fmin`` function are chosen
 ``window_size``            ``int``              size of the window for which a threshold is calculated
 ``window_step_size``       ``int``              number of steps the window is moved before another threshold is calculated for the new window
 ``lower_threshold``        ``bool``             indicates whether to apply a lower threshold to find unusually low errors
 ``fixes_threshold``        ``bool``             indicates whether to use fixed or dynamic thresholding
 ``min_percent``            ``float``            percentage of separation the anomalies need to meet between themselves and the highest non-anomalous error in the window sequence
 ``anomaly_padding``        ``int``              number of errors before and after a found anomaly that are added to the anomalous sequence
 
**output**
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 ``y``                      ``numpy.ndarray``    array containing start-index, end-index, score for each anomalous sequence that was found
========================== ==================== ===================================================================================================================================

.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    primitive = load_primitive('orion.primitives.timeseries_anomalies.find_anomalies',
        arguments={"anomaly_padding": 1})

    errors = np.array([0.01] * 45 + [1] * 10 + [0.01] * 45)
    index = np.array(range(100))

    primitive.produce(errors=errors, index=index)

