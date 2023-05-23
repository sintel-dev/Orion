.. highlight:: shell

ARIMA
~~~~~

**path**: ``statsmodels.tsa.arima_model.Arima``

**description**: this is an Autoregressive Integrated Moving Average (ARIMA) prediction model.

see `json <https://github.com/MLBazaar/mlstars/blob/master/mlstars/primitives/statsmodels.tsa.arima_model.Arima.json>`__.

==================== =================== ==================================================================
argument              type                description  

**parameters**
-----------------------------------------------------------------------------------------------------------

 ``X``                ``numpy.ndarray``   n-dimensional array containing the input sequences for the model 

**hyperparameters**
-----------------------------------------------------------------------------------------------------------

 ``steps``            ``int``             number of forward steps to predict 
 ``p``                ``int``             the number of autoregressive parameters to use
 ``d``                ``int``             the number of differences to use
 ``q``                ``int``             the number of moving average (MA) parameters to use

**output**
-----------------------------------------------------------------------------------------------------------

 ``y``                ``numpy.ndarray``   predicted values
==================== =================== ==================================================================

.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    X = np.array(range(100)).reshape(-1, 1)
    primitive = load_primitive('statsmodels.tsa.arima_model.Arima', 
        arguments={"X": X, "steps": 1, })

    primitive.produce(X=X)
