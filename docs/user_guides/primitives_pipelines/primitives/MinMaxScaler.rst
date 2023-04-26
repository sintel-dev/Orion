.. highlight:: shell

MinMaxScaler
~~~~~~~~~~~~

**path**: ``sklearn.preprocessing.MinMaxScaler``

**description**: this primitive transforms features by scaling each feature to a given range.

see `json <https://github.com/MLBazaar/mlstars/blob/master/mlstars/primitives/sklearn.preprocessing.MinMaxScaler.json>`__.

==================== =================== =============================================================================================================
argument              type                description  
**parameters**
------------------------------------------------------------------------------------------------------------------------------------------------------

 ``X``                ``numpy.ndarray``   the data used to compute the per-feature minimum and maximum used for later scaling along the features axis

**hyperparameters**
------------------------------------------------------------------------------------------------------------------------------------------------------

 ``feature_range``    ``tuple``           desired range of transformed data. Default set to ``[0, 1]`` 
 ``copy``             ``bool``            if True, a copy of X will be created

**output**
------------------------------------------------------------------------------------------------------------------------------------------------------

 ``X``                ``numpy.ndarray``   a transformed version of X
==================== =================== =============================================================================================================


.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    X = np.array(range(5)).reshape(-1, 1)
    primitive = load_primitive('sklearn.preprocessing.MinMaxScaler', 
        arguments={"X": X, "feature_range":[0, 1]})

    primitive.fit()
    primitive.produce(X=X)
