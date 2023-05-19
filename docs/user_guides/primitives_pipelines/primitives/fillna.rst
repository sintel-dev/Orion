.. highlight:: shell

fillna
~~~~~~

**path**: ``orion.primitives.timeseries_preprocessing.fillna``

**description**: this primitive is an iterative imputation transformer for filling missing values using ``pandas.fillna``.

see `json <https://github.com/sintel-dev/Orion/blob/master/orion/primitives/jsons/orion.primitives.timeseries_preprocessing.fillna.json>`__.

================== =============================================================== ============================================
argument            type                                                            description  

**parameters**
-------------------------------------------------------------------------------------------------------------------------------
 ``X``              ``numpy.ndarray``                                               n-dimensional sequence of values

**hyperparameters**
-------------------------------------------------------------------------------------------------------------------------------

 ``value``          ``int``, ``dict``, ``pandas.Series``, or ``pandas.DataFrame``   Value to use to fill holes (e.g. 0), alternately a dict/Series/DataFrame of values specifying which value to use for each index (for a Series) or column (for a DataFrame). Values not in the dict/Series/DataFrame will not be filled. This value cannot be a list
 ``method``         ``str``, or ``list``                                            String or list of strings describing whether to use forward or backward fill. pad / ffill: propagate last valid observation forward to next valid. backfill / bfill: use next valid observation to fill gap. Otherwise use ``None`` to fill with desired value. Possible values include ``['backfill', 'bfill', 'pad', 'ffill', None]``
 ``axis``           ``int``, or ``str``                                             Axis along which to fill missing value. Possible values include 0 or "index", 1 or "columns"
 ``limit``          ``int``                                                         If method is specified, this is the maximum number of consecutive NaN values to forward/backward fill. In other words, if there is a gap with more than this number of consecutive NaNs, it will only be partially filled. If method is not specified, this is the maximum number of entries along the entire axis where NaNs will be filled. Must be greater than 0 if not ``None``.
 ``downcast``       ``dict``                                                        A dict of item->dtype of what to downcast if possible, or the string "infer" which will try to downcast to an appropriate equal type (e.g. float64 to int64 if possible)

**output**
-------------------------------------------------------------------------------------------------------------------------------

 ``X``              ``numpy.ndarray``                                               Array of input sequence with imputed values
================== =============================================================== ============================================

.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    X = np.array([1] * 4 + [np.nan]).reshape(-1, 1)
    primitive = load_primitive('orion.primitives.timeseries_preprocessing.fillna', 
        arguments={"X": X, "value": 0})

    primitive.fit()
    primitive.produce(X=X)
