.. highlight:: shell

intervals to mask
~~~~~~~~~~~~~~~~~

**path**: ``mlstars.custom.timeseries_preprocessing.intervals_to_mask``

**description**: this primitive creates boolean mask from given intervals.

see `json <https://github.com/MLBazaar/mlstars/blob/master/mlstars/primitives/mlstars.custom.timeseries_preprocessing.intervals_to_mask.json>`__.

==================== =============================== =================================================================================================================================
argument              type                            description  

**parameters**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 ``index``            ``numpy.ndarray``               array containing the index values
 ``intervals``        ``list`` or ``numpy.ndarray``   list or array of intervals, consisting of start-index and end-index for each interval

**output**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 ``mask``             ``numpy.ndarray``               array of boolean values, with one boolean value for each index value (True if the index value is contained in a given interval)
==================== =============================== =================================================================================================================================

.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    primitive = load_primitive('mlstars.custom.timeseries_preprocessing.intervals_to_mask')

    index = np.array(range(10))
    intervals = [(1, 3), (7, 7)]

    primitive.produce(index=index, intervals=intervals)
