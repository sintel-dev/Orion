.. highlight:: shell

time segments aggregate
~~~~~~~~~~~~~~~~~~~~~~~

**path**: ``mlstars.custom.timeseries_preprocessing.time_segments_aggregate``

**description**: this primitive creates an equi-spaced time series by aggregating values over fixed specified interval.

see `json <https://github.com/MLBazaar/mlstars/blob/master/mlstars/primitives/mlstars.custom.timeseries_preprocessing.time_segments_aggregate.json>`__.

==================== =========================================== =============================================================================================================================
argument              type                                        description  
**parameters**
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 ``X``                ``numpy.ndarray`` or ``pandas.DataFrame``   n-dimensional sequence of values
 ``time_column``      ``str``                                     column of ``X`` that contains time values
**hyperparameters**
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 ``interval``         ``int``                                     integer denoting time span to compute aggregation of
 ``method``           ``str``                                     string describing aggregation method or list of strings describing multiple aggregation methods. If not given, mean is used
**output**
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 ``X``                ``numpy.ndarray``                           sequence of aggregated values, one column for each aggregation method
 ``index``            ``numpy.ndarray``                           sequence of index values (first index of each aggregated segment)
==================== =========================================== =============================================================================================================================

.. ipython:: python
    :okwarning:

    from mlstars import load_primitive

    primitive = load_primitive('mlstars.custom.timeseries_preprocessing.time_segments_aggregate', 
        arguments={"time_column": "timestamp", "interval":10, "method":'mean'})

    df = pd.DataFrame({
        'timestamp': list(range(50)),
        'value': [1] * 50})

    X, index = primitive.produce(X=df)
    pd.DataFrame({"timestamp": index, "value": X[:, 0]})
    