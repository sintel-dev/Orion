.. highlight:: shell

rolling window sequence
~~~~~~~~~~~~~~~~~~~~~~~

**path**: ``mlstars.custom.timeseries_preprocessing.rolling_window_sequences``

**description**: this primitive generates many sub-sequences of the original sequence. it uses a rolling window approach to create the sub-sequences out of time series data.

see `json <https://github.com/MLBazaar/mlstars/blob/master/mlstars/primitives/mlstars.custom.timeseries_preprocessing.rolling_window_sequences.json>`__.

==================== ============================================================== ==================================================================
 argument             type                                                           description  

**parameters**
------------------------------------------------------------------------------------------------------------------------------------------------------

 ``X``                ``numpy.ndarray``                                              n-dimensional sequence to iterate over
 ``index``            ``numpy.ndarray``                                              array containing the index values of X
 ``drop``             ``numpy.ndarray``, ``str``, ``float``, ``bool``, or ``None``   array of boolean values indicating which value should be dropped 

**hyperparameters**
------------------------------------------------------------------------------------------------------------------------------------------------------

 ``window_size``      ``int``                                                        length of the input sequences
 ``target_size``      ``int``                                                        length of the target sequences
 ``step_size``        ``int``                                                        indicating the number of steps to move the window forward each round
 ``target_column``    ``int``                                                        indicating which column of ``X`` is the target
 ``drop_windows``     ``bool``                                                       indicates whether the dropping functionality should be enabled

**output**
------------------------------------------------------------------------------------------------------------------------------------------------------

 ``X``                ``numpy.ndarray``                                              input sequences
 ``y``                ``numpy.ndarray``                                              target sequences
 ``index``            ``numpy.ndarray``                                              first index value of each input sequence
 ``target_index``     ``numpy.ndarray``                                              first index value of each target sequence
==================== ============================================================== ==================================================================

.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    primitive = load_primitive('mlstars.custom.timeseries_preprocessing.rolling_window_sequences', 
        arguments={"window_size": 10, "target_size": 1, "step_size": 1, "target_column": 0})

    X = np.array([1] * 50).reshape(-1, 1)
    index = np.array(range(50)).reshape(-1, 1)

    X, y, index, target_index = primitive.produce(X=X, index=index)
    print("X shape = {}\ny shape = {}\nindex shape = {}\ntarget index shape = {}".format(
        X.shape, y.shape, index.shape, target_index.shape))
