.. highlight:: shell

LSTM AE
~~~~~~~

**path**: ``keras.Sequential.LSTMSeq2Seq``

**description**: this is a reconstruction model autoencoder using LSTM layers.

see `json <https://github.com/sintel-dev/Orion/blob/master/orion/primitives/jsons/keras.Sequential.LSTMSeq2Seq.json>`__.

====================== =================== ===========================================================================================================================================
argument                type                description  

**parameters**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 ``X``                  ``numpy.ndarray``   n-dimensional array containing the input sequences for the model 
 ``y``                  ``numpy.ndarray``   n-dimensional array containing the target sequences for the model 

**hyperparameters**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 ``classification``     ``bool``            indicator of whether this is a classification or regression model. Default is False
 ``epochs``             ``int``             number of epochs to train the model. An epoch is an iteration over the entire X and y data provided. Default is 35
 ``callbacks``          ``list``            list of callbacks to apply during training
 ``validation_split``   ``float``           float between 0 and 1. Fraction of the training data to be used as validation data. Default is 0.2
 ``batch_size``         ``int``             number of samples per gradient update. Default is 64
 ``window_size``        ``int``             integer denoting the size of the window per input sample
 ``input_shape``        ``tuple``           tuple denoting the shape of an input sample
 ``target_shape``       ``tuple``           tuple denoting the shape of an output sample
 ``optimizer``          ``str``             string (name of optimizer) or optimizer instance. Default is ``keras.optimizers.Adam``
 ``loss``               ``str``             string (name of the objective function) or an objective function instance. Default is ``keras.losses.mean_squared_error``
 ``metrics``            ``list``            list of metrics to be evaluated by the model during training and testing. Default is ["mse"]
 ``return_seqeunces``   ``bool``            whether to return the last output in the output sequence, or the full sequence. Default is False
 ``layers``             ``list``            list of keras layers which are the basic building blocks of a neural network
 ``verbose``            ``bool``            verbosity mode. Default is False
 ``lstm_1_unit``        ``int``             dimensionality of the output space for the first LSTM layer. Default is 80
 ``dropout_1_rate``     ``float``           float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs for the first LSTM layer. Default: 0.3
 ``lstm_2_unit``        ``int``             dimensionality of the output space for the second LSTM layer. Default is 80
 ``dropout_2_rate``     ``float``           float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs for the second LSTM layer. Default: 0.3

**output**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 ``y``                  ``numpy.ndarray``   predicted values
====================== =================== ===========================================================================================================================================

.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    X = np.array([1] * 100).reshape(1, -1, 1)

    primitive = load_primitive('keras.Sequential.LSTMSeq2Seq', 
        arguments={"X": X, "y": X, "input_shape":(100, 1), "target_shape":(100, 1), 
                   "window_size": 100, "batch_size": 1, "validation_split": 0, "epochs": 5})

    primitive.fit()
    pred = primitive.produce(X=X)
    pred.mean()
