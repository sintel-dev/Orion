.. highlight:: shell

AER
~~~~~~

**path**: ``orion.primitives.aer.AER``

**description**: this an autoencoder-based model capable of creating both prediction-based and reconstruction-based anomaly scores.

see `json <https://github.com/sintel-dev/Orion/tree/master/orion/primitives/jsons/orion.primitives.aer.AER.json>`__.

========================== =================== =================================================================================================
argument                    type                description

**parameters**
------------------------------------------------------------------------------------------------------------------------------------------------
 ``X``                      ``numpy.ndarray``   n-dimensional array containing the input sequences for the model
 ``y``                      ``numpy.ndarray``   n-dimensional array containing the target sequences we want to reconstruct. Typically ``y`` is a signal from a selected set of channels from ``X``.
**hyperparameters**
------------------------------------------------------------------------------------------------------------------------------------------------

 ``epochs``                 ``int``             number of epochs to train the model. An epoch is an iteration over the entire X data provided
 ``input_shape``            ``tuple``           tuple denoting the shape of an input sample
 ``optimizer``              ``str``             string (name of optimizer) or optimizer instance. Default is ``keras.optimizers.Adam``
 ``learning_rate``          ``float``           float denoting the learning rate of the optimizer. Default is 0.001
 ``batch_size``             ``int``             number of samples per gradient update. Default is 64
 ``layers_encoder``         ``list``            list containing layers of encoder
 ``layers_generator``       ``list``            list containing layers of generator

**output**
------------------------------------------------------------------------------------------------------------------------------------------------

 ``ry_hat``                 ``numpy.ndarray``    n-dimensional array containing the regression for each input sequence (reverse)
 ``y_hat``                  ``numpy.ndarray``    n-dimensional array containing the reconstructions for each input sequence
 ``fy_hat``                 ``numpy.ndarray``    n-dimensional array containing the regression for each input sequence (forward)
========================== =================== =================================================================================================


.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    X = np.ones((64, 100, 1))
    y = X[:,:, [0]] # signal to reconstruct from X (channel 0)
    primitive = load_primitive('orion.primitives.aer.AER',
        arguments={"X": X, "y": y, "epochs": 1, "batch_size": 1})

    primitive.fit()
    ry, y, fy = primitive.produce(X=X)

    print("Reverse Prediction: {}\nReconstructed Values: {}, Forward Prediction: {}".format(ry, y, fy))

