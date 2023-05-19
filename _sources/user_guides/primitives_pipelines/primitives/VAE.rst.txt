.. highlight:: shell

VAE
~~~

**path**: ``orion.primitives.vae.VAE``

**description**: this is a reconstruction model using Variational AutoEncoder.

see `json <https://github.com/sintel-dev/Orion/tree/master/orion/primitives/jsons/orion.primitives.vae.VAE.json>`__.

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
 ``output_shape``           ``tuple``           tuple denoting the shape of an output sample
 ``latent_dim``             ``int``             integer denoting dimension of latent space. Default is 20.
 ``learning_rate``          ``float``           float denoting the learning rate of the optimizer. Default is 0.001
 ``optimizer``              ``str``             string (name of optimizer) or optimizer instance. Default is ``keras.optimizers.Adam``
 ``batch_size``             ``int``             number of samples per gradient update. Default is 64
 ``shuffle``                ``bool``            whether to shuffle the training data before each epoch. Default is True.
 ``verbose``                ``int``             verbosity mode where 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 0.
 ``lstm_units``             ``int``             number of neurons (dimensionality of the output space).
 ``length``                 ``int``             equal to input_shape[0].
 ``callbacks``              ``list``            list of ``keras.callbacks.Callback`` instances. List of callbacks to apply during training.
 ``validation_split``       ``float``           fraction of the training data to be used as validation data. Default 0.
 ``output_dim``             ``int``             equal to output_shape[-1]
 ``layers_encoder``         ``list``            list containing layers of encoder
 ``layers_generator``       ``list``            list containing layers of generator

**output**
------------------------------------------------------------------------------------------------------------------------------------------------

 ``y``                      ``numpy.ndarray``   predicted values
========================== =================== =================================================================================================

.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    X = np.array([1] * 100).reshape(1, -1, 1)

    primitive = load_primitive('orion.primitives.vae.VAE',
        arguments={"X": X, "y": X, "input_shape":(100, 1), "output_shape":(100, 1),
                   "validation_split": 0, "batch_size": 1, "epochs": 5})

    primitive.fit()
    pred = primitive.produce(X=X)
    pred.mean()
