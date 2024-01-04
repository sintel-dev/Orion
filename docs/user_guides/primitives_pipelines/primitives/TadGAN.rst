.. highlight:: shell

TadGAN
~~~~~~

**path**: ``orion.primitives.tadgan.TadGAN``

**description**: this is a reconstruction model, namely Generative Adversarial Networks (GAN), containing multiple neural networks and cycle consistency loss. the proposed model is described in the `related paper <https://arxiv.org/pdf/2009.07769.pdf>`__.

see `json <https://github.com/sintel-dev/Orion/tree/master/orion/primitives/jsons/orion.primitives.tadgan.TadGAN.json>`__.

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
 ``target_shape``           ``tuple``           tuple denoting the shape of an reconstructed sample
 ``optimizer``              ``str``             string (name of optimizer) or optimizer instance. Default is ``keras.optimizers.Adam``
 ``learning_rate``          ``float``           float denoting the learning rate of the optimizer. Default is 0.005
 ``latent_dim``             ``int``             integer denoting dimension of latent space. Default is 20
 ``batch_size``             ``int``             number of samples per gradient update. Default is 64
 ``iterations_critic``      ``int``             number of critic training steps per generator/encoder training steps. Default is 5
 ``layers_encoder``         ``list``            list containing layers of encoder
 ``layers_generator``       ``list``            list containing layers of generator
 ``layers_critic_x``        ``list``            list containing layers of ``critic_x``
 ``layers_critic_z``        ``list``            list containing layers of ``critic_z``

**output**
------------------------------------------------------------------------------------------------------------------------------------------------

 ``y``                     ``numpy.ndarray``    n-dimensional array containing the reconstructions for each input sequence
 ``critic``                ``numpy.ndarray``    n-dimensional array containing the critic score for each input sequence
========================== =================== =================================================================================================


.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    X = np.array([1] * 100).reshape(1, -1, 1)
    y = X[:,:, [0]] # signal to reconstruct from X (channel 0)
    primitive = load_primitive('orion.primitives.tadgan.TadGAN', 
        arguments={"X": X, "y":X, "epochs": 5, "batch_size": 1,
                   "iterations_critic": 1})

    primitive.fit()
    y, critic = primitive.produce(X=X, y=y)

    print("average reconstructed value: {:.2f}, critic score {:.2f}".format(
        y.mean(), critic[0][0])) 
 
