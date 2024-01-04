# -*- coding: utf-8 -*-
"""
Implementation of VAE.

References
    - https://stackoverflow.com/questions/63987125/keras-lstm-vae-variational-autoencoder-for-time-series-anamoly-detection # noqa: E501
    - https://github.com/twairball/keras_lstm_vae/blob/master/lstm_vae/vae.py
"""
import logging
import tempfile

import numpy as np
import tensorflow as tf
from mlstars.utils import import_object
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

LOGGER = logging.getLogger(__name__)


def build_layer(layer: dict, hyperparameters: dict):
    layer_class = import_object(layer['class'])
    layer_kwargs = layer['parameters'].copy()
    # TODO: Upgrade to using tf.keras.layers.Wrapper in mlprimitives.
    if issubclass(layer_class, tf.keras.layers.Wrapper):
        layer_kwargs['layer'] = build_layer(layer_kwargs['layer'], hyperparameters)
    for key, value in layer_kwargs.items():
        if isinstance(value, str):
            layer_kwargs[key] = hyperparameters.get(value, value)
    return layer_class(**layer_kwargs)


class VAE(object):
    """VAE model for time series reconstruction.

    Args:
        layers_encoder (list):
            List containing layers of encoder.
        layers_generator (list):
            List containing layers of generator.
        optimizer (str):
            String denoting the keras optimizer.
        input_shape (tuple):
            Optional. Tuple denoting the shape of an input sample.
        output_shape (tuple):
            Optional. Tuple denoting the shape of an output sample.
        learning_rate (float):
            Optional. Float denoting the learning rate of the optimizer. Default 0.005.
        epochs (int):
            Optional. Integer denoting the number of epochs. Default 2000.
        latent_dim (int):
            Optional. Integer denoting dimension of latent space. Default 20.
        batch_size (int):
            Integer denoting the batch size. Default 64.
        shuffle (bool):
            Optional. Whether to shuffle the training data before each epoch Default True.
        verbose (int):
            Verbosity mode where 0 = silent, 1 = progress bar, 2 = one line per epoch. Default 0.
        callbacks (list):
            Optional. List of keras callbacks to apply during evaluation.
        validation_split (float):
            Optional. Fraction of the training data to be used as validation data. Default 0.
        hyperparameters (dictionary):
            Optional. Dictionary containing any additional inputs.
    """

    def __getstate__(self):
        networks = ['encoder', 'generator']
        modules = ['optimizer', 'vae_model', 'mse_loss', 'fit_history']

        state = self.__dict__.copy()

        for module in modules:
            del state[module]

        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                tf.keras.models.save_model(state.pop(network), fd.name, overwrite=True)

                state[network + '_str'] = fd.read()

        return state

    def __setstate__(self, state):
        networks = ['encoder', 'generator']
        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                fd.write(state.pop(network + '_str'))
                fd.flush()

                state[network] = tf.keras.models.load_model(fd.name)

        self.__dict__ = state

    def _build_model(self, hyperparameters, layers, input_shape):
        x = Input(shape=input_shape)
        model = tf.keras.models.Sequential()

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        return Model(x, model(x))

    def _setdefault(self, kwargs, key, value):
        if key in kwargs:
            return

        if key in self.hyperparameters and self.hyperparameters[key] is None:
            kwargs[key] = value

    def __init__(self, layers_encoder: list, layers_generator: list, optimizer: str,
                 input_shape: tuple = None, output_shape: tuple = None, latent_dim: int = 20,
                 learning_rate: float = 0.001, epochs: int = 35, batch_size: int = 64,
                 shuffle: bool = True, verbose: bool = True, callbacks: tuple = tuple(),
                 validation_split: float = 0.0, **hyperparameters):

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_shape = (latent_dim,)
        self.output_shape = output_shape

        self.layers_encoder = layers_encoder
        self.layers_generator = layers_generator

        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = import_object(optimizer)(learning_rate)
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.shuffle = shuffle
        self.verbose = verbose
        self.hyperparameters = hyperparameters
        self.validation_split = validation_split
        for callback in callbacks:
            callback['class'] = import_object(callback['class'])
        self.callbacks = callbacks
        self._fitted = False
        self.fit_history = None

    def _augment_hyperparameters(self, X, y, kwargs):
        input_shape = np.asarray(X)[0].shape
        output_shape = np.asarray(y)[0].shape
        self.input_shape = self.input_shape or input_shape
        self.output_shape = self.output_shape or output_shape

        self._setdefault(kwargs, 'length', input_shape[0])
        self._setdefault(kwargs, 'output_dim', self.output_shape[-1])
        return kwargs

    def _set_shapes(self):
        self.encoder_input_shape = self.input_shape
        self.generator_input_shape = self.latent_shape

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch_size, self.latent_dim), mean=0., stddev=1.)
        return z_mean + z_log_sigma * epsilon

    def _vae_loss(self, y_true, y_pred, z_log_sigma, z_mean):
        rc_loss = self.mse_loss(y_true, y_pred)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = rc_loss + kl_loss
        return loss

    def _build_vae(self, **kwargs):
        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)

        self.encoder = self._build_model(
            hyperparameters, self.layers_encoder, self.encoder_input_shape)
        self.generator = self._build_model(
            hyperparameters, self.layers_generator, self.generator_input_shape)

        x = Input(shape=self.input_shape)
        y = Input(shape=self.output_shape)

        # Sample latent vector.
        h = self.encoder(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim)(h)
        z_log_sigma = tf.keras.layers.Dense(self.latent_dim)(h)
        z = tf.keras.layers.Lambda(self._sampling)([z_mean, z_log_sigma])

        y_ = self.generator(z)

        self.vae_model = Model([x, y], y_)
        self.vae_model.add_loss(self._vae_loss(y, y_, z_log_sigma, z_mean))
        self.vae_model.compile(optimizer=self.optimizer)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the model.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.
            y (ndarray):
                N-dimensional array containing the output sequences we want to reconstruct.
        """
        # Reconstruct the same input.
        if y is None:
            y = X.copy()

        if not self._fitted:
            self._augment_hyperparameters(X, y, kwargs)
            self._set_shapes()
            self._build_vae(**kwargs)

        callbacks = [
            callback['class'](**callback.get('args', dict()))
            for callback in self.callbacks
        ]

        self.fit_history = self.vae_model.fit((X, y),
                                              batch_size=self.batch_size,
                                              epochs=self.epochs,
                                              shuffle=self.shuffle,
                                              verbose=self.verbose,
                                              callbacks=callbacks,
                                              validation_split=self.validation_split,
                                              )
        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple:
        """Predict values using the initialized object.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.

        Returns:
            ndarray:
                N-dimensional array containing the reconstructions for each input sequence.
        """
        dummy = np.empty(shape=(X.shape[0], *self.output_shape))
        return self.vae_model.predict((X, dummy))
