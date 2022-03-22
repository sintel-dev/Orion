# -*- coding: utf-8 -*-
"""Implementation of AE with Regressor."""
import logging
import tempfile

import numpy as np
import tensorflow as tf
from mlprimitives.utils import import_object
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from orion.primitives.timeseries_errors import reconstruction_errors
from orion.primitives.timeseries_errors import regression_errors

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


class AEReg(object):
    """AEReg model for time series reconstruction and regression.

    Args:
        layers_encoder (list):
            List containing layers of encoder.
        layers_generator (list):
            List containing layers of generator.
        optimizer (str):
            String denoting the keras optimizer.
        input_shape (tuple):
            Optional. Tuple denoting the shape of an input sample.
        learning_rate (float):
            Optional. Float denoting the learning rate of the optimizer. Default 0.005.
        epochs (int):
            Optional. Integer denoting the number of epochs. Default 2000.
        latent_dim (int):
            Optional. Integer denoting dimension of latent space. Default 20.
        batch_size (int):
            Integer denoting the batch size. Default 64.
        hyperparameters (dictionary):
            Optional. Dictionary containing any additional inputs.
    """

    def __getstate__(self):
        networks = ['encoder', 'generator']
        modules = ['optimizer', 'ae_reg_model', 'mse_loss', 'fit_history']

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
                 input_shape: tuple = None, output_shape: tuple = None, latent_dim: int = 60,
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
        self.hyperparameters['latent_dim'] = self.latent_dim
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

    def _build_ae_reg(self, **kwargs):
        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)
        self.encoder = self._build_model(
            hyperparameters, self.layers_encoder, self.encoder_input_shape)
        self.generator = self._build_model(
            hyperparameters, self.layers_generator, self.generator_input_shape)

        x = Input(shape=self.input_shape)
        y = Input(shape=self.output_shape)
        t = Input(shape=(1,))

        z = self.encoder(x)
        y_ = self.generator(z)
        t_ = tf.keras.layers.Dense(1)(z)

        self.ae_reg_model = Model([x, y, t], [y_, t_])
        self.ae_reg_model.add_loss([self.mse_loss(y, y_), self.mse_loss(t, t_)])
        self.ae_reg_model.compile(optimizer=self.optimizer, loss_weights=[.5, .5])

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray, **kwargs):
        """Fit the model.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.
            y (ndarray):
                N-dimensional array containing the output sequences we want to reconstruct.
            t (ndarray):
                N-dimensional array containing the regression target.
        """
        # Reconstruct the same input.
        if y is None:
            y = X.copy()

        if not self._fitted:
            self._augment_hyperparameters(X, y, kwargs)
            self._set_shapes()
            self._build_ae_reg(**kwargs)

        callbacks = [
            callback['class'](**callback.get('args', dict()))
            for callback in self.callbacks
        ]

        self.fit_history = self.ae_reg_model.fit((X, y, t),
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
            ndarray:
                N-dimensional array containing the regression for each input sequence.
        """
        y, t = self.ae_reg_model.predict(X)
        return y, t, self.fit_history.history


def score_anomalies(y, y_hat, index, t, t_hat, t_index, comb="reg", lambda_rec=0.5):
    """Compute an array of anomaly scores.

    Anomaly scores are calculated using a combination of reconstruction error and critic score.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        index (ndarray):
            time index for each y (start position of the window)
        t (ndarray):
            Ground truth.
        t_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        t_index (ndarray):
            time index for each y (start position of the window)
        comb (str):
            Optional. How to combine critic and reconstruction error. Can be one
            of `["mult", "sum", "rec", "reg"]`. If not given, 'reg' is used.
        lambda_rec (float):
            Optional. Used if `comb="sum"` as a lambda weighted sum to combine
            scores. If not given, 0.5 is used.

    Returns:
        ndarray:
            Array of anomaly scores.
        ndarray:
            Array of anomaly scores.
    """

    # Regression Score
    reg_scores = regression_errors(t, t_hat, smoothing_window=0.01, smooth=True)

    # Reconstruction Score
    rec_scores, predictions = reconstruction_errors(y, y_hat)

    # Combine the two scores
    if comb == "mult":
        reg_scores = MinMaxScaler([1, 2]).fit_transform(reg_scores.reshape(-1, 1)).flatten()
        rec_scores = MinMaxScaler([1, 2]).fit_transform(rec_scores.reshape(-1, 1)).flatten()
        rec_scores = rec_scores[y.shape[1] - 1:]
        final_scores = np.multiply(reg_scores, rec_scores)
        true_index = t_index

    elif comb == "sum":
        reg_scores = MinMaxScaler([0, 1]).fit_transform(reg_scores.reshape(-1, 1)).flatten()
        rec_scores = MinMaxScaler([0, 1]).fit_transform(rec_scores.reshape(-1, 1)).flatten()
        rec_scores = rec_scores[y.shape[1] - 1:]
        final_scores = (1 - lambda_rec) * reg_scores + lambda_rec * rec_scores
        true_index = t_index

    elif comb == "rec":
        final_scores = rec_scores
        true_index = index

    elif comb == "reg":
        final_scores = reg_scores
        true_index = t_index

    else:
        raise ValueError(
            'Unknown combination specified {}, use "mult", "sum", "rec", or "reg" instead.'.format(
                comb))

    return final_scores, true_index
