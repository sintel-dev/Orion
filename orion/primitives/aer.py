# -*- coding: utf-8 -*-
import logging
import tempfile

import numpy as np
import tensorflow as tf
from mlstars.utils import import_object
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from orion.primitives.timeseries_errors import reconstruction_errors, regression_errors

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


class AER(object):
    """Autoencoder with bi-directional regression for time series anomaly detection.

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
        batch_size (int):
            Integer denoting the batch size. Default 64.
        hyperparameters (dictionary):
            Optional. Dictionary containing any additional inputs.
    """

    def __getstate__(self):
        networks = ['encoder', 'decoder']
        modules = ['aer_model', 'optimizer', 'fit_history']

        state = self.__dict__.copy()

        for module in modules:
            del state[module]

        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                tf.keras.models.save_model(state.pop(network), fd.name, overwrite=True)

                state[network + '_str'] = fd.read()

        return state

    def __setstate__(self, state):
        networks = ['encoder', 'decoder']
        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                fd.write(state.pop(network + '_str'))
                fd.flush()

                state[network] = tf.keras.models.load_model(fd.name)

        self.__dict__ = state

    @staticmethod
    def _build_model(hyperparameters, layers, input_shape):
        x = Input(shape=input_shape)
        model = tf.keras.models.Sequential()

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        return Model(x, model(x))

    def __init__(self, layers_encoder: list, layers_decoder: list,
                 optimizer: str, learning_rate: float = 0.001,
                 epochs: int = 35, batch_size: int = 64, shuffle: bool = True,
                 verbose: bool = True, callbacks: tuple = tuple(), reg_ratio: float = 0.5,
                 lstm_units: int = 30, validation_split: float = 0.0, **hyperparameters):

        self.layers_encoder = layers_encoder
        self.layers_decoder = layers_decoder
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_ratio = reg_ratio
        self.optimizer = import_object(optimizer)(learning_rate)
        self.lstm_units = lstm_units

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
        # Infers the shape from the inputs.
        self.input_shape = np.asarray(X)[0].shape
        self.latent_shape = (self.lstm_units * 2,)

        kwargs['repeat_vector_n'] = self.input_shape[0] + 2
        kwargs['lstm_units'] = self.lstm_units
        return kwargs

    def _build_aer(self, **kwargs):
        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)

        # Build encoder and decoder.
        self.encoder = self._build_model(hyperparameters, self.layers_encoder,
                                         self.input_shape)
        self.decoder = self._build_model(hyperparameters, self.layers_decoder,
                                         self.latent_shape)

        # Build AER Model.
        x = tf.keras.Input(self.input_shape)
        x_ = self.decoder(self.encoder(x))
        ry, y, fy = x_[:, 0], x_[:, 1:-1], x_[:, -1]

        self.aer_model = tf.keras.Model(inputs=x, outputs=[ry, y, fy])
        self.aer_model.compile(
            loss=['mse', 'mse', 'mse'],
            optimizer=self.optimizer,
            loss_weights=[self.reg_ratio / 2, 1 - self.reg_ratio, self.reg_ratio / 2],
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """Fit the model.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.
            y (ndarray):
                N-dimensional array containing the output sequences we want to reconstruct.
        """
        if y is None:
            y = X.copy()  # Reconstruct the same input.

        X = X[:, 1:-1, :]
        ry, y, fy = y[:, 0], y[:, 1:-1], y[:, -1]

        if not self._fitted:
            self._augment_hyperparameters(X, y, kwargs)
            self._build_aer(**kwargs)

        callbacks = [
            callback['class'](**callback.get('args', dict()))
            for callback in self.callbacks
        ]

        self.fit_history = self.aer_model.fit(
            X, [ry, y, fy], batch_size=self.batch_size, epochs=self.epochs,
            shuffle=self.shuffle, verbose=self.verbose, callbacks=callbacks,
            validation_split=self.validation_split
        )

        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple:
        """Predict values using the initialized object.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.

        Returns:
            ndarray:
                N-dimensional array containing the regression for each input sequence (forward).
            ndarray:
                N-dimensional array containing the reconstructions for each input sequence.
            ndarray:
                N-dimensional array containing the regression for each input sequence (reverse).
        """
        X = X[:, 1:-1, :]
        x_ = self.decoder.predict(self.encoder.predict(X))
        ry, y, fy = x_[:, 0], x_[:, 1:-1], x_[:, -1]
        return ry, y, fy


def bi_regression_errors(y: ndarray, ry_hat: ndarray, fy_hat: ndarray,
                         smoothing_window: float = 0.01, smooth: bool = True, mask: bool = True):
    """Compute an array of absolute errors comparing the forward and reverse predictions with
    the expected output.

    Anomaly scores are created in the forward and reverse directions. Scores in overlapping indices
    are averaged while scores in non-overlapping indices are taken directly from either forward or
    reverse anomaly scores.

    If smooth is True, apply EWMA to the resulting array of errors.

    Args:
        y (ndarray):
            Ground truth.
        ry_hat (ndarray):
            Predicted values (reverse).
        fy_hat (ndarray):
            Predicted values (forward).
        smoothing_window (float):
            Optional. Size of the smoothing window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed with EWMA.
            If not given, `True` is used.
        mask (bool): bool = True
            Optional. Mask anomaly score errors in the beginning.
            If not given, `True` is used.

    Returns:
        ndarray:
            Array of errors.
    """
    time_steps = len(y[0]) - 1
    mask_steps = int(smoothing_window * len(y)) if mask else 0
    ry, fy = y[:, 0], y[:, -1]

    f_scores = regression_errors(fy, fy_hat, smoothing_window=smoothing_window, smooth=smooth)
    f_scores[:mask_steps] = 0
    f_scores = np.concatenate([np.zeros(time_steps), f_scores])

    r_scores = regression_errors(ry, ry_hat, smoothing_window=smoothing_window, smooth=smooth)
    r_scores[:mask_steps] = min(r_scores)
    r_scores = np.concatenate([r_scores, np.zeros(time_steps)])

    scores = f_scores + r_scores
    scores[time_steps + mask_steps:-time_steps] /= 2
    return scores


def score_anomalies(y: ndarray, ry_hat: ndarray, y_hat: ndarray, fy_hat: ndarray,
                    smoothing_window: float = 0.01, smooth: bool = True, mask: bool = True,
                    comb: str = 'mult', lambda_rec: float = 0.5, rec_error_type: str = "dtw"):
    """Compute an array of absolute errors comparing predictions and expected output.

    If smooth is True, apply EWMA to the resulting array of errors.

    Args:
        y (ndarray):
            Ground truth.
        ry_hat (ndarray):
            Predicted values (reverse).
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        fy_hat (ndarray):
            Predicted values (forward).
        smoothing_window (float):
            Optional. Size of the smoothing window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed with EWMA.
            If not given, `True` is used.
        mask (bool): bool = True
            Optional. Mask anomaly score errors in the beginning.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. The method to compute reconstruction error. Can be one of
            `["point", "area", "dtw"]`. If not given, 'dtw' is used.
        comb (str):
            Optional. How to combine critic and reconstruction error. Can be one
            of `["mult", "sum", "rec"]`. If not given, 'mult' is used.
        lambda_rec (float):
            Optional. Used if `comb="sum"` as a lambda weighted sum to combine
            scores. If not given, 0.5 is used.

    Returns:
        ndarray:
            Array of errors.
    """

    reg_scores = bi_regression_errors(y, ry_hat, fy_hat,
                                      smoothing_window=smoothing_window,
                                      smooth=smooth,
                                      mask=mask
                                      )
    rec_scores, _ = reconstruction_errors(y[:, 1:-1], y_hat,
                                          smoothing_window=smoothing_window,
                                          smooth=smooth,
                                          rec_error_type=rec_error_type)
    mask_steps = int(smoothing_window * len(y)) if mask else 0
    rec_scores[:mask_steps] = min(rec_scores)
    rec_scores = np.concatenate([np.zeros(1), rec_scores, np.zeros(1)])

    scores = None
    if comb == "mult":
        reg_scores = MinMaxScaler([1, 2]).fit_transform(reg_scores.reshape(-1, 1)).flatten()
        rec_scores = MinMaxScaler([1, 2]).fit_transform(rec_scores.reshape(-1, 1)).flatten()
        scores = np.multiply(reg_scores, rec_scores)

    elif comb == "sum":
        reg_scores = MinMaxScaler([0, 1]).fit_transform(reg_scores.reshape(-1, 1)).flatten()
        rec_scores = MinMaxScaler([0, 1]).fit_transform(rec_scores.reshape(-1, 1)).flatten()
        scores = (1 - lambda_rec) * reg_scores + lambda_rec * rec_scores

    elif comb == "rec":
        scores = rec_scores

    elif comb == "reg":
        scores = reg_scores

    return scores
