# -*- coding: utf-8 -*-

import logging
import math
import tempfile
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from mlprimitives.utils import import_object
from scipy import stats
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.losses import mean_squared_error

from orion.primitives.timeseries_errors import reconstruction_errors

LOGGER = logging.getLogger(__name__)
tf.keras.backend.set_floatx('float32')


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


class TadGAN(tf.keras.Model):
    """
    encoder_input_shape (tuple): Shape of encoder input.
    generator_input_shape (tuple): Shape of generator input.
    critic_x_input_shape (tuple): Shape of critic_x input.
    critic_z_input_shape (tuple): Shape of critic_z input.
    """

    def __getstate__(self) -> None:
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        modules = ['optimizer', 'critic_x_model', 'critic_z_model', 'encoder_generator_model']

        state = self.__dict__.copy()

        for module in modules:
            del state[module]

        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                tf.keras.models.save_model(state.pop(network), fd.name, overwrite=True)

                state[network + '_str'] = fd.read()

        return state

    def __setstate__(self, state: dict) -> None:
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                fd.write(state.pop(network + '_str'))
                fd.flush()

                state[network] = tf.keras.models.load_model(fd.name)

        self.__dict__ = state

    @classmethod
    def _build_model(cls, hyperparameters: dict, layers: list, input_shape: tuple, name: str):
        x = Input(shape=input_shape)
        model = Sequential(name=name)

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        return Model(x, model(x))

    def _setdefault(self, kwargs: dict, key: str, value: any):
        if key in kwargs:
            return

        if key in self.hyperparameters and self.hyperparameters[key] is None:
            kwargs[key] = value

    @classmethod
    def _wasserstein_loss(cls, y_true, y_pred):
        return K.mean(y_true * y_pred)

    @classmethod
    def _gradient_penalty_loss(cls, real, fake, critic):
        """Implementation of gradient penalty loss.

        References:
            https://keras.io/examples/generative/wgan_gp/
        """

        # Random Weighted Average
        batch_size = real.shape[0]
        alpha = tf.random.uniform([batch_size, 1, 1], dtype=tf.float32)
        interpolated = (alpha * real) + ((1 - alpha) * fake)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            validity_interpolated = critic(interpolated)
        gradients = gp_tape.gradient(validity_interpolated, [interpolated])[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def __init__(self, layers_encoder: list, layers_generator: list, layers_critic_x: list,
                 layers_critic_z: list, optimizer: str, input_shape: tuple = (100, 1),
                 target_shape: tuple = (100, 1), latent_dim: int = 20,
                 learning_rate: float = 0.0005, epochs: int = 2000,
                 batch_size: int = 64, iterations_critic: int = 5,
                 shuffle: bool = True, callbacks: tuple = tuple(), validation_ratio: float = 0.2,
                 detailed: bool = True, verbose: Union[int, bool] = True,
                 **hyperparameters):
        """Tensorflow 2.x TadGAN model for time series reconstruction.

        Args:
            layers_encoder: list containing layers of encoder
            layers_generator: list containing layers of generator
            layers_critic_x: list containing layers of critic_x
            layers_critic_z: list containing layers of critic_z
            optimizer: string denoting the tf.keras optimizer
            input_shape: tuple denoting the shape of an input sample
            target_shape: tuple denoting the shape of an output sample
            latent_dim: integer denoting dimension of latent space
            learning_rate: float denoting the learning rate of the optimizer
            epochs: integer denoting the number of epochs
            batch_size: integer denoting the batch size
            iterations_critic: integer denoting the number of critic training steps per one
                Generator/Encoder training step.
            **hyperparameters: dictionary containing any additional inputs
        """
        super(TadGAN, self).__init__()

        self.shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.latent_shape = (latent_dim, 1)
        self.target_shape = target_shape
        self.iterations_critic = iterations_critic

        self.layers_encoder, self.layers_generator = layers_encoder, layers_generator
        self.layers_critic_x, self.layers_critic_z = layers_critic_x, layers_critic_z

        self.encoder_input_shape = self.shape
        self.generator_input_shape = self.latent_shape
        self.critic_x_input_shape = self.target_shape
        self.critic_z_input_shape = self.latent_shape
        self.generator_reshape_dim = self.shape[0] // 2
        self.generator_reshape_shape = (self.generator_reshape_dim, 1)
        self.encoder_reshape_shape = self.latent_shape

        self.optimizer = import_object(optimizer)(learning_rate)

        self.hyperparameters = hyperparameters

        self.encoder = self._build_model(hyperparameters, self.layers_encoder,
                                         self.encoder_input_shape, name='encoder')
        self.generator = self._build_model(hyperparameters, self.layers_generator,
                                           self.generator_input_shape, name='generator')
        self.critic_x = self._build_model(hyperparameters, self.layers_critic_x,
                                          self.critic_x_input_shape, name='critic_x')
        self.critic_z = self._build_model(hyperparameters, self.layers_critic_z,
                                          self.critic_z_input_shape, name='critic_z')

        # Training parameters
        self.shuffle = shuffle
        self.validation_ratio = validation_ratio
        for callback in callbacks:
            callback['class'] = import_object(callback['class'])
        self.callbacks = callbacks
        self.verbose = verbose
        self.detailed = detailed

        self.compile()

    def call(self, inputs, training=None, mask=None):
        X = inputs
        z_ = self.encoder(X)
        y_hat = self.generator(z_)
        critic = self.critic_x(X)
        return y_hat, critic

    def compile(self, **kwargs):
        super(TadGAN, self).compile(**kwargs)

    def _format_losses(self, losses: list) -> dict:
        loss_names = [
            ['cx_loss', 'cx_real', 'cx_fake', 'cx_gp'],
            ['cz_loss', 'cz_real', 'cz_fake', 'cz_gp'],
            ['eg_loss', 'eg_x', 'eg_z', 'eg_mse']
        ]
        output = dict()
        if self.detailed:
            for i in range(len(loss_names)):
                for j in range(len(loss_names[i])):
                    output[loss_names[i][j]] = losses[i][j]
        else:
            for i in range(len(loss_names)):
                output[loss_names[i][0]] = losses[i][0]

        return output

    def train_step(self, X) -> dict:

        batch_size = tf.shape(X)[0]
        mini_batch_size = batch_size // self.iterations_critic

        fake = tf.ones((mini_batch_size, 1), dtype=tf.float32)
        valid = -tf.ones((mini_batch_size, 1), dtype=tf.float32)

        batch_g_loss, batch_cx_loss, batch_cz_loss = [], [], []

        # Train the critics
        for j in range(self.iterations_critic):
            x = X[j * mini_batch_size: (j + 1) * mini_batch_size]
            z = tf.random.normal(shape=(mini_batch_size, self.latent_dim, 1), dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Train critic x
                x_ = self.generator(z, training=True)
                cx_valid = self.critic_x(x, training=True)
                cx_fake = self.critic_x(x_, training=True)

                cx_valid_loss = self._wasserstein_loss(valid, cx_valid)
                cx_fake_loss = self._wasserstein_loss(fake, cx_fake)
                cx_gp = self._gradient_penalty_loss(x, x_, self.critic_x)
                cx_loss = cx_valid_loss + cx_fake_loss + 10 * cx_gp

                # Train Critic Z
                z_ = self.encoder(x, training=True)
                cz_valid = self.critic_z(z, training=True)
                cz_fake = self.critic_z(z_, training=True)

                cz_valid_loss = self._wasserstein_loss(valid, cz_valid)
                cz_fake_loss = self._wasserstein_loss(fake, cz_fake)
                cz_gp = self._gradient_penalty_loss(z, z_, self.critic_z)
                cz_loss = cz_valid_loss + cz_fake_loss + 10 * cz_gp

            # Get the gradients for the critics
            cx_grads = tape.gradient(cx_loss, self.critic_x.trainable_weights)
            cz_grads = tape.gradient(cz_loss, self.critic_z.trainable_weights)

            # Update the weights of the critics
            self.optimizer.apply_gradients(zip(cx_grads, self.critic_x.trainable_weights))
            self.optimizer.apply_gradients(zip(cz_grads, self.critic_z.trainable_weights))

            # Record loss
            batch_cx_loss.append([cx_loss, cx_valid_loss, cx_fake_loss, cx_gp])
            batch_cz_loss.append([cz_loss, cz_valid_loss, cz_fake_loss, cz_gp])

        # Train Encoder Generator
        with tf.GradientTape() as tape:
            x_ = self.generator(z, training=True)
            cx_fake = self.critic_x(x_, training=True)
            z_ = self.encoder(x, training=True)
            cz_fake = self.critic_z(z_, training=True)
            x_rec_ = self.generator(z_, training=True)

            # Encoder Generator Loss
            eg_x_loss = self._wasserstein_loss(valid, cx_fake)
            eg_z_loss = self._wasserstein_loss(valid, cz_fake)
            eg_mse = mean_squared_error(x, x_rec_)
            eg_loss = eg_x_loss + eg_z_loss + 10 * eg_mse

        # Get the gradients for the encoder/generator
        encoder_generator_grads = tape.gradient(eg_loss,
                                                self.encoder.trainable_variables +
                                                self.generator.trainable_variables)
        self.optimizer.apply_gradients(
            zip(encoder_generator_grads, self.encoder.trainable_variables +
                self.generator.trainable_variables))

        batch_cx_loss = np.mean(np.array(batch_cx_loss), axis=1)
        batch_cz_loss = np.mean(np.array(batch_cz_loss), axis=1)
        batch_eg_loss = (eg_loss, eg_x_loss, eg_z_loss, eg_mse)
        output = self._format_losses([batch_cx_loss, batch_cz_loss, batch_eg_loss])

        return output

    def fit(self, X, **kwargs):
        """Fit the TadGAN.
        Args:
            X (ndarray):
                N-dimensional array containing the input training sequences for the model.
        """
        if self.validation_ratio > 0:
            valid_length = round(len(X) * self.validation_ratio)
            train = X[:-valid_length].copy()
            valid = X[-valid_length:].copy()

            valid = valid.astype(np.float32)
            valid = tf.data.Dataset.from_tensor_slices(valid).shuffle(valid.shape[0])
            valid = valid.batch(self.batch_size, drop_remainder=True)

            callbacks = [
                callback['class'](**callback.get('args', dict()))
                for callback in self.callbacks
            ]

        else:
            train = X.copy()
            valid = None
            callbacks = None

        train = train.astype(np.float32)
        train = tf.data.Dataset.from_tensor_slices(train).shuffle(train.shape[0])
        train = train.batch(self.batch_size, drop_remainder=True)

        super().fit(train, validation_data=valid, epochs=self.epochs, verbose=self.verbose,
                    callbacks=callbacks, batch_size=self.batch_size,
                    shuffle=self.shuffle, **kwargs)

    def predict(self, X):
        """Predict values using the initialized object.
        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.
        Returns:
            ndarray:
                N-dimensional array containing the reconstructions for each input sequence.
            ndarray:
                N-dimensional array containing the critic scores for each input sequence.
        """
        y_hat, critic = self.call(X)

        return y_hat.numpy(), critic.numpy()


def _compute_critic_score(critics, smooth_window):
    """Compute an array of anomaly scores.

    Args:
        critics (ndarray):
            Critic values.
        smooth_window (int):
            Smooth window that will be applied to compute smooth errors.

    Returns:
        ndarray:
            Array of anomaly scores.
    """
    critics = np.asarray(critics)
    l_quantile = np.quantile(critics, 0.25)
    u_quantile = np.quantile(critics, 0.75)
    in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
    critic_mean = np.mean(critics[in_range])
    critic_std = np.std(critics)

    z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) + 1
    z_scores = pd.Series(z_scores).rolling(
        smooth_window, center=True, min_periods=smooth_window // 2).mean().values

    return z_scores


def score_anomalies(y, y_hat, critic, index, score_window=10, critic_smooth_window=None,
                    error_smooth_window=None, smooth=True, rec_error_type="point", comb="mult",
                    lambda_rec=0.5):
    """Compute an array of anomaly scores.

    Anomaly scores are calculated using a combination of reconstruction error and critic score.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        index (ndarray):
            time index for each y (start position of the window)
        critic (ndarray):
            Critic score. Each timestamp has multiple critic scores.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        critic_smooth_window (int):
            Optional. Size of window over which smoothing is applied to critic.
            If not given, 200 is used.
        error_smooth_window (int):
            Optional. Size of window over which smoothing is applied to error.
            If not given, 200 is used.
        smooth (bool):
            Optional. Indicates whether errors should be smoothed.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. The method to compute reconstruction error. Can be one of
            `["point", "area", "dtw"]`. If not given, 'point' is used.
        comb (str):
            Optional. How to combine critic and reconstruction error. Can be one
            of `["mult", "sum", "rec"]`. If not given, 'mult' is used.
        lambda_rec (float):
            Optional. Used if `comb="sum"` as a lambda weighted sum to combine
            scores. If not given, 0.5 is used.

    Returns:
        ndarray:
            Array of anomaly scores.
    """

    critic_smooth_window = critic_smooth_window or math.trunc(y.shape[0] * 0.01)
    error_smooth_window = error_smooth_window or math.trunc(y.shape[0] * 0.01)

    step_size = 1  # expected to be 1

    true_index = index  # no offset

    true = [item[0] for item in y.reshape((y.shape[0], -1))]

    for item in y[-1][1:]:
        true.extend(item)

    critic_extended = list()
    for c in critic:
        critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())

    critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))

    critic_kde_max = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)

    for i in range(num_errors):
        critic_intermediate = []

        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            critic_intermediate.append(critic_extended[i - j, j])

        if len(critic_intermediate) > 1:
            discr_intermediate = np.asarray(critic_intermediate)
            try:
                critic_kde_max.append(discr_intermediate[np.argmax(
                    stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
            except np.linalg.LinAlgError:
                critic_kde_max.append(np.median(discr_intermediate))
        else:
            critic_kde_max.append(np.median(np.asarray(critic_intermediate)))

    # Compute critic scores
    critic_scores = _compute_critic_score(critic_kde_max, critic_smooth_window)

    # Compute reconstruction scores
    rec_scores, predictions = reconstruction_errors(
        y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)

    rec_scores = stats.zscore(rec_scores)
    rec_scores = np.clip(rec_scores, a_min=0, a_max=None) + 1

    # Combine the two scores
    if comb == "mult":
        final_scores = np.multiply(critic_scores, rec_scores)

    elif comb == "sum":
        final_scores = (1 - lambda_rec) * (critic_scores - 1) + lambda_rec * (rec_scores - 1)

    elif comb == "rec":
        final_scores = rec_scores

    else:
        raise ValueError(
            'Unknown combination specified {}, use "mult", "sum", or "rec" instead.'.format(comb))

    true = [[t] for t in true]
    return final_scores, true_index, true, predictions
