# -*- coding: utf-8 -*-

import logging
from functools import partial

import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.models import Model
from mlprimitives.adapters.keras import build_layer
from mlprimitives.utils import import_object
from scipy import integrate, stats

LOGGER = logging.getLogger(__name__)


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class CycleGAN():
    """CycleGAN class"""

    def _build_model(self, hyperparameters, layers, input_shape):
        x = Input(shape=input_shape)
        model = keras.models.Sequential()

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        return Model(x, model(x))

    def _wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def _gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def __init__(self, shape, encoder_input_shape, generator_input_shape, critic_x_input_shape,
                 critic_z_input_shape, layers_encoder, layers_generator, layers_critic_x,
                 layers_critic_z, optimizer, learning_rate=0.0005, epochs=2000, latent_dim=20,
                 batch_size=64, iterations_critic=5, **hyperparameters):
        """Initialize the ARIMA object.

        Args:
            shape (tuple):
                Tuple denoting the shape of an input sample.
            encoder_input_shape (tuple):
                Shape of encoder input.
            generator_input_shape (tuple):
                Shape of generator input.
            critic_x_input_shape (tuple):
                Shape of critic_x input.
            critic_z_input_shape (tuple):
                Shape of critic_z input.
            layers_encoder (list):
                List containing layers of encoder.
            layers_generator (list):
                List containing layers of generator.
            layers_critic_x (list):
                List containing layers of critic_x.
            layers_critic_z (list):
                List containing layers of critic_z.
            optimizer (str):
                String denoting the keras optimizer.
            learning_rate (float):
                Optional. Float denoting the learning rate of the optimizer. Default 0.005.
            epochs (int):
                Optional. Integer denoting the number of epochs. Default 2000.
            latent_dim (int):
                Optional. Integer denoting dimension of latent space. Default 20.
            batch_size (int):
                Integer denoting the batch size. Default 64.
            iterations_critic (int):
                Optional. Integer denoting the number of critic training steps per one
                Generator/Encoder training step. Default 5.
            hyperparameters (dictionary):
                Optional. Dictionary containing any additional inputs.
        """

        self.shape = shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.iterations_critic = iterations_critic
        self.epochs = epochs
        self.hyperparameters = hyperparameters

        self.encoder_input_shape = encoder_input_shape
        self.generator_input_shape = generator_input_shape
        self.critic_x_input_shape = critic_x_input_shape
        self.critic_z_input_shape = critic_z_input_shape

        self.layers_encoder, self.layers_generator = layers_encoder, layers_generator
        self.layers_critic_x, self.layers_critic_z = layers_critic_x, layers_critic_z

        self.optimizer = import_object(optimizer)(learning_rate)

    def _build_cyclegan(self, **kwargs):

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)

        self.encoder = self._build_model(hyperparameters, self.layers_encoder,
                                         self.encoder_input_shape)
        self.generator = self._build_model(hyperparameters, self.layers_generator,
                                           self.generator_input_shape)
        self.critic_x = self._build_model(hyperparameters, self.layers_critic_x,
                                          self.critic_x_input_shape)
        self.critic_z = self._build_model(hyperparameters, self.layers_critic_z,
                                          self.critic_z_input_shape)

        self.generator.trainable = False
        self.encoder.trainable = False

        z = Input(shape=(self.latent_dim, 1))
        x = Input(shape=self.shape)
        x_ = self.generator(z)
        z_ = self.encoder(x)
        fake_x = self.critic_x(x_)
        valid_x = self.critic_x(x)
        interpolated_x = RandomWeightedAverage()([x, x_])

        validity_interpolated_x = self.critic_x(interpolated_x)
        partial_gp_loss_x = partial(self._gradient_penalty_loss, averaged_samples=interpolated_x)
        partial_gp_loss_x.__name__ = 'gradient_penalty'
        self.critic_x_model = Model(inputs=[x, z], outputs=[valid_x, fake_x,
                                                            validity_interpolated_x])
        self.critic_x_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss,
                                          partial_gp_loss_x], optimizer=self.optimizer,
                                    loss_weights=[1, 1, 5])

        fake_z = self.critic_z(z_)
        valid_z = self.critic_z(z)
        interpolated_z = RandomWeightedAverage()([z, z_])
        validity_interpolated_z = self.critic_z(interpolated_z)
        partial_gp_loss_z = partial(self._gradient_penalty_loss, averaged_samples=interpolated_z)
        partial_gp_loss_z.__name__ = 'gradient_penalty'
        self.critic_z_model = Model(inputs=[x, z], outputs=[valid_z, fake_z,
                                                            validity_interpolated_z])
        self.critic_z_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss,
                                          partial_gp_loss_z], optimizer=self.optimizer,
                                    loss_weights=[1, 1, 10])

        self.critic_x.trainable = False
        self.critic_z.trainable = False
        self.generator.trainable = True
        self.encoder.trainable = True

        z_gen = Input(shape=(self.latent_dim, 1))
        x_gen_ = self.generator(z_gen)
        x_gen = Input(shape=self.shape)
        z_gen_ = self.encoder(x_gen)
        x_gen_rec = self.generator(z_gen_)
        fake_gen_x = self.critic_x(x_gen_)
        fake_gen_z = self.critic_z(z_gen_)

        self.encoder_generator_model = Model([x_gen, z_gen], [fake_gen_x, fake_gen_z, x_gen_rec])
        self.encoder_generator_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss,
                                                   'mse'], optimizer=self.optimizer,
                                             loss_weights=[1, 1, 50])

    def _fit(self, X):
        fake = np.ones((self.batch_size, 1))
        valid = -np.ones((self.batch_size, 1))
        delta = np.ones((self.batch_size, 1)) * 10

        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(X)
            epoch_g_loss = []
            epoch_cx_loss = []
            epoch_cz_loss = []

            minibatches_size = self.batch_size * self.iterations_critic
            num_minibatches = int(X.shape[0] // minibatches_size)

            for i in range(num_minibatches):
                minibatch = X[i * minibatches_size: (i + 1) * minibatches_size]

                for j in range(self.iterations_critic):
                    x = minibatch[j * self.batch_size: (j + 1) * self.batch_size]
                    z = np.random.normal(size=(self.batch_size, self.latent_dim, 1))
                    epoch_cx_loss.append(
                        self.critic_x_model.train_on_batch([x, z], [valid, fake, delta]))
                    epoch_cz_loss.append(
                        self.critic_z_model.train_on_batch([x, z], [valid, fake, delta]))

                epoch_g_loss.append(
                    self.encoder_generator_model.train_on_batch([x, z], [valid, valid, x]))

            cx_loss = np.mean(np.array(epoch_cx_loss), axis=0)
            cz_loss = np.mean(np.array(epoch_cz_loss), axis=0)
            g_loss = np.mean(np.array(epoch_g_loss), axis=0)
            print('Epoch: {}/{}, [Dx loss: {}] [Dz loss: {}] [G loss: {}]'.format(
                epoch, self.epochs, cx_loss, cz_loss, g_loss))

    def fit(self, X, **kwargs):
        """Fit the CycleGAN.

        Args:
            X (ndarray):
                N-dimensional array containing the input training sequences for the model.
        """
        self._build_cyclegan(**kwargs)
        X = X.reshape((-1, self.shape[0], 1))
        self._fit(X)

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
        X = X.reshape((-1, self.shape[0], 1))
        z_ = self.encoder.predict(X)
        y_hat = self.generator.predict(z_)
        critic = self.critic_x.predict(X)

        return y_hat, critic


def score_anomalies(y, y_hat, critic, score_window=10, smooth_window=200):
    """Compute an array of anomaly scores.

    Anomaly scores are calculated using a combination of reconstruction error and critic score.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        critic (ndarray):
            Critic score. Each timestamp has multiple critic scores.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        smooth_window (int):
            Optional. Size of window over which smoothing is applied.
            If not given, 200 is used.

    Returns:
        ndarray:
            Array of anomaly scores.
    """

    true = [item[0] for item in y.reshape((y.shape[0], -1))]

    for item in y[-1][1:]:
        true.extend(item)

    critic_extended = list()
    for c in critic:
        critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())

    predictions = []
    critic_kde_max = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + (y_hat.shape[0] - 1)
    y_hat = np.asarray(y_hat)
    critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))

    for i in range(num_errors):
        intermediate = []
        critic_intermediate = []

        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])
            critic_intermediate.append(critic_extended[i - j, j])

        if intermediate:
            predictions.append(np.median(np.asarray(intermediate)))

            if len(critic_intermediate) > 1:
                discr_intermediate = np.asarray(critic_intermediate)
                try:
                    critic_kde_max.append(discr_intermediate[np.argmax(
                        stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
                except np.linalg.LinAlgError:
                    critic_kde_max.append(np.median(discr_intermediate))
            else:
                critic_kde_max.append(np.median(np.asarray(critic_intermediate)))

    predictions = np.asarray(predictions)
    score_window_min = score_window // 2

    pd_true = pd.Series(np.asarray(true).flatten())
    pd_pred = pd.Series(np.asarray(predictions).flatten())
    score_measure_true = pd_true.rolling(score_window, center=True, min_periods=score_window_min)\
        .apply(integrate.trapz)
    score_measure_pred = pd_pred.rolling(score_window, center=True, min_periods=score_window_min)\
        .apply(integrate.trapz)
    scores = abs(score_measure_true - score_measure_pred)
    scores_smoothed = pd.Series(scores).rolling(smooth_window, center=True,
                                                min_periods=smooth_window // 2,
                                                win_type='triang').mean().values

    z_score_scores = stats.zscore(scores_smoothed)
    z_score_scores_clip = np.clip(z_score_scores, a_min=0, a_max=None) + 1

    critic_kde_max = np.asarray(critic_kde_max)
    l_quantile = np.quantile(critic_kde_max, 0.25)
    u_quantile = np.quantile(critic_kde_max, 0.75)
    in_range = np.logical_and(critic_kde_max >= l_quantile, critic_kde_max <= u_quantile)
    critic_mean = np.mean(critic_kde_max[in_range])
    critic_std = np.std(critic_kde_max)

    z_score_critic = np.absolute((np.asarray(critic_kde_max) - critic_mean) / critic_std) + 1
    z_score_critic = pd.Series(z_score_critic).rolling(
        100, center=True, min_periods=50).mean().values

    return np.multiply(z_score_scores_clip, z_score_critic)
