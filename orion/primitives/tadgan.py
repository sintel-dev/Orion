# -*- coding: utf-8 -*-

import logging
import math
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from mlprimitives.adapters.keras import build_layer
from mlprimitives.utils import import_object
from scipy import stats
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential

from orion.primitives.timeseries_errors import reconstruction_errors

LOGGER = logging.getLogger(__name__)

tf.keras.backend.set_floatx('float64')

class TadGAN(tf.keras.Model):
    """TadGAN class"""

    def __getstate__(self):
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        modules = ['optimizer', 'critic_x_model', 'critic_z_model', 'encoder_generator_model']

        state = self.__dict__.copy()

        for module in modules:
            del state[module]

        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                tf.keras.models.save_model(state.pop(network), fd.name, overwrite=True)

                state[network + '_str'] = fd.read()

        return state

    def __setstate__(self, state):
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                fd.write(state.pop(network + '_str'))
                fd.flush()

                state[network] = tf.keras.models.load_model(fd.name)

        self.__dict__ = state

    def _build_model(self, hyperparameters, layers, input_shape, name):
        x = Input(shape=input_shape)
        model = Sequential(name=name)

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        return tf.keras.Model(x, model(x), name=name)

    def _wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def __init__(self, shape, encoder_input_shape, generator_input_shape, critic_x_input_shape,
                 critic_z_input_shape, layers_encoder, layers_generator, layers_critic_x,
                 layers_critic_z, optimizer, learning_rate=0.0005, epochs=50, latent_dim=20,
                 batch_size=64, iterations_critic=5, validation_split=0.2, callbacks=tuple(),
                 shuffle=True, verbose=True, detailed=False, **hyperparameters):
        """Initialize the TadGAN object.

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
            validation_split (float): Optional. Float between 0 and 1. Fraction of the training
                data to be used as validation data. Default 0.2.
            callacks (tuple):
                Optional. List of callbacks to apply during training.
            verbose (int or bool):
                Optional. Verbosity mode where 0 = silent, 1 = progress bar,
                2 = one line per epoch. Default False.
            detailed (bool):
                Optional. Whether to output all loss values in verbose mode.
            hyperparameters (dictionary):
                Optional. Dictionary containing any additional inputs.
        """
        super(TadGAN, self).__init__()

        self.shape = shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.iterations_critic = iterations_critic
        self.epochs = epochs
        self.shuffle = shuffle
        self.verbose = verbose
        self.detailed = detailed
        self.validation_split = validation_split
        self.hyperparameters = hyperparameters

        self.optimizer = import_object(optimizer)(learning_rate)

        for callback in callbacks:
            callback['class'] = import_object(callback['class'])

        self.callbacks = callbacks

        self.encoder_input_shape = encoder_input_shape
        self.generator_input_shape = generator_input_shape
        self.critic_x_input_shape = critic_x_input_shape
        self.critic_z_input_shape = critic_z_input_shape

        self.layers_encoder, self.layers_generator = layers_encoder, layers_generator
        self.layers_critic_x, self.layers_critic_z = layers_critic_x, layers_critic_z

        self.encoder = self._build_model(hyperparameters, self.layers_encoder,
                                         self.encoder_input_shape, name='encoder')
        self.generator = self._build_model(hyperparameters, self.layers_generator,
                                           self.generator_input_shape, name='generator')
        self.critic_x = self._build_model(hyperparameters, self.layers_critic_x,
                                          self.critic_x_input_shape, name='critic_x')
        self.critic_z = self._build_model(hyperparameters, self.layers_critic_z,
                                          self.critic_z_input_shape, name='critic_z')

        self.gp_weight = 10
        self.cycle_weight = 10

        self.compile()

    def compile(self, **kwargs):
        super(TadGAN, self).compile(**kwargs)

        # losses
        self.encoder_generator_loss_fn = self._wasserstein_loss
        self.critic_x_loss_fn = self._wasserstein_loss
        self.critic_z_loss_fn = self._wasserstein_loss
        self.cycle_loss_fn = tf.keras.losses.MeanSquaredError()

    def get_output(self, cx_loss, cz_loss, eg_loss):
        if self.detailed:
            output = {
                # format Cx loss
                "Cx_loss": cx_loss[0],
                "Cx_real": cx_loss[1],
                "Cx_fake": cx_loss[2],
                "Cx_gp": cx_loss[3],
                # format Cz loss
                "Cz_loss": cz_loss[0],
                "Cz_real": cz_loss[1],
                "Cz_fake": cz_loss[2],
                "Cz_gp": cz_loss[3],
                # format EG loss
                "EG_loss": eg_loss[0],
                "EG_x": eg_loss[1],
                "EG_z": eg_loss[2],
                "EG_mse": eg_loss[3]
            }

        else:
            output = {
                "Cx_loss": cx_loss[0],
                "Cz_loss": cz_loss[0],
                "EG_loss": eg_loss[0]
            }

        return output

    def gradient_penalty(self, critic, batch_size, inputs):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated signal
        and added to the discriminator loss.

        Args:
            inputs[0] x     original input
            inputs[1] x_    predicted input
        """
        # Get the interpolated signal
        alpha = tf.random.uniform([batch_size, 1, 1], dtype=tf.float64)
        interpolated = (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the critic output for this interpolated signal.
            pred = critic(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=np.arange(1, len(grads.shape))))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, X):
        if isinstance(X, tuple):
            X = X[0]

        batch_size = tf.shape(X)[0]
        mini_batch_size = batch_size // self.iterations_critic

        fake = tf.ones((mini_batch_size, 1), dtype=tf.float64)
        valid = -tf.ones((mini_batch_size, 1), dtype=tf.float64)

        batch_cx_loss = []
        batch_cz_loss = []

        # Train the critics
        for j in range(self.iterations_critic):
            x = X[j * mini_batch_size: (j + 1) * mini_batch_size]
            z = tf.random.normal(shape=(mini_batch_size, self.latent_dim, 1), dtype=tf.float64)

            with tf.GradientTape(persistent=True) as tape:
                x_ = self.generator(z, training=True)  # z -> x
                z_ = self.encoder(x, training=True)  # x -> z

                cx_real = self.critic_x(x, training=True)
                cx_fake = self.critic_x(x_, training=True)

                cz_real = self.critic_z(z, training=True)
                cz_fake = self.critic_z(z_, training=True)

                # Calculate loss real
                cx_real_cost = self.critic_x_loss_fn(cx_real, valid)
                cz_real_cost = self.critic_z_loss_fn(cz_real, valid)

                # Calculate loss fake
                cx_fake_cost = self.critic_x_loss_fn(cx_fake, fake)
                cz_fake_cost = self.critic_z_loss_fn(cz_fake, fake)

                # Calculate the gradient penalty
                cx_gp = self.gradient_penalty(self.critic_x, mini_batch_size, (x, x_))
                cz_gp = self.gradient_penalty(self.critic_z, mini_batch_size, (z, z_))

                # Add the gradient penalty to the original loss
                cx_loss = cx_real_cost + cx_fake_cost + self.gp_weight * cx_gp
                cz_loss = cz_real_cost + cz_fake_cost + self.gp_weight * cz_gp

            # Get the gradients for the critics
            cx_grads = tape.gradient(cx_loss, self.critic_x.trainable_weights)
            cz_grads = tape.gradient(cz_loss, self.critic_z.trainable_weights)

            # Update the weights of the critics
            self.optimizer.apply_gradients(zip(cx_grads, self.critic_x.trainable_weights))
            self.optimizer.apply_gradients(zip(cz_grads, self.critic_z.trainable_weights))

            # Record loss
            batch_cx_loss.append([cx_loss, cx_real_cost, cx_fake_cost, cx_gp])
            batch_cz_loss.append([cz_loss, cz_real_cost, cz_fake_cost, cz_gp])

        with tf.GradientTape() as tape:
            x_ = self.generator(z, training=True)  # z -> x
            z_ = self.encoder(x, training=True)  # x -> z
            cycled_x = self.generator(z_, training=True)  # x -> z -> x

            cx_fake = self.critic_x(x_, training=True)
            cz_fake = self.critic_z(z_, training=True)

            # Generator/encoder loss
            x_cost = self.encoder_generator_loss_fn(cx_fake, valid)
            z_cost = self.encoder_generator_loss_fn(cz_fake, valid)

            # Generator/encoder cycle loss
            cycle_loss = self.cycle_loss_fn(cycled_x, x)

            # Total loss
            eg_loss = x_cost + z_cost + self.cycle_weight * cycle_loss

        # Get the gradients for the encoder/generator
        encoder_generator_grads = tape.gradient(eg_loss,
                                                self.encoder.trainable_variables +
                                                self.generator.trainable_variables)

        # Update the weights of the encoder/generator
        self.optimizer.apply_gradients(
            zip(encoder_generator_grads, self.encoder.trainable_variables +
                self.generator.trainable_variables))

        batch_cx_loss = np.mean(np.array(batch_cx_loss), axis=1)
        batch_cz_loss = np.mean(np.array(batch_cz_loss), axis=1)
        batch_eg_loss = (eg_loss, x_cost, z_cost, cycle_loss)

        return self.get_output(batch_cx_loss, batch_cz_loss, batch_eg_loss)

    def test_step(self, x):
        if isinstance(x, tuple):
            x = x[0]

        batch_size = tf.shape(x)[0]

        # Prepare data
        fake = tf.ones((batch_size, 1), dtype=tf.float64)
        valid = -tf.ones((batch_size, 1), dtype=tf.float64)

        z = tf.random.normal(shape=(batch_size, self.latent_dim, 1), dtype=tf.float64)

        x_ = self.generator(z)  # z -> x
        z_ = self.encoder(x)  # x -> z
        cycled_x = self.generator(z_)  # x -> z -> x

        cx_real = self.critic_x(x)
        cx_fake = self.critic_x(x_)

        cz_real = self.critic_z(z)
        cz_fake = self.critic_z(z_)

        # Calculate losses
        cx_real_cost = self.critic_x_loss_fn(cx_real, valid)
        cz_real_cost = self.critic_z_loss_fn(cz_real, valid)

        cx_fake_cost = self.critic_x_loss_fn(cx_fake, fake)
        cz_fake_cost = self.critic_z_loss_fn(cz_fake, fake)

        cx_gp = self.gradient_penalty(self.critic_x, batch_size, (x, x_))
        cz_gp = self.gradient_penalty(self.critic_z, batch_size, (z, z_))

        x_cost = self.encoder_generator_loss_fn(cx_fake, valid)
        z_cost = self.encoder_generator_loss_fn(cz_fake, valid)

        cycle_loss = self.cycle_loss_fn(cycled_x, x)

        # Loss combination
        cx_loss = cx_real_cost + cx_fake_cost + cx_gp * self.gp_weight
        cz_loss = cz_real_cost + cz_fake_cost + cz_gp * self.gp_weight
        eg_loss = x_cost + z_cost + self.cycle_weight * cycle_loss

        return self.get_output((cx_loss, cx_real_cost, cx_fake_cost, cx_gp),
                               (cz_loss, cz_real_cost, cz_fake_cost, cz_gp),
                               (eg_loss, x_cost, z_cost, cycle_loss))

    def call(self, X):
        z_ = self.encoder(X)
        y_hat = self.generator(z_)
        critic = self.critic_x(X)

        return y_hat, critic

    def fit(self, X, **kwargs):
        """Fit the TadGAN.

        Args:
            X (ndarray):
                N-dimensional array containing the input training sequences for the model.
        """
        if self.validation_split > 0:
            valid_length = round(len(X) * self.validation_split)
            train = X[:-valid_length].copy()
            valid = X[-valid_length:].copy()

            valid = valid.astype(np.float)
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

        train = train.astype(np.float)
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
