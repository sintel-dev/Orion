# -*- coding: utf-8 -*-

import logging
import math
from functools import partial

import tensorflow as tf

import numpy as np
import pandas as pd
import similaritymeasures as sm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model, Sequential
from mlprimitives.adapters.keras import build_layer
from mlprimitives.utils import import_object
from scipy import integrate, stats

LOGGER = logging.getLogger(__name__)

class GAN(tf.keras.Model):
    """GAN class"""

    def _build_model(self, hyperparameters, layers, input_shape, name):
        x = Input(shape=input_shape)
        model = Sequential(name=name)

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        return tf.keras.Model(x, model(x), name=name)

    def _wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def __init__(self, encoder_input_shape, generator_input_shape, critic_x_input_shape,
                 critic_z_input_shape, layers_encoder, layers_generator, layers_critic_x,
                 layers_critic_z, latent_dim=20, iterations_critic=5, **hyperparameters):
        
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
            hyperparameters (dictionary):
                Optional. Dictionary containing any additional inputs.
        """
        super(GAN, self).__init__()
        print("I'm here ")
        self.latent_dim = latent_dim
        self.iterations_critic = iterations_critic
        self.hyperparameters = hyperparameters

        self.encoder_input_shape = (100, 1) # encoder_input_shape
        self.generator_input_shape = (20, 1) # generator_input_shape
        self.critic_x_input_shape = (100, 1) # critic_x_input_shape
        self.critic_z_input_shape = (20, 1) # critic_z_input_shape

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

        # print(self.encoder.summary())
        # print(self.generator.summary())
        # print(self.critic_x.summary())
        # print(self.critic_z.summary())

        self.gp_weight = 10
        self.cycle_weight = 10


    def compile(self, cx_optimizer, cz_optimizer, encoder_generator_optimizer,
                encoder_generator_loss_fn, critic_x_loss_fn, 
                critic_z_loss_fn, cycle_loss_fn, **kwargs):
        super(GAN, self).compile(**kwargs)

        # optimizers
        self.cx_optimizer = cx_optimizer
        self.cz_optimizer = cz_optimizer
        self.encoder_generator_optimizer = encoder_generator_optimizer

        print(self.cx_optimizer)

        # losses
        self.encoder_generator_loss_fn = encoder_generator_loss_fn
        self.critic_x_loss_fn = critic_x_loss_fn
        self.critic_z_loss_fn = critic_z_loss_fn
        self.cycle_loss_fn = cycle_loss_fn

    @tf.function
    def critic_x_gradient_penalty(self, batch_size, inputs):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.

        Args:
            inputs[0] x     original input
            inputs[1] x_    predicted input
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1])
        interpolated = (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic_x(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=np.arange(1, len(grads.shape))))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    @tf.function
    def critic_z_gradient_penalty(self, batch_size, inputs):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.

        Args:
            inputs[0] x     original input
            inputs[1] x_    predicted input
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1])
        interpolated = (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic_z(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=np.arange(1, len(grads.shape))))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    @tf.function
    def train_step(self, batch):
        if isinstance(batch, tuple):
            batch = batch[0]

        batch_size = tf.shape(batch)[0]
        mini_batch_size = batch_size // self.iterations_critic

        fake = tf.ones((mini_batch_size, 1))
        valid = -tf.ones((mini_batch_size, 1))

        batch_eg_loss = []
        batch_cx_loss = []
        batch_cz_loss = []

        print("STEP 0")

        # Train the critics
        for j in range(self.iterations_critic):
            # x = batch
            x = batch[j * mini_batch_size: (j + 1) * mini_batch_size]
            z = tf.random.normal(shape=(mini_batch_size, self.latent_dim, 1))

            print("STEP 1")
            with tf.GradientTape(persistent=True) as tape:
                x_ = self.generator(z, training=True) # z -> x
                z_ = self.encoder(x, training=True) # x -> z

                print("STEP 2")
                cx_real = self.critic_x(x, training=True)
                cx_fake = self.critic_x(x_, training=True)

                print("STEP 3")
                cz_real = self.critic_z(z, training=True)
                cz_fake = self.critic_z(z_, training=True)

                # Calculate loss real
                print("STEP 4")
                cx_real_cost = self.critic_x_loss_fn(valid, cx_real)
                cz_real_cost = self.critic_z_loss_fn(valid, cz_real)

                # Calculate loss fake
                print("STEP 5")
                cx_fake_cost = self.critic_x_loss_fn(fake, cx_fake)
                cz_fake_cost = self.critic_z_loss_fn(fake, cz_fake)

                # Calculate the gradient penalty
                print("STEP 6")
                cx_gp = self.critic_x_gradient_penalty(mini_batch_size, (x, x_))
                cz_gp = self.critic_z_gradient_penalty(mini_batch_size, (z, z_))

                # Add the gradient penalty to the original loss
                print("STEP 7")
                cx_loss = cx_real_cost + cx_fake_cost + cx_gp * self.gp_weight
                cz_loss = cz_real_cost + cz_fake_cost + cz_gp * self.gp_weight

            # Get the gradients for the critics
            print("STEP 8")
            cx_grads = tape.gradient(cx_loss, self.critic_x.trainable_weights)
            cz_grads = tape.gradient(cz_loss, self.critic_z.trainable_weights)

            # Update the weights of the discriminators
            print("STEP 9")
            self.cx_optimizer.apply_gradients(zip(cx_grads, self.critic_x.trainable_weights))
            self.cz_optimizer.apply_gradients(zip(cz_grads, self.critic_z.trainable_weights))

            # Record loss
            print("STEP 10")
            batch_cx_loss.append([cx_loss, cx_real_cost, cx_fake_cost, cx_gp])
            batch_cz_loss.append([cz_loss, cz_real_cost, cz_fake_cost, cz_gp])
            
        with tf.GradientTape() as tape:
            print("STEP 11")
            x_ = self.generator(z, training=True) # z -> x
            z_ = self.encoder(x, training=True) # x -> z
            cycled_x = self.generator(z_, training=True) # x -> z -> x

            print("STEP 12")
            cx_fake = self.critic_x(x_, training=True)
            cz_fake = self.critic_z(z_, training=True)

            # Generator/encoder adverserial loss
            print("STEP 13")
            x_cost = self.encoder_generator_loss_fn(valid, cx_fake)
            z_cost = self.encoder_generator_loss_fn(valid, cz_fake)

            # Generator/encoder cycle loss
            print("STEP 14")
            cycle_loss = self.cycle_loss_fn(x, cycled_x)

            # Total loss
            print("STEP 15")
            eg_loss = x_cost + z_cost + self.cycle_weight * cycle_loss

        # Get the gradients for the generators
        print("STEP 16")
        encoder_generator_grads = tape.gradient((eg_loss, x_cost, z_cost, cycle_loss), 
            self.encoder.trainable_variables + self.generator.trainable_variables)

        # Update the weights of the generators
        print("STEP 17")
        self.encoder_generator_optimizer.apply_gradients(
            zip(encoder_generator_grads, self.encoder.trainable_variables + self.generator.trainable_variables))

        batch_eg_loss = (eg_loss, x_cost, z_cost, cycle_loss)

        return {
            # "Dx loss": tf.reduce_mean(batch_cx_loss, axis=0)[0], 
            # "Dz loss": tf.reduce_mean(batch_cz_loss, axis=0)[0], 
            "EG loss": batch_eg_loss[0]
        }


    @tf.function
    def test_step(self, X):
        if isinstance(batch, tuple):
            batch = batch[0]

        batch_size = tf.shape(batch)[0]

        # Prepare data
        fake = tf.ones((batch_size, 1))
        valid = -tf.ones((batch_size, 1))

        z = tf.random.normal(shape=(batch_size, self.latent_dim, 1))

        x_ = self.generator(z) # z -> x
        z_ = self.encoder(x) # x -> z
        cycled_x = self.generator(z_) # x -> z -> x

        cx_real = self.critic_x(x)
        cx_fake = self.critic_x(x_)

        cz_real = self.critic_z(z)
        cz_fake = self.critic_z(z_)

        # Calculate losses
        cx_real_cost = self.critic_x_loss_fn(valid, cx_real)
        cz_real_cost = self.critic_z_loss_fn(valid, cz_real)

        cx_fake_cost = self.critic_x_loss_fn(fake, cx_fake)
        cz_fake_cost = self.critic_z_loss_fn(fake, cz_fake)

        cx_gp = self.critic_x_gradient_penalty(batch_size, (x, x_))
        cz_gp = self.critic_z_gradient_penalty(batch_size, (z, z_))

        g_cost = self.generator_loss_fn(valid, cx_fake)
        e_cost = self.encoder_loss_fn(valid, cz_fake)

        cycle_loss = self.cycle_loss_fn(x, cycled_x)

        # Loss combination
        cx_loss = cx_real_cost + cx_fake_cost + cx_gp * self.gp_weight
        cz_loss = cz_real_cost + cz_fake_cost + cz_gp * self.gp_weight
        eg_loss = g_cost + e_cost + self.cycle_weight * cycle_loss
        
        return {
            "Dx loss": cx_loss, 
            "Dz loss": cz_loss, 
            "EG loss": eg_loss
        }

    @tf.function
    def call(self, X):
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
        z_ = self.encoder(X)
        y_hat = self.generator(z_)
        critic = self.critic_x(X)

        return y_hat, critic


class TadGAN():
    """TadGAN class."""

    def _wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def __init__(self, shape, optimizer, batch_size=64, learning_rate=0.0005, epochs=35, verbose=False, **kwargs):
        
        """Initialize the TadGAN object.
        """
        self.model = GAN(**kwargs)
        print(self.model)

        self.shape = shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.critic_x_optimizer = import_object(optimizer)(learning_rate)
        self.critic_z_optimizer = import_object(optimizer)(learning_rate)
        self.encoder_decoder_optimizer = import_object(optimizer)(learning_rate)

        self.encoder_generator_loss_fn = self._wasserstein_loss
        self.critic_x_loss_fn = self._wasserstein_loss
        self.critic_z_loss_fn = self._wasserstein_loss
        self.cycle_loss_fn = tf.keras.losses.MeanSquaredError()


    def fit(self, X, **kwargs):
        """Fit the TadGAN.
        Args:
            X (ndarray):
                N-dimensional array containing the input training sequences for the model.
        """
        train = X.copy()
        train = train.astype(np.float32)
        train = tf.data.Dataset.from_tensor_slices(train).shuffle(train.shape[0])
        train = train.batch(self.batch_size, drop_remainder=True)

        self.model.compile(self.critic_x_optimizer, self.critic_z_optimizer, 
            self.encoder_decoder_optimizer, self.encoder_generator_loss_fn,
            self.critic_x_loss_fn, self.critic_z_loss_fn, self.cycle_loss_fn, **kwargs)

        self.model.fit(train, batch_size=self.batch_size, verbose=True, epochs=2)      


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
        return self.model(X)


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


def _compute_rec_score(predictions, trues, score_window, smooth_window, rec_error_type):
    """Compute an array of anomaly scores.

    Args:
        predictions (ndarray):
            Predicted values.
        trues (ndarray):
            Ground truth.
        score_window (int):
            Size of the window over which the scores are calculated.
        smooth_window (int):
            Smooth window that will be applied to compute smooth errors.
        rec_error_type (str):
            Reconstruction error types.

    Returns:
        ndarray:
            Array of anomaly scores.
    """
    if (rec_error_type == "point"):
        errors = [abs(y_h - y) for y_h, y in zip(predictions, trues)]
        errors_smoothed = pd.Series(errors).rolling(
            smooth_window, center=True, min_periods=smooth_window // 2).mean().values
        z_scores = stats.zscore(errors_smoothed)
        z_scores = np.clip(z_scores, a_min=0, a_max=None) + 1

    elif (rec_error_type == "area"):
        pd_true = pd.Series(np.asarray(trues).flatten())
        pd_pred = pd.Series(np.asarray(predictions).flatten())
        score_measure_true = pd_true.rolling(score_window, center=True,
                                             min_periods=score_window // 2).apply(integrate.trapz)
        score_measure_pred = pd_pred.rolling(score_window, center=True,
                                             min_periods=score_window // 2).apply(integrate.trapz)
        errors = abs(score_measure_true - score_measure_pred)
        errors_smoothed = pd.Series(errors).rolling(smooth_window, center=True,
                                                    win_type='triang',
                                                    min_periods=smooth_window // 2).mean().values
        z_scores = stats.zscore(errors_smoothed)
        z_scores = np.clip(z_scores, a_min=0, a_max=None) + 1

    elif (rec_error_type == "dtw"):
        # DTW
        i = 0
        similarity_dtw = list()
        length_dtw = (score_window // 2) * 2 + 1
        hafl_length_dtw = length_dtw // 2
        # add padding
        true_pad = np.pad(trues, (hafl_length_dtw, hafl_length_dtw),
                          'constant', constant_values=(0, 0))
        predictions_pad = np.pad(
            predictions,
            (hafl_length_dtw,
             hafl_length_dtw),
            'constant',
            constant_values=(
                0,
                0))

        while i < len(trues) - length_dtw:
            true_data = np.zeros((length_dtw, 2))
            true_data[:, 0] = np.arange(length_dtw)
            true_data[:, 1] = true_pad[i:i + length_dtw]
            preds_data = np.zeros((length_dtw, 2))
            preds_data[:, 0] = np.arange(length_dtw)
            preds_data[:, 1] = predictions_pad[i:i + length_dtw]
            dtw, _ = sm.dtw(true_data, preds_data)
            similarity_dtw = similarity_dtw + [dtw]
            i += 1
        similarity_dtw = [0] * int(length_dtw / 2) + similarity_dtw + [0] * (
            len(trues) - len(similarity_dtw) - int(length_dtw / 2))
        errors = similarity_dtw
        errors_smoothed = pd.Series(errors).rolling(smooth_window, center=True,
                                                    min_periods=smooth_window // 2).mean().values
        z_scores = stats.zscore(errors_smoothed)
        z_scores = np.clip(z_scores, a_min=0, a_max=None) + 1

    return z_scores


def score_anomalies(y, y_hat, critic, index, score_window=10, critic_smooth_window=None,
                    error_smooth_window=None, rec_error_type="point", comb="mult", lambda_rec=0.5):
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

    true_index = index  # no offset

    true = [item[0] for item in y.reshape((y.shape[0], -1))]

    for item in y[-1][1:]:
        true.extend(item)

    critic_extended = list()
    for c in critic:
        critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())

    critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))

    predictions_md = []
    predictions = []

    critic_kde_max = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + (y_hat.shape[0] - 1)
    y_hat = np.asarray(y_hat)

    for i in range(num_errors):
        intermediate = []
        critic_intermediate = []

        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])
            critic_intermediate.append(critic_extended[i - j, j])

        if intermediate:
            predictions_md.append(np.median(np.asarray(intermediate)))

            predictions.append([[
                np.min(np.asarray(intermediate)),
                np.percentile(np.asarray(intermediate), 25),
                np.percentile(np.asarray(intermediate), 50),
                np.percentile(np.asarray(intermediate), 75),
                np.max(np.asarray(intermediate))
            ]])

            if len(critic_intermediate) > 1:
                discr_intermediate = np.asarray(critic_intermediate)
                try:
                    critic_kde_max.append(discr_intermediate[np.argmax(
                        stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
                except np.linalg.LinAlgError:
                    critic_kde_max.append(np.median(discr_intermediate))
            else:
                critic_kde_max.append(np.median(np.asarray(critic_intermediate)))

    predictions_md = np.asarray(predictions_md)

    # Compute critic scores
    critic_scores = _compute_critic_score(critic_kde_max, critic_smooth_window)

    # Compute reconstruction scores
    rec_scores = _compute_rec_score(
        predictions_md,
        true,
        score_window,
        error_smooth_window,
        rec_error_type)

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
