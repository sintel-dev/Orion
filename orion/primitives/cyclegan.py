# -*- coding: utf-8 -*-

import logging
from functools import partial

import keras
import numpy as np
import pandas as pd
import math
import similaritymeasures as sm
from keras import backend as K
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.models import Model
from scipy import integrate, stats

from mlprimitives.adapters.keras import build_layer
from mlprimitives.utils import import_object

LOGGER = logging.getLogger(__name__)


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        """
        Args:
            inputs[0] x     original input
            inputs[1] x_    predicted input
        """
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
        """Initialize the CycleGAN object.
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
        """Creates a discriminator model that takes an timeseries as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.
        """
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
                                    loss_weights=[1, 1, 10])

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
        z_gen_rec = self.encoder(x_gen_)
        fake_gen_x = self.critic_x(x_gen_)
        fake_gen_z = self.critic_z(z_gen_)

        """
        part to tune:
        cycle consistent loss, single or bi-directional
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
        cycleGAN paper lambda_identity=0.5, lambda_A=10, lambda_B = 10
        lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)
        """
        # single
        self.encoder_generator_model = Model([x_gen, z_gen], [fake_gen_x, fake_gen_z, x_gen_rec])
        self.encoder_generator_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss,
                                                   'mse'], optimizer=self.optimizer,
                                             loss_weights=[1, 1, 10])
        # bi-directional
#         self.encoder_generator_model = Model(
#             [x_gen, z_gen], [fake_gen_x, fake_gen_z, z_gen_rec, x_gen_rec])
#         self.encoder_generator_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss,
#                                                    'mse', 'mse'], optimizer=self.optimizer,
#                                              loss_weights=[1, 1, 5, 5])
        

    def _fit(self, X):
        fake = np.ones((self.batch_size, 1))
        valid = -np.ones((self.batch_size, 1))
#         delta = np.ones((self.batch_size, 1)) * 10
        delta = np.zeros((self.batch_size, 1))
        

        for epoch in range(self.epochs):
            for _ in range(self.iterations_critic):
                idx = np.random.randint(0, X.shape[0], self.batch_size)
                x = X[idx]
                z = np.random.normal(size=(self.batch_size, self.latent_dim, 1))

                cx_loss = self.critic_x_model.train_on_batch([x, z], [valid, fake, delta])
                cz_loss = self.critic_z_model.train_on_batch([x, z], [valid, fake, delta])
            """
            part to tune:
            cycle consistent loss, single or both
            """
            # single direction
            g_loss = self.encoder_generator_model.train_on_batch([x, z], [valid, valid, x])
            # both direction
#             g_loss = self.encoder_generator_model.train_on_batch([x, z], [valid, valid, z, x])

            if epoch == 0 or (epoch + 1) % 500 == 0:
                print('Epoch: {}, [Dx loss: {}] [Dz loss: {}] [G loss: {}]'.format(
                    epoch + 1, cx_loss, cz_loss, g_loss))

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

def score_anomalies(y, y_hat, critic, index, score_window=10, critic_smooth_window=200, error_smooth_window=200):
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
        smooth_window (int):
            Optional. Size of window over which smoothing is applied.
            If not given, 200 is used.
    Returns:
        ndarray:
            Array of anomaly scores.
    """
    
    critic_smooth_window = math.trunc(y.shape[1] * 0.01)
    error_smooth_window = math.trunc(y.shape[1] * 0.01)
    
    """
    part to tune:
    what offset to index shall we add? Answer: not to add
    """
    true_index = index # no offset
#     true_index = index - (index[1] - index[0]) * (y.shape[1] // 2) # left offset for half window_size
#     true_index = index - (index[1] - index[0]) * (y.shape[1]) # left offset for one window_size
    
    true = [item[0] for item in y.reshape((y.shape[0], -1))]

    for item in y[-1][1:]:
        true.extend(item)

    critic_extended = list()
    for c in critic:
        critic_extended = critic_extended + np.repeat(c, y_hat.shape[1]).tolist()

    predictions_md = []
    predictions_mn = []
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

        """
        Option: 1) use median; 2) use mean;
        Answer: only use median
        """
        if intermediate:
            predictions_md.append(np.median(np.asarray(intermediate)))
            predictions_mn.append(np.mean(np.asarray(intermediate)))

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
    predictions_mn = np.asarray(predictions_mn)
    print('true shape', np.array(true).shape)
    print('predictions shape', predictions_md.shape)
    
    """
    Compute critic scores
    Methods include:
        1. std times for all values (alex)
    """
    critic_kde_max = np.asarray(critic_kde_max)
    l_quantile = np.quantile(critic_kde_max, 0.25)
    u_quantile = np.quantile(critic_kde_max, 0.75)
    in_range = np.logical_and(critic_kde_max >= l_quantile, critic_kde_max <= u_quantile)
    critic_mean = np.mean(critic_kde_max[in_range])
    critic_std = np.std(critic_kde_max)
    
    print('critic score stats', l_quantile, u_quantile, critic_mean, np.min(critic_kde_max), np.max(critic_kde_max))

    # alex's critic
    z_score_critic1 = np.absolute((np.asarray(critic_kde_max) - critic_mean) / critic_std) + 1
    z_score_critic1 = pd.Series(z_score_critic_clip).rolling(
        error_smooth_window, center=True, min_periods=error_smooth_window//2).mean().values
    
    """
    Compute reconstruction scores
    Methods include:
        1. Pointwise difference
    """
    
    # Point-wise difference
    errors_md1 = [abs(y_h - y) for y_h, y in zip(predictions_md, true)]
    errors_smoothed_md1 = pd.Series(errors_md1).rolling(
        error_smooth_window, center=True, min_periods=error_smooth_window//2).mean().values
    z_score_res_md1 = stats.zscore(errors_smoothed_md1)
    z_score_res_md1 = np.clip(z_score_res_md1, a_min=0, a_max=None) + 1
    

    """ Combine all the options together """
    final_scores = np.multiply(z_score_critic1, z_score_res_md1)
    return final_scores, true_index


def score_anomalies_set(y, y_hat, critic, index, score_window=10, critic_smooth_window=200, error_smooth_window=200):
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
        smooth_window (int):
            Optional. Size of window over which smoothing is applied.
            If not given, 200 is used.
    Returns:
        ndarray:
            Array of anomaly scores.
    """
    
    critic_smooth_window = min(math.trunc(y.shape[0] * 0.01), 200)
    error_smooth_window = min(math.trunc(y.shape[0] * 0.01), 200)
    print('critic_smooth_window: {}, error_smooth_window: {}, \
           score_window: {}'.format(critic_smooth_window, error_smooth_window, score_window))
    
    """
    part to tune:
    what offset to index shall we add?
    """
    true_index = index # no offset
#     true_index = index - (index[1] - index[0]) * (y.shape[1] // 2) # left offset for half window_size
#     true_index = index - (index[1] - index[0]) * (y.shape[1]) # left offset for one window_size
    
    true = [item[0] for item in y.reshape((y.shape[0], -1))]

    for item in y[-1][1:]:
        true.extend(item)

    critic_extended = list()
    for c in critic:
        critic_extended = critic_extended + np.repeat(c, y_hat.shape[1]).tolist()

    predictions_md = []
    predictions_mn = []
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

        """
        Option: 1) use median; 2) use mean
        """
        if intermediate:
            predictions_md.append(np.median(np.asarray(intermediate)))
            predictions_mn.append(np.mean(np.asarray(intermediate)))

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
    predictions_mn = np.asarray(predictions_mn)
#     print('true shape', np.array(true).shape)
#     print('predictions shape', predictions_md.shape)
    
    """
    Compute critic scores
    Methods include:
        1. std times for all values (alex)
        2. std times only for the values <= mean (dyu)
    """
    critic_kde_max = np.asarray(critic_kde_max)
    l_quantile = np.quantile(critic_kde_max, 0.25)
    u_quantile = np.quantile(critic_kde_max, 0.75)
    in_range = np.logical_and(critic_kde_max >= l_quantile, critic_kde_max <= u_quantile)
    critic_mean = np.mean(critic_kde_max[in_range])
    critic_std = np.std(critic_kde_max)
    
#     print('critic score stats', l_quantile, u_quantile, critic_mean, np.min(critic_kde_max), np.max(critic_kde_max))

    # alex's critic
    z_score_critic1 = np.absolute((np.asarray(critic_kde_max) - critic_mean) / critic_std) + 1
    z_score_critic1 = pd.Series(z_score_critic1).rolling(
        critic_smooth_window, center=True, min_periods=critic_smooth_window//2).mean().values
    
#     # dyu's critic: only consider values <= mean
#     z_score_critic2 = (np.asarray(critic_kde_max) - critic_mean) / critic_std
#     z_score_critic2 = np.absolute(np.clip(z_score_critic2, a_min=None, a_max=0)) + 1
#     z_score_critic2 = pd.Series(z_score_critic2).rolling(
#         error_smooth_window, center=True, min_periods=error_smooth_window//2).mean().values

    
    """
    Compute reconstruction scores
    Methods include:
        1. Pointwise difference
        2. Area diference
        3. DTW
    """
    
    # Point-wise difference
    errors_md1 = [abs(y_h - y) for y_h, y in zip(predictions_md, true)]
    errors_smoothed_md1 = pd.Series(errors_md1).rolling(
        error_smooth_window, center=True, min_periods=error_smooth_window//2).mean().values
    z_score_res_md1 = stats.zscore(errors_smoothed_md1)
    z_score_res_md1 = np.clip(z_score_res_md1, a_min=0, a_max=None) + 1
    
    
#     errors_mn1 = [abs(y_h - y) for y_h, y in zip(predictions_mn, true)]
#     errors_smoothed_mn1 = pd.Series(errors_mn1).rolling(
#         error_smooth_window, center=True, min_periods=error_smooth_window//2).mean().values
#     z_score_res_mn1 = stats.zscore(errors_smoothed_mn1)
#     z_score_res_mn1 = np.clip(z_score_res_mn1, a_min=0, a_max=None) + 1
    
    # Area difference
    pd_true = pd.Series(np.asarray(true).flatten())
    pd_pred = pd.Series(np.asarray(predictions_md).flatten())
    score_measure_true = pd_true.rolling(score_window, center=True, min_periods=score_window//2)\
        .apply(integrate.trapz)
    score_measure_pred = pd_pred.rolling(score_window, center=True, min_periods=score_window//2)\
        .apply(integrate.trapz)
    errors_md2 = abs(score_measure_true - score_measure_pred)
#     errors_smoothed_md2 = pd.Series(errors_md2).rolling(error_smooth_window, center=True,
#                                                       min_periods=error_smooth_window//2).mean().values
    errors_smoothed_md2 = pd.Series(errors_md2).rolling(error_smooth_window, center=True,
                                                        win_type='triang',
                                                        min_periods=error_smooth_window//2).mean().values
    z_score_res_md2 = stats.zscore(errors_smoothed_md2)
    z_score_res_md2 = np.clip(z_score_res_md2, a_min=0, a_max=None) + 1
    
#     pd_true = pd.Series(np.asarray(true).flatten())
#     pd_pred = pd.Series(np.asarray(predictions_mn).flatten())
#     score_measure_true = pd_true.rolling(score_window, center=True, min_periods=score_window//2)\
#         .apply(integrate.trapz)
#     score_measure_pred = pd_pred.rolling(score_window, center=True, min_periods=score_window//2)\
#         .apply(integrate.trapz)
#     errors_mn2 = abs(score_measure_true - score_measure_pred)
#     errors_smoothed_mn2 = pd.Series(errors_mn2).rolling(error_smooth_window, center=True,
#                                                       min_periods=error_smooth_window//2).mean().values
#     z_score_res_mn2 = stats.zscore(errors_smoothed_mn2)
#     z_score_res_mn2 = np.clip(z_score_res_mn2, a_min=0, a_max=None) + 1
    
    # DTW
    """One problem: not centered; added by dyu"""
    i = 0
    similarity_dtw = list()
    length_dtw = (score_window // 2) * 2 + 1
    hafl_length_dtw = length_dtw // 2
    # add padding
    true_pad = np.pad(true, (hafl_length_dtw, hafl_length_dtw), 'constant', constant_values=(0, 0))
    predictions_md_pad = np.pad(predictions_md, (hafl_length_dtw, hafl_length_dtw), 'constant', constant_values=(0, 0))
    
    while i < len(true) - length_dtw:
        true_data = np.zeros((length_dtw, 2))
        true_data[:, 0] = np.arange(length_dtw)
#         true_data[:, 1] = true[i:i + length_dtw]
        true_data[:, 1] = true_pad[i:i + length_dtw]
        preds_data = np.zeros((length_dtw, 2))
        preds_data[:, 0] = np.arange(length_dtw)
#         preds_data[:, 1] = predictions_md[i:i + length_dtw]
        preds_data[:, 1] = predictions_md_pad[i:i + length_dtw]
        dtw, _ = sm.dtw(true_data, preds_data)
        similarity_dtw = similarity_dtw + [dtw]
        i += 1
    similarity_dtw = [0] * int(length_dtw / 2) + similarity_dtw + [0] * (
            len(true) - len(similarity_dtw) - int(length_dtw / 2))
    errors_md3 = similarity_dtw
    errors_smoothed_md3 = pd.Series(errors_md3).rolling(error_smooth_window, center=True,
                                                      min_periods=error_smooth_window//2).mean().values
    z_score_res_md3 = stats.zscore(errors_smoothed_md3)
    z_score_res_md3 = np.clip(z_score_res_md3, a_min=0, a_max=None) + 1
    
#     i = 0
#     similarity_dtw = list()
#     length_dtw = score_window
#     while i < len(true) - length_dtw:
#         true_data = np.zeros((length_dtw, 2))
#         true_data[:, 0] = np.arange(length_dtw)
#         true_data[:, 1] = true[i:i + length_dtw]
#         preds_data = np.zeros((length_dtw, 2))
#         preds_data[:, 0] = np.arange(length_dtw)
#         preds_data[:, 1] = predictions_mn[i:i + length_dtw]
#         dtw, _ = sm.dtw(true_data, preds_data)
#         similarity_dtw = similarity_dtw + [dtw]
#         i += 1
#     similarity_dtw = [0] * int(length_dtw / 2) + similarity_dtw + [0] * (
#             len(true) - len(similarity_dtw) - int(length_dtw / 2))
#     errors_mn3 = similarity_dtw
#     errors_smoothed_mn3 = pd.Series(errors_mn3).rolling(error_smooth_window, center=True,
#                                                       min_periods=error_smooth_window//2).mean().values
#     z_score_res_mn3 = stats.zscore(errors_smoothed_mn3)
#     z_score_res_mn3 = np.clip(z_score_res_mn3, a_min=0, a_max=None) + 1
    

    """ Combine all the options together """
    final_scores = [
        z_score_critic1,       # only critic
        z_score_res_md1,       # Pointwise median
        z_score_res_md2,       # Area difference median
        z_score_res_md3,       # DTW median
        # multiply comb
        np.multiply(z_score_critic1, z_score_res_md1),
        np.multiply(z_score_critic1, z_score_res_md2),
        np.multiply(z_score_critic1, z_score_res_md3),
        # lambda comb
        0.5*(z_score_critic1 - 1) + 0.5*(z_score_res_md1 - 1),
        0.5*(z_score_critic1 - 1) + 0.5*(z_score_res_md2 - 1),
        0.5*(z_score_critic1 - 1) + 0.5*(z_score_res_md3 - 1)
    ]
    
#     final_scores = [
#         z_score_critic1,  # only critic - alex
#         z_score_critic2,  # only critic - dyu
#         z_score_res_md1,       # Pointwise median
#         z_score_res_md2,       # Area difference median
#         z_score_res_md3,       # DTW median
#         # multiply comb
#         np.multiply(z_score_critic1, z_score_res_md1),
#         np.multiply(z_score_critic1, z_score_res_md2),
#         np.multiply(z_score_critic1, z_score_res_md3),
#         np.multiply(z_score_critic1, z_score_res_mn1),
#         np.multiply(z_score_critic1, z_score_res_mn2),
#         np.multiply(z_score_critic1, z_score_res_mn3),
#         # lambda comb
#         0.5*(z_score_critic2 - 1) + 0.5*(z_score_res_md1 - 1),
#         0.5*(z_score_critic2 - 1) + 0.5*(z_score_res_md2 - 1),
#         0.5*(z_score_critic2 - 1) + 0.5*(z_score_res_md3 - 1),
#         0.5*(z_score_critic2 - 1) + 0.5*(z_score_res_mn1 - 1),
#         0.5*(z_score_critic2 - 1) + 0.5*(z_score_res_mn2 - 1),
#         0.5*(z_score_critic2 - 1) + 0.5*(z_score_res_mn3 - 1)
#     ]
    
    return final_scores, true_index
