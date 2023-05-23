# -*- coding: utf-8 -*-

import logging
import math
import tempfile
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from mlstars.utils import import_object
from numpy import ndarray
from scipy import stats
from tensorflow.keras import Model

from orion.primitives.timeseries_errors import reconstruction_errors

LOGGER = logging.getLogger(__name__)
tf.keras.backend.set_floatx('float64')

LOSS_NAMES = [
    ['cx_loss', 'cx_real', 'cx_fake', 'cx_gp'],
    ['cz_loss', 'cz_real', 'cz_fake', 'cz_gp'],
    ['eg_loss', 'eg_cx_fake', 'eg_cz_fake', 'eg_mse']
]


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


class TadGAN:
    """TadGAN model for time series reconstruction.

    Args:
        layers_encoder (list):
            List containing layers of encoder.
        layers_generator (list):
            List containing layers of generator.
        layers_critic_x (list):
            List containing layers of critic_x.
        layers_critic_z (list):
            List containing layers of critic_z.
        input_shape (tuple):
            Optional. Tuple denoting the shape of an input sample.
        latent_dim (int):
            Optional. Integer denoting dimension of latent space. Default 20.
        target_shape (tuple):
            Optional. Tuple denoting the shape of an output sample.
        encoder_input_shape (tuple):
            Shape of encoder input.
        generator_input_shape (tuple):
            Shape of generator input.
        critic_x_input_shape (tuple):
            Shape of critic_x input.
        critic_z_input_shape (tuple):
            Shape of critic_z input.
        optimizer (str):
            String denoting the keras optimizer.
        learning_rate (float):
            Optional. Float denoting the learning rate of the optimizer. Default 0.005.
        epochs (int):
            Optional. Integer denoting the number of epochs. Default 50.
        latent_dim (int):
            Optional. Integer denoting dimension of latent space. Default 20.
        iterations_critic (int):
            Optional. Integer denoting the number of critic training steps per one
            Generator/Encoder training step. Default 5.
        shuffle (bool):
            Whether to shuffle the dataset for each epoch. Default True.
        verbose (int):
            Verbosity mode where 0 = silent, 1 = progress bar, 2 = one line per epoch. Default 0.
        detailed_losses (bool):
            Whether to output all loss values in verbose mode. Default False.
        **hyperparameters (dict):
            Optional. Dictionary containing any additional inputs.
    """
    @staticmethod
    def _build_model(hyperparameters: dict, layers: list,
                     input_shape: tuple, name: str) -> Model:
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Sequential(name=name)

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        return Model(x, model(x))

    @staticmethod
    def _wasserstein_loss(y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    @staticmethod
    def _gradient_penalty_loss_wrapper(critic):
        def _gradient_penalty_loss(real, fake):
            # Random weighted average to create interpolated signals.
            batch_size = tf.shape(real)[0]
            alpha = tf.random.uniform([batch_size, 1, 1], dtype=tf.float64)
            interpolated = (alpha * real) + ((1 - alpha) * fake)

            # Get the critic output for this interpolated signal.
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                validity_interpolated = critic(interpolated)

            # Calculate the gradients w.r.t to this interpolated signal.
            grads = gp_tape.gradient(validity_interpolated, [interpolated])[0]

            # Calculate the norm of the gradients.
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=np.arange(1, len(grads.shape))))
            gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
            return gradient_penalty

        return _gradient_penalty_loss

    def __init__(self, layers_encoder: list, layers_generator: list, layers_critic_x: list,
                 layers_critic_z: list, optimizer: str, input_shape: Optional[tuple] = None,
                 target_shape: Optional[tuple] = None, latent_dim: int = 20,
                 learning_rate: float = 0.005, epochs: int = 50, batch_size: int = 64,
                 iterations_critic: int = 5, shuffle: bool = True, detailed_losses: bool = False,
                 verbose: Union[int, bool] = True, **hyperparameters):
        """Initialize the TadGAN model."""
        super(TadGAN, self).__init__()

        # Required model hyperparameters to construct model.
        self.layers_encoder, self.layers_generator = layers_encoder, layers_generator
        self.layers_critic_x, self.layers_critic_z = layers_critic_x, layers_critic_z

        # Optional model hyperparameters.
        self.shape = input_shape
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        self.hyperparameters = hyperparameters

        # Model training hyperparameters.
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations_critic = iterations_critic
        self.shuffle = shuffle
        self.verbose = verbose
        self.detailed_losses = detailed_losses
        self.fitted = False

    def __getstate__(self):
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        modules = ['critic_x_model', 'critic_z_model', 'encoder_generator_model',
                   'critic_x_optimizer', 'critic_z_optimizer', 'encoder_generator_optimizer']

        state = self.__dict__.copy()

        for module in modules:
            del state[module]

        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                tf.keras.models.save_model(state.pop(network), fd.name, overwrite=True)
                state[network + '_str'] = fd.read()

        return state

    def __setstate__(self, state: dict):
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                fd.write(state.pop(network + '_str'))
                fd.flush()

                state[network] = tf.keras.models.load_model(fd.name)

        self.__dict__ = state

    def _augment_hyperparameters(self, X: ndarray, y: ndarray, **kwargs) -> dict:
        shape = np.asarray(X)[0].shape
        length, input_dim = shape
        target_shape = np.asarray(y)[0].shape
        output_dim = target_shape[1]

        # Infers the shape.
        self.shape = self.shape or shape
        self.target_shape = self.target_shape or target_shape
        self.latent_shape = (self.latent_dim, output_dim)

        kwargs.update({
            'shape': self.shape,
            'target_shape': self.target_shape,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'generator_reshape_dim': length // 2,
            'generator_reshape_shape': (length // 2, output_dim),
            'encoder_reshape_shape': self.latent_shape
        })
        return kwargs

    def _set_shapes(self) -> None:
        self.encoder_input_shape = self.shape
        self.generator_input_shape = self.latent_shape
        self.critic_x_input_shape = self.target_shape
        self.critic_z_input_shape = self.latent_shape

    def _format_losses(self, losses: list) -> dict:
        """Format losses into dictionary mapping metric names to their current value."""
        output = dict()
        if self.detailed_losses:
            for i in range(len(losses)):
                if not isinstance(losses[i], list):
                    losses[i] = [losses[i]]
                for j in range(len(losses[i])):
                    output[LOSS_NAMES[i][j]] = losses[i][j]
        else:
            for i in range(len(losses)):
                output[LOSS_NAMES[i][0]] = losses[i][0]

        return output

    def _build_tadgan(self, **kwargs) -> None:
        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)

        # Models
        self.encoder = self._build_model(
            hyperparameters, self.layers_encoder, self.encoder_input_shape, name='encoder')
        self.generator = self._build_model(
            hyperparameters, self.layers_generator, self.generator_input_shape, name='generator')
        self.critic_x = self._build_model(
            hyperparameters, self.layers_critic_x, self.critic_x_input_shape, name='critic_x')
        self.critic_z = self._build_model(
            hyperparameters, self.layers_critic_z, self.critic_z_input_shape, name='critic_z')

        # Optimizers
        self.critic_x_optimizer = import_object(self.optimizer)(self.learning_rate)
        self.critic_z_optimizer = import_object(self.optimizer)(self.learning_rate)
        self.encoder_generator_optimizer = import_object(self.optimizer)(self.learning_rate)

        x = tf.keras.Input(shape=self.shape)
        y = tf.keras.Input(shape=self.target_shape)
        z = tf.keras.Input(shape=self.latent_shape)

        self.generator.trainable = False
        self.encoder.trainable = False

        # Critic x model
        x_ = self.generator(z)
        cx_real = self.critic_x(y)
        cx_fake = self.critic_x(x_)
        self.critic_x_model = Model(inputs=[y, z], outputs=[cx_real, cx_fake, x_])
        self.critic_x_model.compile(loss=[self._wasserstein_loss,
                                          self._wasserstein_loss,
                                          self._gradient_penalty_loss_wrapper(self.critic_x)
                                          ],
                                    optimizer=self.critic_x_optimizer,
                                    loss_weights=[1, 1, 10])

        # Critic z model
        z_ = self.encoder(x)
        cz_real = self.critic_z(z)
        cz_fake = self.critic_z(z_)
        self.critic_z_model = Model(inputs=[x, z], outputs=[cz_real, cz_fake, z_])
        self.critic_z_model.compile(loss=[self._wasserstein_loss,
                                          self._wasserstein_loss,
                                          self._gradient_penalty_loss_wrapper(self.critic_z)
                                          ],
                                    optimizer=self.critic_z_optimizer,
                                    loss_weights=[1, 1, 10])

        self.critic_x.trainable = False
        self.critic_z.trainable = False
        self.generator.trainable = True
        self.encoder.trainable = True

        x_ = self.generator(z)
        cx_fake = self.critic_x(x_)
        z_ = self.encoder(x)
        cz_fake = self.critic_z(z_)
        x_rec_ = self.generator(z_)

        self.encoder_generator_model = Model([x, z], [cx_fake, cz_fake, x_rec_])
        self.encoder_generator_model.compile(loss=[self._wasserstein_loss,
                                                   self._wasserstein_loss,
                                                   'mse'
                                                   ],
                                             optimizer=self.encoder_generator_optimizer,
                                             loss_weights=[1, 1, 10])

    def _fit(self, data: tuple):
        X_train, y_train = data
        minibatch_size = self.batch_size * self.iterations_critic
        num_minibatch = X_train.shape[0] // minibatch_size
        z_shape = (self.batch_size, self.latent_shape[0], self.latent_shape[1])

        # Wasserstein GAN formulation.
        fake = tf.ones((self.batch_size, 1), dtype=tf.float64)
        real = -tf.ones((self.batch_size, 1), dtype=tf.float64)

        indices = np.arange(X_train.shape[0])
        for epoch in range(1, self.epochs + 1):
            if self.shuffle:
                np.random.shuffle(indices)

            x_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            epoch_cx_loss, epoch_cz_loss, epoch_eg_loss = [], [], []

            for i in range(num_minibatch):
                x_minibatch = x_shuffled[i * minibatch_size: (i + 1) * minibatch_size]
                y_minibatch = y_shuffled[i * minibatch_size: (i + 1) * minibatch_size]

                for j in range(self.iterations_critic):
                    x = x_minibatch[j * self.batch_size: (j + 1) * self.batch_size]
                    y = y_minibatch[j * self.batch_size: (j + 1) * self.batch_size]
                    z = tf.random.normal(shape=z_shape, dtype=tf.float64)
                    epoch_cx_loss.append(
                        self.critic_x_model.train_on_batch([y, z], [real, fake, y]))
                    epoch_cz_loss.append(
                        self.critic_z_model.train_on_batch([x, z], [real, fake, z]))

                epoch_eg_loss.append(
                    self.encoder_generator_model.train_on_batch([x, z], [real, real, y]))

            epoch_cx_loss = np.round(np.mean(np.array(epoch_cx_loss), axis=0), 4)
            epoch_cz_loss = np.round(np.mean(np.array(epoch_cz_loss), axis=0), 4)
            epoch_eg_loss = np.round(np.mean(np.array(epoch_eg_loss), axis=0), 4)
            losses = self._format_losses([epoch_cx_loss, epoch_cz_loss, epoch_eg_loss])
            if self.verbose:
                print('Epoch: {}/{}, Losses: {}'.format(epoch, self.epochs, losses))

    def fit(self, X: ndarray, y: Optional[ndarray] = None, **kwargs) -> None:
        """Fit the TadGAN model.

        Args:
            X (ndarray):
                N-dimensional array containing the input training sequences for the model.
            y (ndarray):
                N-dimensional array containing the target sequences we want to reconstruct.
            **kwargs (dict):
                Optional. Additional inputs.
        """
        if y is None:
            y = X.copy()
        X, y = X.astype(np.float64), y.astype(np.float64)

        # Infer dimensions and compile model.
        if not self.fitted:
            kwargs = self._augment_hyperparameters(X, y, **kwargs)
            self._set_shapes()
            self._build_tadgan(**kwargs)

        self._fit((X, y))
        self.fitted = True

    def predict(self, X: ndarray, y: Optional[ndarray] = None) -> tuple:
        """Predict using TadGAN model.

        Args:
            X (ndarray):
                N-dimensional array containing the input training sequences for the model.
            y (ndarray):
                N-dimensional array containing the target sequences we want to reconstruct.
        Returns:
            ndarray:
                N-dimensional array containing the reconstructions for each input sequence.
            ndarray:
                N-dimensional array containing the critic scores for each input sequence.
        """
        if y is None:
            y = X.copy()
        X, y = X.astype(np.float64), y.astype(np.float64)

        z_ = self.encoder.predict(X)
        y_hat = self.generator.predict(z_)
        critic = self.critic_x.predict(y)

        return y_hat, critic


def _compute_critic_score(critics: ndarray, smooth_window: int) -> ndarray:
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


def score_anomalies(y: ndarray, y_hat: ndarray, critic: ndarray, index: ndarray,
                    score_window: int = 10, critic_smooth_window: int = None,
                    error_smooth_window: int = None, smooth: bool = True,
                    rec_error_type: str = "point", comb: str = "mult",
                    lambda_rec: float = 0.5) -> tuple:
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
