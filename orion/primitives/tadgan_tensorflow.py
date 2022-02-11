# -*- coding: utf-8 -*-
"""TadGAN tensorflow implementation with support for attention."""

import logging
import tempfile
from functools import partial

import numpy as np
import tensorflow as tf
from mlprimitives.utils import import_object
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, Input, TimeDistributed
from tensorflow.keras.models import Model

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
LOGGER = logging.getLogger(__name__)


def build_layer(layer, hyperparameters):
    layer_class = import_object(layer['class'])
    layer_kwargs = layer['parameters'].copy()
    if issubclass(layer_class, TimeDistributed) or issubclass(layer_class, Bidirectional):
        layer_kwargs['layer'] = build_layer(layer_kwargs['layer'], hyperparameters)
    for key, value in layer_kwargs.items():
        if isinstance(value, str):
            layer_kwargs[key] = hyperparameters.get(value, value)
    return layer_class(**layer_kwargs)


class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = K.random_uniform((self.batch_size, 1, 1), dtype=tf.float32)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class TadGANTensorFlow(object):
    """TadGAN model for time series reconstruction.

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

    def __getstate__(self):
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        modules = ['optimizer', 'critic_x_model', 'critic_z_model', 'encoder_generator_model']

        state = self.__dict__.copy()

        for module in modules:
            del state[module]

        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                keras.models.save_model(state.pop(network), fd.name, overwrite=True)

                state[network + '_str'] = fd.read()

        return state

    def __setstate__(self, state):
        networks = ['critic_x', 'critic_z', 'encoder', 'generator']
        for network in networks:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                fd.write(state.pop(network + '_str'))
                fd.flush()

                state[network] = keras.models.load_model(fd.name)

        self.__dict__ = state

    def _build_model(self, hyperparameters: dict, layers: list, input_shape: tuple) -> Model:
        x = Input(shape=input_shape)
        model = keras.models.Sequential()

        for layer in layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        return Model(x, model(x))

    def _setdefault(self, kwargs, key, value):
        if key in kwargs:
            return

        if key in self.hyperparameters and self.hyperparameters[key] is None:
            kwargs[key] = value

    def _wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def _gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def __init__(self, layers_encoder: list, layers_generator: list, layers_critic_x: list,
                 layers_critic_z: list, optimizer: str, input_shape: tuple = (100, 1),
                 target_shape: tuple = (100, 1), latent_dim: int = 20,
                 learning_rate: float = 0.0005, epochs: int = 2000, batch_size: int = 64,
                 iterations_critic: int = 5, attention_optimizer: bool = False,
                 print_loss: bool = False, **hyperparameters):

        self.shape = input_shape  # [n, d_model]
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.latent_shape = (latent_dim, 1)
        self.target_shape = target_shape
        self.iterations_critic = iterations_critic

        self.layers_encoder, self.layers_generator = layers_encoder, layers_generator
        self.layers_critic_x, self.layers_critic_z = layers_critic_x, layers_critic_z

        self.optimizer = import_object(optimizer)(learning_rate)
        if attention_optimizer:
            self.optimizer = import_object(optimizer)(CustomSchedule(self.shape[-1]),
                                                      beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.hyperparameters = hyperparameters
        self._fitted = False

        # Fit Summary
        self.print_loss = print_loss
        self.train_index = None
        self.train_y = None
        self.train_losses = []
        self.train_predictions = []

    def _augment_hyperparameters(self, X, y, kwargs):
        shape = np.asarray(X)[0].shape
        length = shape[0]
        target_shape = np.asarray(y)[0].shape

        # to infer the shape
        self.shape = self.shape or shape
        self.target_shape = self.target_shape or target_shape

        self._setdefault(kwargs, 'generator_reshape_dim', length // 2)
        self._setdefault(kwargs, 'generator_reshape_shape', (length // 2, 1))
        self._setdefault(kwargs, 'encoder_reshape_shape', self.latent_shape)

        return kwargs

    def _set_shapes(self):
        self.encoder_input_shape = self.shape
        self.generator_input_shape = self.latent_shape
        self.critic_x_input_shape = self.target_shape
        self.critic_z_input_shape = self.latent_shape

    def _build_tadgan(self, **kwargs):
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

        x = Input(shape=self.shape)
        y = Input(shape=self.target_shape)
        z = Input(shape=(self.latent_dim, 1))

        x_ = self.generator(z)
        z_ = self.encoder(x)
        fake_x = self.critic_x(x_)
        valid_x = self.critic_x(y)

        interpolated_x = RandomWeightedAverage()([y, x_])
        validity_interpolated_x = self.critic_x(interpolated_x)
        partial_gp_loss_x = partial(self._gradient_penalty_loss, averaged_samples=interpolated_x)
        partial_gp_loss_x.__name__ = 'gradient_penalty'
        self.critic_x_model = Model(inputs=[y, z], outputs=[valid_x, fake_x,
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
        fake_gen_x = self.critic_x(x_gen_)
        fake_gen_z = self.critic_z(z_gen_)

        self.encoder_generator_model = Model([x_gen, z_gen], [fake_gen_x, fake_gen_z, x_gen_rec])
        self.encoder_generator_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss,
                                                   'mse'], optimizer=self.optimizer,
                                             loss_weights=[1, 1, 10])

    def _fit(self, X, target):

        fake = np.ones((self.batch_size, 1))
        valid = -np.ones((self.batch_size, 1))
        delta = np.ones((self.batch_size, 1))

        indices = np.arange(X.shape[0])
        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(indices)
            X_ = X[indices]
            y_ = target[indices]

            epoch_g_loss = []
            epoch_cx_loss = []
            epoch_cz_loss = []

            minibatches_size = self.batch_size * self.iterations_critic
            num_minibatches = int(X_.shape[0] // minibatches_size)

            for i in range(num_minibatches):
                minibatch = X_[i * minibatches_size: (i + 1) * minibatches_size]
                y_minibatch = y_[i * minibatches_size: (i + 1) * minibatches_size]

                for j in range(self.iterations_critic):
                    x = minibatch[j * self.batch_size: (j + 1) * self.batch_size]
                    y = y_minibatch[j * self.batch_size: (j + 1) * self.batch_size]
                    z = np.random.normal(size=(self.batch_size, self.latent_dim, 1))
                    epoch_cx_loss.append(
                        self.critic_x_model.train_on_batch([y, z], [valid, fake, delta]))
                    epoch_cz_loss.append(
                        self.critic_z_model.train_on_batch([x, z], [valid, fake, delta]))

                epoch_g_loss.append(
                    self.encoder_generator_model.train_on_batch([x, z], [valid, valid, y]))

            cx_loss = np.mean(np.array(epoch_cx_loss), axis=0)
            cz_loss = np.mean(np.array(epoch_cz_loss), axis=0)
            g_loss = np.mean(np.array(epoch_g_loss), axis=0)
            self.train_losses.append([cx_loss, cz_loss, g_loss])

            if self.print_loss:
                print('Epoch: {}/{}, [Dx loss: {}] [Dz loss: {}] [G loss: {}]'.format(
                    epoch, self.epochs, cx_loss, cz_loss, g_loss))

            if epoch % 5 == 0:
                z_ = self.encoder.predict(X)
                y_hat = self.generator.predict(z_)
                critic = self.critic_x.predict(target)
                self.train_predictions.append([y_hat, critic])

    def fit(self, X: np.array, y: np.array = None, **kwargs) -> None:
        """Fit the TadGAN.

        Args:
            X: N-dimensional array containing the input training sequences for the model
            y: N-dimensional array containing the target sequences we want to reconstruct
            **kwargs: extra hyperparameters.
        """
        self.train_index = kwargs.get('index')

        if y is None:
            y = X.copy()  # Reconstruct the same input.
        self.train_y = y
        X, y = X.astype('float32'), y.astype('float32')

        if not self._fitted:
            self._augment_hyperparameters(X, y, kwargs)
            self._set_shapes()
            self._build_tadgan(**kwargs)

        self._fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray, y: np.ndarray = None) -> tuple:
        """Predict values using the initialized object.

        Args:
            X: N-dimensional array containing the input sequences for the model.
            y: N-dimensional array containing the target sequences we want to reconstruct.

        Returns:
            ndarray: N-dimensional array containing the reconstructions for each input sequence.
            ndarray: N-dimensional array containing the critic scores for each input sequence.
            list: model fit summary [losses, [y_hat, critic]]
        """
        if y is None:
            y = X.copy()  # reconstruct the same input
        X, y = X.astype('float32'), y.astype('float32')

        z_ = self.encoder.predict(X)
        y_hat = self.generator.predict(z_)
        critic = self.critic_x.predict(y)

        fit_summary = [self.train_losses, self.train_y, self.train_predictions, self.train_index]

        return y_hat, critic, fit_summary
