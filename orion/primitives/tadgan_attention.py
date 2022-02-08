# -*- coding: utf-8 -*-
"""
Tensorflow implementation of transformer with time series specific changes.

https://www.tensorflow.org/text/tutorials/transformer
"""

import logging
import pickle
import tempfile
from functools import partial
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
from mlprimitives.utils import import_object
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
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


class TadGANAttention(object):
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

    def _build_model(self, hyperparameters, layers, input_shape):
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

    def __init__(self, layers_encoder, layers_generator, layers_critic_x, layers_critic_z,
                 optimizer, input_shape=(100, 1), target_shape=(100, 1), latent_dim=20,
                 learning_rate=0.0005, epochs=2000, batch_size=64, iterations_critic=5,
                 print_loss=False, save_logs=False, **hyperparameters):

        self.shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.latent_shape = (latent_dim, 1)
        self.target_shape = target_shape
        self.iterations_critic = iterations_critic

        self.layers_encoder, self.layers_generator = layers_encoder, layers_generator
        self.layers_critic_x, self.layers_critic_z = layers_critic_x, layers_critic_z

        # self.optimizer = import_object(optimizer)(learning_rate, beta_1=0.9, beta_2=0.98,
        #                                           epsilon=1e-9)
        self.optimizer = import_object(optimizer)(learning_rate)

        self.hyperparameters = hyperparameters
        self._fitted = False

        self.print_loss = print_loss
        self.save_logs = save_logs
        self.epoch_losses = dict()
        self.validation = dict()

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
            self.epoch_losses[epoch] = [cx_loss, cz_loss, g_loss]

            if self.print_loss:
                print('Epoch: {}/{}, [Dx loss: {}] [Dz loss: {}] [G loss: {}]'.format(
                    epoch, self.epochs, cx_loss, cz_loss, g_loss))

            if epoch % 10 == 0:
                z_ = self.encoder.predict(X)
                y_hat = self.generator.predict(z_)
                critic = self.critic_x.predict(target)
                self.validation[epoch] = [y_hat, critic]

    def fit(self, X, y=None, **kwargs):
        """Fit the TadGAN.

        Args:
            X (ndarray):
                N-dimensional array containing the input training sequences for the model.
            y (ndarray):
                N-dimensional array containing the target sequences we want to reconstruct.
        """
        if y is None:
            y = X.copy()  # reconstruct the same input
        X, y = X.astype('float32'), y.astype('float32')

        if not self._fitted:
            self._augment_hyperparameters(X, y, kwargs)
            self._set_shapes()
            self._build_tadgan(**kwargs)

        self._fit(X, y)
        self._fitted = True

        if self.save_logs:
            log_directory = os.path.join(os.getcwd(), 'logs')
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)
            filenames = os.listdir(log_directory)
            max_idx = 0
            for filename in filenames:
                if '.pkl' in filename:
                    idx, _ = filename.split('.')
                    max_idx = max(int(idx), max_idx)
            max_idx += 1

            with open(os.path.join(log_directory, f'{max_idx}.pkl'), 'wb') as f:
                pickle.dump([self.epoch_losses, self.validation], f)

    def predict(self, X, y=None):
        """Predict values using the initialized object.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.
            y (ndarray):
                N-dimensional array containing the target sequences we want to reconstruct.

        Returns:
            ndarray:
                N-dimensional array containing the reconstructions for each input sequence.
            ndarray:
                N-dimensional array containing the critic scores for each input sequence.
        """
        if y is None:
            y = X.copy()  # reconstruct the same input
        X, y = X.astype('float32'), y.astype('float32')

        z_ = self.encoder.predict(X)
        y_hat = self.generator.predict(z_)
        critic = self.critic_x.predict(y)

        return y_hat, critic


def point_wise_feed_forward_network(d_model: int, dff: int):
    """Consists of two fully-connected layers with a ReLU activation in between."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    """Instead of one single attention head, Q, K, and V are split into multiple heads because it allows the model to
    jointly attend to information from different representation subspaces at different positions. After the split each
    head has a reduced dimensionality, so the total computation cost is the same as a single head attention with full
    dimensionality."""

    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1,
                                              3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1,
                                       self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def split_heads(self, x, batch_size: int):
        """Split the last dimension into (num_heads, depth).

        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @classmethod
    def scaled_dot_product_attention(cls, q, k, v, mask) -> tuple:
        """Calculate the attention weights.

        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits,
                                          axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
        })
        return config


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, d_model: int, maximum_position_encoding: int = 10000, rate: float = 0.1,
                 **kwargs):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.pos_encoding = self.get_encoding(maximum_position_encoding, self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=True):
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        return x

    @classmethod
    def get_encoding(cls, position: int, d_model: int):
        """Creates positional encoding of sequences in time series.

        Args:
            position: maximum position to encode
            d_model: dimensions of time series

        Returns:
            positional encoding with shape (n, d)
        """
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = np.arange(position)[:, np.newaxis] * angle_rates

        # Apply sin to even indices in the array; 2i.
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1.
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'maximum_position_encoding': self.maximum_position_encoding,
            'rate': self.rate,
        })

        return config


class EncoderLayer(tf.keras.layers.Layer):
    """Consist of Multi-head attention and point wise feed forward networks. Each of these sublayers
    has a residual connection around it followed by a layer normalization. Residual connections help
    in avoiding the vanishing gradient problem in deep networks."""

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1, **kwargs):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate  # dropout rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=True, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'dff': self.dff,
            'rate': self.rate,

        })
        return config


class Encoder(tf.keras.layers.Layer):
    """The input is summed with the positional encoding.
    The output of this summation is the input to the encoder layers.
    The output of the encoder is the input to the decoder."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int,
                 maximum_position_encoding: int = 10000, rate: float = 0.1, **kwargs):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.pos_encoding = PositionalEncoding(self.d_model, self.maximum_position_encoding, rate)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=True, mask=None):
        x = self.pos_encoding(x, training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'rate': self.rate,
        })

        return config


class DecoderLayer(tf.keras.layers.Layer):
    """Masked multi-head attention (with look ahead mask).
    Multi-head attention (with padding mask).
        V (value) and K (key) receive the encoder output as inputs.
        Q (query) receives the output from the masked multi-head attention sublayer.
    Point wise feed forward networks"""

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, enc_output, look_ahead_mask, training=True, mask=None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x,
                                               look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layer_norm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layer_norm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layer_norm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'dff': self.dff,
            'rate': self.rate,

        })
        return config


class Decoder(tf.keras.layers.Layer):
    """The target is put through an embedding which is summed with the positional encoding. The output of this
    summation is the input to the decoder layers. The output of the decoder is the input to the final linear layer."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int,
                 maximum_position_encoding: int = 10000, rate: float = 0.1, **kwargs):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.pos_encoding = PositionalEncoding(self.d_model, self.maximum_position_encoding,
                                               self.rate)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, training=True, mask=None):
        attention_weights = {}
        x = self.pos_encoding(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, training, mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'rate': self.rate,
        })

        return config


class Transformer(tf.keras.layers.Layer):
    """Transformer consists of the encoder, decoder and a final linear layer. The output of the decoder is the
    input to the linear layer and its output is returned."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int,
                 maximum_position_encoding: int = 10000, rate: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, maximum_position_encoding,
                               rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, maximum_position_encoding,
                               rate)

    def call(self, x, training=True):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(x, training)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        dec_output, attention_weights = self.decoder(x, enc_output, look_ahead_mask, training)

        return dec_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'rate': self.rate,
        })

        return config


class AttentionLayer(tf.keras.layers.Layer):
    """https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm"""

    def __init__(self, return_sequences: bool = True):
        self.return_sequences = return_sequences
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences,
        })

        return config


class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, begin, size, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.begin = begin
        self.size = size

    def call(self, inputs):
        return tf.slice(inputs, self.begin, self.size)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
        })
        return config
