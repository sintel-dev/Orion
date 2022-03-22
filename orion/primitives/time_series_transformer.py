"""Time series transformer encoder.

References
    - https://www.kaggle.com/yamqwe/tutorial-time-series-transformer-time2vec
    - https://www.tensorflow.org/addons/api_docs/python/tfa/layers/MultiHeadAttention
"""
import tensorflow as tf
from tensorflow.keras import backend as K

from orion.primitives.attention_layers import point_wise_feed_forward_network, MultiHeadAttention

tf.keras.backend.set_floatx('float64')


class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int = 1):
        super(Time2Vec, self).__init__(trainable=True)
        self.kernel_size = kernel_size

        # Trend & Periodic
        self.W_t, self.b_t = None, None
        self.W_p, self.b_p = None, None

    def build(self, input_shape):
        self.W_t = self.add_weight(name='W_t', shape=(input_shape[1],),
                                   initializer='uniform', trainable=True)
        self.b_t = self.add_weight(name='b_t', shape=(input_shape[1],),
                                   initializer='uniform', trainable=True)
        self.W_p = self.add_weight(name='W_p', shape=(1, input_shape[1], self.kernel_size),
                                   initializer='uniform', trainable=True)
        self.b_p = self.add_weight(name='b_p', shape=(1, input_shape[1], self.kernel_size),
                                   initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.W_t * inputs + self.b_t
        dp = K.dot(inputs, self.W_p) + self.b_p
        wgts = K.sin(dp)  # or K.cos(.)
        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1] * (self.kernel_size + 1)))
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] * (self.kernel_size + 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
        })
        return config


class EncoderLayer(tf.keras.layers.Layer):
    """Consist of Multi-head attention and point wise feed forward networks. Each of these
    sublayers has a residual connection around it followed by a layer normalization. Residual
    connections help in avoiding the vanishing gradient problem in deep networks."""

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1, **kwargs):
        super(EncoderLayer, self).__init__(trainable=True)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.layer_norm1 = tf.keras.layers.BatchNormalization()

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layer_norm2 = tf.keras.layers.BatchNormalization()

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
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
        })
        return config


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model: int = None, num_heads: int = None, dff: int = None,
                 rate: float = 0.1, num_layers: int = 2, time2vec_dim: int = 2,
                 skip_connection_strength: float = 0.9, return_sequences: bool = False, **kwargs):
        super(Encoder, self).__init__(trainable=True)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.num_layers = num_layers
        self.time2vec_dim = time2vec_dim
        self.skip_connection_strength = skip_connection_strength
        self.return_sequences = return_sequences

        self.embedding = None
        self.enc_layers = None
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.time_encoding = tf.keras.layers.TimeDistributed(Time2Vec(time2vec_dim - 1))
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        self.d_model = self.d_model if self.d_model else input_shape[-1]
        self.embedding = tf.keras.layers.Dense(self.d_model)
        self.d_model *= (self.time2vec_dim + 1)
        self.num_heads = self.num_heads if self.num_heads else self.time2vec_dim + 1
        self.dff = self.dff if self.dff else 4 * self.d_model
        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff,
                                        self.rate) for _ in range(self.num_layers)]

    def call(self, x, training=True, mask=None):
        x = self.embedding(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, self.time_encoding(x)])
        x = self.layer_norm(x)
        for i in range(self.num_layers):
            prev_x = x
            x = self.enc_layers[i](x, training, mask)
            x = ((1.0 - self.skip_connection_strength) * x) + (
                self.skip_connection_strength * prev_x)
        return x if self.return_sequences else x[:, -1, :]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
            'num_layers': self.num_layers,
            'time2vec_dim': self.time2vec_dim,
            'skip_connection_strength': self.skip_connection_strength,
            'return_sequences': self.return_sequences
        })
        return config
