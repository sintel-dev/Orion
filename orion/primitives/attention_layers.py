# -*- coding: utf-8 -*-
"""
Tensorflow implementation of transformer with time series specific changes.

https://www.tensorflow.org/text/tutorials/transformer
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float64')


def point_wise_feed_forward_network(d_model: int, dff: int):
    """Consists of two fully-connected layers with a ReLU activation in between."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float64)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


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
        dk = tf.cast(tf.shape(k)[-1], tf.float64)
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
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float64))
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
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float64(d_model))
        angle_rads = np.arange(position)[:, np.newaxis] * angle_rates

        # Apply sin to even indices in the array; 2i.
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1.
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float64)

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

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int = None,
                 maximum_position_encoding: int = 10000, rate: float = 0.1, **kwargs):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = d_model * 4 if dff is None else dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.pos_encoding = PositionalEncoding(self.d_model, self.maximum_position_encoding, rate)
        self.enc_layers = [EncoderLayer(d_model, num_heads, self.dff, rate) for _ in range(num_layers)]

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

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int = None,
                 maximum_position_encoding: int = 10000, rate: float = 0.1, **kwargs):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.dff = d_model * 4 if dff is None else dff

        self.pos_encoding = PositionalEncoding(self.d_model, self.maximum_position_encoding,
                                               self.rate)

        self.dec_layers = [DecoderLayer(d_model, num_heads, self.dff, rate) for _ in range(num_layers)]
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

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int = None,
                 maximum_position_encoding: int = 10000, rate: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = d_model * 4 if dff is None else dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.encoder = Encoder(num_layers, d_model, num_heads, self.dff, maximum_position_encoding,
                               rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, self.dff, maximum_position_encoding,
                               rate)

    def call(self, x, training=True):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(x, training)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        seq_len = tf.shape(x)[1]
        look_ahead_mask = tf.cast(1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0),
                                  tf.float64)
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
    """Attention layer over LSTM."""

    def __init__(self, return_sequences: bool = True):
        self.return_sequences = return_sequences
        super(AttentionLayer, self).__init__()

    def build(self, input_shape: tuple):
        self.att_weight = self.add_weight(shape=(input_shape[-1], 1), initializer="normal")
        self.att_bias = self.add_weight(shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, X):
        e = K.tanh(K.dot(X, self.att_weight) + self.att_bias)
        a = K.softmax(e, axis=1)
        output = X * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences,
        })

        return config
