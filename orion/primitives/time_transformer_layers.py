import tensorflow as tf
from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float64')


def point_wise_feed_forward_network(d_model: int, dff: int):
    """Consists of two fully-connected layers with a ReLU activation in between."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.kernel_size = kernel_size

        # Trend
        self.W_t, self.b_t = None, None

        # Periodic
        self.W_p, self.b_p = None, None

    def build(self, input_shape):
        self.W_t = self.add_weight(name='W_t', shape=(input_shape[1],), initializer='uniform',
                                   trainable=True)
        self.b_t = self.add_weight(name='b_t', shape=(input_shape[1],), initializer='uniform',
                                   trainable=True)
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


class EncoderLayer(tf.keras.layers.Layer):
    """Consist of Multi-head attention and point wise feed forward networks. Each of these sublayers
    has a residual connection around it followed by a layer normalization. Residual connections help
    in avoiding the vanishing gradient problem in deep networks."""

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1, **kwargs):
        super(EncoderLayer, self).__init__()
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
    """The input is summed with the positional encoding.
    The output of this summation is the input to the encoder layers.
    The output of the encoder is the input to the decoder.

    References
        - https://www.kaggle.com/yamqwe/tutorial-time-series-transformer-time2vec/notebook
    """

    def __init__(self, d_model: int = None, num_heads: int = None, dff: int = None,
                 rate: float = 0.1, num_layers: int = 2, time2vec_dim: int = 2,
                 skip_connection_strength: float = 0.9, return_sequences: bool = False,
                 **kwargs):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.num_layers = num_layers
        self.time2vec_dim = time2vec_dim
        self.skip_connection_strength = skip_connection_strength
        self.return_sequences = return_sequences

        self.enc_layers = None
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.time_encoding = tf.keras.layers.TimeDistributed(Time2Vec(time2vec_dim - 1))
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        self.d_model = self.d_model if self.d_model else input_shape[-1] * (self.time2vec_dim + 1)
        self.num_heads = self.num_heads if self.num_heads else self.time2vec_dim + 1
        self.dff = self.dff if self.dff else 4 * self.d_model
        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff,
                                        self.rate) for _ in range(self.num_layers)]

    def call(self, x, training=True, mask=None):
        x = tf.keras.layers.Concatenate(axis=-1)([x, self.time_encoding(x)])
        x = self.layer_norm(x)
        for i in range(self.num_layers):
            prev_x = x
            x = self.enc_layers[i](x, training, mask)
            x = ((1.0 - self.skip_connection_strength) * x) + (
                self.skip_connection_strength * prev_x)
        return x if self.return_sequences else tf.squeeze(x[:, -1, :])

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
            'return_sequences': self.return_sequences,
        })
        return config
