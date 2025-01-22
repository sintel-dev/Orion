# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
import torch

from orion.primitives.anomaly_transformer import (
    AnomalyAttention, AnomalyTransformer, ATModel, AttentionLayer, DataEmbedding, Encoder,
    EncoderLayer, PositionalEncoding, Signal, TokenEmbedding, TriangularCausalMask,
    threshold_anomalies)


class TestSignal(TestCase):

    @classmethod
    def setup_class(cls):
        # Create some sample data for testing
        cls.data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cls.window_size = 3
        cls.step = 2

    def test_len_train_mode(self):
        signal = Signal(self.data, self.window_size, step=self.step, mode='train')
        expected_length = (self.data.shape[0] - self.window_size) // self.step + 1
        self.assertEqual(len(signal), expected_length)

    def test_len_test_mode(self):
        signal = Signal(self.data, self.window_size, step=self.step, mode='test')
        expected_length = (self.data.shape[0] - self.window_size) // self.window_size + 1
        self.assertEqual(len(signal), expected_length)

    def test_getitem_train_mode(self):
        signal = Signal(self.data, self.window_size, step=self.step, mode='train')
        index = 1
        expected_output = np.float32([3, 4, 5])
        np.testing.assert_array_equal(signal[index], expected_output)

    def test_getitem_test_mode(self):
        signal = Signal(self.data, self.window_size, step=self.step, mode='test')
        index = 1
        expected_output = np.float32([4, 5, 6])
        np.testing.assert_array_equal(signal[index], expected_output)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            signal = Signal(self.data, self.window_size, mode='unknown')
            index = 1
            signal[index]


class TestPositionalEncoding(TestCase):

    def test_forward(self):
        d_model = 16
        max_len = 10
        pos_enc = PositionalEncoding(d_model, max_len)
        input_tensor = torch.randn(1, 5, d_model)
        output_tensor = pos_enc(input_tensor)

        self.assertEqual(output_tensor.size(), (1, 5, d_model))
        self.assertTrue(torch.all(torch.eq(output_tensor, pos_enc.pe[:, :5])))


class TestTokenEmbedding(TestCase):

    def test_forward(self):
        input_size = 8
        d_model = 16
        token_emb = TokenEmbedding(input_size, d_model)
        input_tensor = torch.randn(1, 5, input_size)
        output_tensor = token_emb(input_tensor)

        self.assertEqual(output_tensor.size(), (1, 5, d_model))


class TestDataEmbedding(TestCase):

    def test_forward(self):
        input_size = 8
        d_model = 16
        dropout = 0.1
        data_emb = DataEmbedding(input_size, d_model, dropout)
        input_tensor = torch.randn(1, 5, input_size)
        output_tensor = data_emb(input_tensor)

        self.assertEqual(output_tensor.size(), (1, 5, d_model))


class TestTriangularCausalMask(TestCase):

    def test_mask_shape(self):
        B, L = 3, 5
        mask = TriangularCausalMask(B, L).mask

        self.assertEqual(mask.size(), (B, 1, L, L))

    def test_mask_values(self):
        B, L = 3, 5
        mask = TriangularCausalMask(B, L).mask

        self.assertTrue(torch.all(mask[:, :, -1:, :] == 0))
        self.assertTrue(torch.all(mask[:, :, :, :1] == 0))


class TestAnomalyAttention(TestCase):

    @classmethod
    def setup_class(cls):
        cls.queries = torch.randn(2, 4, 3, 16)
        cls.keys = torch.randn(2, 4, 3, 16)
        cls.values = torch.randn(2, 4, 3, 16)
        cls.sigma = torch.randn(2, 3, 4)
        cls.attn_mask = None

    def test_calculate_attention_dim(self):
        attention = AnomalyAttention(window_size=1)
        expected_shape = (2, 4, 3, 16)

        output = attention.forward(
            self.queries, self.keys, self.values, self.sigma, self.attn_mask)

        self.assertEqual(output[0].shape, expected_shape)

    def test_mask_flag_false(self):
        attention = AnomalyAttention(window_size=1, mask_flag=False)
        expected_shape = (2, 4, 3, 16)

        output = attention.forward(
            self.queries, self.keys, self.values, self.sigma, self.attn_mask)

        # Check output shape
        self.assertEqual(output[0].shape, expected_shape)

    def test_window_size(self):
        attention = AnomalyAttention(window_size=4)
        sigma = torch.randn(2, 4, 4)
        expected_shape = (2, 4, 3, 16)

        output = attention.forward(self.queries, self.keys, self.values, sigma, self.attn_mask)

        self.assertEqual(output[0].shape, expected_shape)


class TestAttentionLayer(TestCase):

    def test_forward(self):
        attention = AnomalyAttention(window_size=1, output_attention=True)
        attention_layer = AttentionLayer(attention=attention, d_model=16, num_heads=1)

        queries = torch.randn(2, 5, 16)
        keys = torch.randn(2, 5, 16)
        values = torch.randn(2, 5, 16)
        attn_mask = None

        out, series, prior, sigma = attention_layer.forward(queries, keys, values, attn_mask)

        self.assertEqual(out.shape, (2, 5, 16))
        self.assertEqual(series.shape, (2, 1, 5, 5))
        self.assertEqual(prior.shape, (2, 1, 5, 5))
        self.assertEqual(sigma.shape, (2, 1, 5, 5))


class TestEncoderLayer(TestCase):

    def test_forward(self):
        attention_layer = AttentionLayer(
            AnomalyAttention(window_size=1, output_attention=True),
            d_model=16,
            num_heads=1
        )

        d_model = 16
        n_hidden = 8
        dropout = 0.2
        activation = "relu"
        x = torch.randn(2, 5, d_model)

        layer = EncoderLayer(
            attention_layer,
            d_model=d_model,
            n_hidden=n_hidden,
            dropout=dropout,
            activation=activation
        )
        output = layer.forward(x)

        self.assertEqual(layer.attention, attention_layer)
        self.assertEqual(layer.dropout.p, dropout)
        self.assertEqual(layer.activation, torch.nn.functional.relu)
        self.assertEqual(output[0].shape, x.shape)

    def test_gelu(self):
        attention = AnomalyAttention(window_size=1, output_attention=True)
        d_model = 16
        activation = "gelu"

        layer = EncoderLayer(attention, d_model=d_model, activation=activation)

        self.assertEqual(layer.activation, torch.nn.functional.gelu)


class TestEncoder(TestCase):

    def test_forward(self):
        d_model = 16
        n_hidden = 8
        dropout = 0.2
        activation = "relu"
        num_layers = 1

        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    AnomalyAttention(
                        window_size=1, output_attention=True
                    ),
                    d_model=16, num_heads=1
                ),
                d_model=d_model,
                n_hidden=n_hidden,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ]

        encoder = Encoder(encoder_layers)

        x = torch.randn(2, 5, 16)
        output = encoder.forward(x)

        self.assertEqual(output[0].shape, x.shape)


class TestATModel(TestCase):

    @classmethod
    def setup_class(cls):
        cls.d_model = 16
        cls.n_hidden = 8
        cls.dropout = 0.2
        cls.activation = "relu"
        cls.num_layers = 1
        cls.num_heads = 1
        cls.window_size = 1
        cls.input_size = 1
        cls.output_size = 1
        cls.output_attention = True
        cls.attention_dropout = 0.

        cls.model = ATModel(
            window_size=cls.window_size,
            input_size=cls.input_size,
            output_size=cls.output_size,
            d_model=cls.d_model,
            num_heads=cls.num_heads,
            num_layers=cls.num_layers,
            n_hidden=cls.n_hidden,
            dropout=cls.dropout,
            activation=cls.activation,
            output_attention=cls.output_attention,
            attention_dropout=cls.attention_dropout,
        )

    def test__init__(self):
        self.assertEqual(self.model.output_attention, self.output_attention)
        assert isinstance(self.model.embedding, DataEmbedding)
        assert isinstance(self.model.encoder, Encoder)
        assert isinstance(self.model.projection, torch.nn.Linear)

    def test_forward(self):
        x = torch.randn(1, self.window_size, self.input_size)
        output = self.model.forward(x)

        assert len(output) == 4
        self.assertEqual(output[0].shape, x.shape)


class TestAnomalyTransformer(TestCase):

    @staticmethod
    def setup_class(cls):
        cls.input_size = 1
        cls.output_size = 1
        cls.window_size = 1
        cls.step = 1
        cls.k = 3
        cls.d_model = 16
        cls.n_hidden = 8
        cls.num_layers = 1
        cls.num_heads = 1
        cls.dropout = 0.2
        cls.activation = "relu"
        cls.output_attention = True
        cls.attention_dropout = 0.

        cls.model = AnomalyTransformer(
            window_size=cls.window_size,
            input_size=cls.input_size,
            output_size=cls.output_size,
            d_model=cls.d_model,
            num_heads=cls.num_heads,
            num_layers=cls.num_layers,
            n_hidden=cls.n_hidden,
            dropout=cls.dropout,
            activation=cls.activation,
            output_attention=cls.output_attention,
            attention_dropout=cls.attention_dropout
        )

    def test__init__(self):
        self.assertEqual(self.model.input_size, 1)
        self.assertEqual(self.model.output_size, 1)
        self.assertEqual(self.model.window_size, 1)
        self.assertEqual(self.model.step, 1)
        self.assertEqual(self.model.k, 3)
        self.assertEqual(self.model.d_model, 16)
        self.assertEqual(self.model.n_hidden, 8)
        self.assertEqual(self.model.num_layers, 1)
        self.assertEqual(self.model.num_heads, 1)
        self.assertEqual(self.model.dropout, 0.2)
        self.assertEqual(self.model.activation, 'relu')
        self.assertEqual(self.model.output_attention, True)
        self.assertEqual(self.model.attention_dropout, 0.0)

        # defaults
        self.assertEqual(self.model.batch_size, 256)
        self.assertEqual(self.model.learning_rate, 1e-4)
        self.assertEqual(self.model.temperature, 50)
        self.assertEqual(self.model.epochs, 10)
        self.assertEqual(self.model.valid_split, 0.0)
        self.assertEqual(self.model.shuffle, True)
        self.assertEqual(self.model.cuda, True)
        self.assertEqual(self.model.verbose, False)
        self.assertEqual(self.model.output_dir, False)

        assert isinstance(self.model.optimizer, torch.optim.Adam)

    def test_fit(self):
        x = np.random.rand(100, 1)

        self.model.fit(x)

        self.assertIsNotNone(self.model.train_energy)

    def test_predict(self):
        x = np.random.rand(100, 1)

        self.model.fit(x)
        predictions, energy, train_energy = self.model.predict(x)

        self.assertEqual(predictions.shape, (100, 1, 1))
        self.assertEqual(energy.shape, (100, 1))


def test_threshold_anomalies_no_anomalies():
    energy = np.array([1])
    index = np.array([0])
    train_energy = np.array([0, 1, 2, 3, 4])
    anomaly_ratio = 1.0
    min_percent = 0.1
    anomaly_padding = 0

    result = threshold_anomalies(
        energy, index, train_energy, anomaly_ratio, min_percent, anomaly_padding)

    expected_result = np.array([])
    np.testing.assert_array_equal(result, expected_result)


def test_threshold_anomalies_with_anomalies():
    energy = np.array([100, 101, 102, 0, 1, 2, 3])
    index = np.array([0, 1, 2, 3, 4, 5, 6])
    train_energy = np.array([0, 1, 2, 3, 4])
    anomaly_ratio = 50.0
    min_percent = 0.1
    anomaly_padding = 0

    result = threshold_anomalies(
        energy, index, train_energy, anomaly_ratio, min_percent, anomaly_padding)

    expected_result = np.array([(0, 2, 101)])
    np.testing.assert_array_equal(result, expected_result)
