# -*- coding: utf-8 -*-

import unittest
import numpy as np
import torch

from orion.primitives.anomaly_transformer import Signal
from orion.primitives.anomaly_transformer import (
    PositionalEncoding, TokenEmbedding, DataEmbedding, TriangularCausalMask)


class TestSignal(unittest.TestCase):

    def setUp(self):
        # Create some sample data for testing
        self.data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.window_size = 3
        self.step = 2

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
            item = signal[index]


class TestPositionalEncoding(unittest.TestCase):

    def test_forward(self):
        d_model = 16
        max_len = 10
        pos_enc = PositionalEncoding(d_model, max_len)
        input_tensor = torch.randn(1, 5, d_model)
        output_tensor = pos_enc(input_tensor)

        self.assertEqual(output_tensor.size(), (1, 5, d_model))
        self.assertTrue(torch.all(torch.eq(output_tensor, pos_enc.pe[:, :5])))

class TestTokenEmbedding(unittest.TestCase):

    def test_forward(self):
        input_size = 8
        d_model = 16
        token_emb = TokenEmbedding(input_size, d_model)
        input_tensor = torch.randn(1, 5, input_size)
        output_tensor = token_emb(input_tensor)

        self.assertEqual(output_tensor.size(), (1, 5, d_model))

class TestDataEmbedding(unittest.TestCase):

    def test_forward(self):
        input_size = 8
        d_model = 16
        dropout = 0.1
        data_emb = DataEmbedding(input_size, d_model, dropout)
        input_tensor = torch.randn(1, 5, input_size)
        output_tensor = data_emb(input_tensor)

        self.assertEqual(output_tensor.size(), (1, 5, d_model))

class TestTriangularCausalMask(unittest.TestCase):

    def test_mask_shape(self):
        B, L = 3, 5
        mask = TriangularCausalMask(B, L).mask

        self.assertEqual(mask.size(), (B, 1, L, L))

    def test_mask_values(self):
        B, L = 3, 5
        mask = TriangularCausalMask(B, L).mask

        self.assertTrue(torch.all(mask[:, :, -1:, :] == 0))
        self.assertTrue(torch.all(mask[:, :, :, :1] == 0))

