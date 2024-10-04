"""Primitive Anomaly Transformer

This primitive is an pytorch implementation of "Anomaly Transformer:
Time Series Anomaly Detection with Association Discrepancy"
https://arxiv.org/pdf/2110.02642.pdf

This is a modified version of the original code, which can be found
at https://github.com/thuml/Anomaly-Transformer/tree/main
"""
# -*- coding: utf-8 -*-

import logging
import math
import operator
import os
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlstars.utils import import_object
from torch.utils.data import DataLoader

from orion.primitives.timeseries_anomalies import _merge_sequences, _prune_anomalies

LOGGER = logging.getLogger(__name__)


class Signal(object):
    """Data object.

    Args:
        X (ndarray):
            An n-dimensional array of signal values.
        window_size (int):
            Size of the window.
        step (int):
            Stride size.
    """

    def __init__(self, X, window_size, step=1, mode='train'):
        self.data = X
        self.step = step
        self.mode = mode
        self.window_size = window_size

    def __len__(self):
        if self.mode == 'test':
            return (self.data.shape[0] - self.window_size) // self.window_size + 1

        return (self.data.shape[0] - self.window_size) // self.step + 1

    def __getitem__(self, index):
        start = index * self.step
        end = start + self.window_size

        if self.mode == 'train':
            return np.float32(self.data[start: end])
        elif self.mode == 'test':
            start = start // self.step * self.window_size
            return np.float32(self.data[start: start + self.window_size])
        else:
            raise ValueError(f'Unknown {self.mode} mode.')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, input_size, d_model):
        super(TokenEmbedding, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=input_size, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular',
                                   bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, input_size, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(input_size=input_size, d_model=d_model)
        self.position_encoding = PositionalEncoding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_encoding(x)
        return self.dropout(x)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool),
                                    diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, window_size, mask_flag=True, scale=None, attention_dropout=0.0,
                 output_attention=False, device="cpu"):
        super(AnomalyAttention, self).__init__()

        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.device = device
        self.dropout = nn.Dropout(attention_dropout)
        self.distances = torch.zeros((window_size, window_size)).to(device)
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = scale * scores
        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(
            sigma.shape[0], sigma.shape[1], 1, 1).to(self.device)

        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, num_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // num_heads)
        d_values = d_values or (d_model // num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * num_heads)
        self.key_projection = nn.Linear(d_model, d_keys * num_heads)
        self.value_projection = nn.Linear(d_model, d_values * num_heads)
        self.sigma_projection = nn.Linear(d_model, num_heads)
        self.out_projection = nn.Linear(d_values * num_heads, d_model)

        self.num_heads = num_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, n_hidden=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()

        n_hidden = n_hidden or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=n_hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=n_hidden, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class ATModel(nn.Module):
    def __init__(self, window_size, input_size, output_size, d_model=512, num_heads=8,
                 num_layers=3, n_hidden=512, dropout=0.0, activation='gelu',
                 output_attention=True, attention_dropout=0.0, device="cpu"):
        super(ATModel, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.embedding = DataEmbedding(input_size, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            window_size=window_size, mask_flag=False, scale=None,
                            attention_dropout=attention_dropout, output_attention=output_attention,
                            device=device
                        ),
                        d_model=d_model, num_heads=num_heads
                    ),
                    d_model=d_model,
                    n_hidden=n_hidden,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, output_size, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]


class AnomalyTransformer():
    """Anomaly Transformer model for unsupervised time series anomaly detection.

    Args:
        window_size (int):
            Window size of each sample.
        step (int):
            Stride size between samples.
        unit (str):
            String representing the unit of timestamps.
        interval (int):
            The time gap between one sample and another.
        input_size (int):
            Input size for the network.
        output_size (int):
            Output size for the network.
        d_model (int):
            Model dimension.
        n_hidden (int):
            Hidden dimension.
        batch_size (int):
            Number of example per epoch.
        dropout (float):
            Dropout value of the network.
        attention_dropout (float):
            Dropout value for attention.
        epochs (int):
            Number of iterations to train the model.
        learning_rate (float):
            Learning rate for the optimizer.
        temperature (int):
            Scaling value. Default 50.
        verbose (bool):
            Whether to be on verbose mode or not.
        cuda (bool):
            Whether to use GPU or not.
        valid_split (float):
            A float to split data dataframe to validation set. Data needs to contain a label
            column. Use ``target_column`` to change the target column name.
        output_dir (str):
            Path to folder where to save the model.
    """
    @staticmethod
    def _kl_loss(p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    @staticmethod
    def _adjust_learning_rate(optimizer, epoch, lr_):
        lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            LOGGER.info(f'Updating learning rate to {lr}')

    def __init__(self, input_size=1, output_size=1, window_size=100, step=1, k=3, d_model=512,
                 n_hidden=512, num_layers=3, num_heads=8, attention_dropout=0.0, dropout=0.0,
                 activation='gelu', output_attention=True, batch_size=256, learning_rate=1e-4,
                 temperature=50, epochs=10, valid_split=0.0, shuffle=True, cuda=True,
                 optimizer="torch.optim.Adam", verbose=False, output_dir=False):

        self.input_size = input_size
        self.output_size = output_size
        self.window_size = window_size
        self.step = step

        self.k = k
        self.d_model = d_model
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.output_attention = output_attention
        self.attention_dropout = attention_dropout

        self.temperature = temperature
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.valid_split = valid_split
        self.shuffle = shuffle
        self.cuda = cuda
        self.verbose = verbose
        self.output_dir = output_dir

        self.device = "cpu"
        if cuda and torch.cuda.is_available():
            self.device = "cuda"

        # build model
        self.model = ATModel(
            window_size=self.window_size,
            input_size=self.input_size,
            output_size=self.output_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            n_hidden=self.n_hidden,
            dropout=self.dropout,
            activation=self.activation,
            output_attention=self.output_attention,
            attention_dropout=self.attention_dropout,
            device=self.device
        )

        self.optimizer = import_object(optimizer)(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss(reduce=False)
        self.model.to(self.device)

    def _get_loss(self, prior, series, temperature=1, detach=False):
        # calculate Association discrepancy
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_value = series[u]
            prior_value = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1),
                                                     dim=-1).repeat(1, 1, 1, self.window_size)

            series_second = series_value
            prior_second = prior_value

            if detach:
                series_second = series_second.detach()
                prior_second = prior_second.detach()

            series_loss += (
                torch.mean(self._kl_loss(series_second, prior_value.detach()))
                + torch.mean(self._kl_loss(prior_value.detach(), series_second))
            ) * temperature

            prior_loss += (
                torch.mean(self._kl_loss(prior_second, series_value.detach()))
                + torch.mean(self._kl_loss(series_value.detach(), prior_second))
            ) * temperature

        return series_loss, prior_loss

    def _get_energy(self, data_loader):
        self.model.eval()

        energy = []
        predictions = []
        for i, input_data in enumerate(data_loader):
            x = input_data.to(self.device)
            output, series, prior, _ = self.model(x)
            predictions.append(output.detach().cpu().numpy())
            loss = torch.mean(self.mse(x, output), dim=-1)
            series_loss, prior_loss = self._get_loss(
                prior, series, temperature=self.temperature, detach=True)

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            energy.append((metric * loss).detach().cpu().numpy())

        return np.concatenate(energy, axis=0), np.concatenate(predictions, axis=0)

    def _validate(self, valid_loader):
        self.model.eval()

        losses = list()
        for input_data in valid_loader:
            x = input_data.to(self.device)
            output, series, prior, _ = self.model(x)

            series_loss, _ = self._get_loss(prior, series)
            series_loss = series_loss / len(prior)
            rec_loss = self.criterion(output, x)

            losses.append((rec_loss - self.k * series_loss).item())

        return np.mean(losses)

    def _fit(self, train_loader, valid_loader):
        for epoch in range(self.epochs):
            losses = []
            self.model.train()

            for input_data in train_loader:
                self.optimizer.zero_grad()
                x = input_data.to(self.device)
                output, series, prior, _ = self.model(x)

                series_loss, prior_loss = self._get_loss(prior, series)
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, x)

                losses.append((rec_loss - self.k * series_loss).item())
                rec_series_loss = rec_loss - self.k * series_loss
                rec_prior_loss = rec_loss + self.k * prior_loss

                # minimax strategy
                rec_series_loss.backward(retain_graph=True)
                rec_prior_loss.backward()
                self.optimizer.step()

            if valid_loader is not None:
                valid_loss = self._validate(valid_loader)
            else:
                valid_loss = None

            if self.verbose:
                print('Epoch: {}/{}, Loss: {}, Valid Loss {}'.format(
                    epoch + 1, self.epochs, np.mean(losses), valid_loss))

            self._adjust_learning_rate(self.optimizer, epoch + 1, self.learning_rate)

    def fit(self, X):
        train = X
        valid_loader = None

        # split data
        if self.valid_split > 0:
            valid_size = int(len(X) * self.valid_split)
            train = X[: -valid_size]
            valid = X[-valid_size:]

            valid_loader = DataLoader(dataset=Signal(valid, self.window_size, self.step),
                                      batch_size=self.batch_size,
                                      shuffle=False)

        train_loader = DataLoader(dataset=Signal(train, self.window_size, self.step),
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle)

        self._fit(train_loader, valid_loader)

        if self.output_dir:
            model_dir = Path(self.output_dir)
            os.makedirs(model_dir, exist_ok=True)
            LOGGER.info(f"Saving model to {model_dir}.")
            torch.save(self.model.state_dict(), model_dir + f'checkpoint_{self.epochs}.pth')

        self.train_energy, train_predictions = self._get_energy(train_loader)

    def predict(self, X):
        data_loader = DataLoader(dataset=Signal(X, self.window_size, self.step, mode='test'),
                                 batch_size=self.batch_size)

        energy, predictions = self._get_energy(data_loader)
        return predictions, energy, self.train_energy


def threshold_anomalies(energy, index, train_energy, anomaly_ratio=1.0, min_percent=0.1,
                        anomaly_padding=50):
    energy = np.array(energy.reshape(-1))
    train_energy = np.array(train_energy.reshape(-1))
    combined_energy = np.concatenate([train_energy, energy], axis=0)
    thresh = np.percentile(combined_energy, 100 - anomaly_ratio)

    anomalies = (energy > thresh).astype(int)

    intervals = list()
    idx = 0
    for is_anomaly, g in groupby(anomalies):
        length = len(list(g))
        if is_anomaly == 1:
            start = max(0, idx - anomaly_padding)
            end = min(idx + length + anomaly_padding, len(anomalies)) - 1
            intervals.append((index[start], index[end], np.mean(energy[start: end + 1])))

        idx += length

    intervals.sort(key=operator.itemgetter(2), reverse=True)
    intervals = pd.DataFrame.from_records(intervals, columns=['start', 'stop', 'max_error'])

    pruned = _prune_anomalies(intervals, min_percent)
    return _merge_sequences(pruned)
