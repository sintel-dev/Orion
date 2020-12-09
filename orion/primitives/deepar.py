# -*- coding: utf-8 -*-

import logging
from datetime import datetime

import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

LOGGER = logging.getLogger(__name__)


class DeepAR():
    """DeepAR class"""

    def __init__(self, freq, context_length, prediction_length, learning_rate=0.001,
                 epochs=100, batch_size=32, num_batches_per_epoch=50, num_samples=100):
        """Initialize the DeepAR.

        Args:
            freq (str):
                Frequency of the data to train on and predict. Must be a valid `pandas` frequency.
            context_length (int):
                Length of the input sequence.
            prediction_length (int):
                Length of the prediction horizon.
            learning_rate (float):
                Optional. Float denoting the learning rate of the optimizer. Defaults to 0.001.
            epochs (int):
                Optional. Number of epochs that the network will train. Defaults to 100.
            batch_size (int):
                Optional. Number of examples in each batch. Defaults to 32.
            num_batches_per_epoch (int):
                Optional. Number of batches at each epoch. Defaults to 50.
            num_samples (int):
                Number of samples to draw on the model when evaluating. Defaultss to 100.
        """
        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_samples = num_samples

    def _create_train_dataset(self, X, index):
        start = datetime.utcfromtimestamp(index[0])
        train_ds = ListDataset(
            [{"start": start, "target": X[:-self.prediction_length, 0]}],
            freq=self.freq
        )
        return train_ds

    def _create_test_dataset(self, X, index):
        length = X.shape[0]
        start = datetime.utcfromtimestamp(index[0])
        test_ds = ListDataset(
            [{"start": start, "target": X[: self.context_length + i * self.prediction_length, 0]}
             for i in range(1, length + 1)],
            freq=self.freq
        )
        return test_ds

    def fit(self, X, index):
        """Fit DeepAR.

        Args:
            X (ndarray):
                N-dimensional array of the input sequence.
            index (ndarray):
                Array containing the index values of X.
        """

        train_ds = self._create_train_dataset(X, index)

        estimator = DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            trainer=Trainer(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                num_batches_per_epoch=self.num_batches_per_epoch
            )
        )

        # training
        self.predictor = estimator.train(train_ds)

    def _predict(self, X, index):
        test_ds = self._create_test_dataset(X, index)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,
            predictor=self.predictor,
            num_samples=self.num_samples
        )

        forecasts = list(forecast_it)
        median = list()
        for f in forecasts:
            median.append(np.median(f.samples))

        return np.asarray(median)

    def predict(self, X, index):
        """Predict values using DeepAR.

        Args:
            X (ndarray):
                N-dimensional array of the input sequence.
            index (ndarray):
                Array containing the index values of X.

        Returns:
            ndarray:
                Array containing the predicted values for the input sequence.
        """

        return self._predict(X, index)
