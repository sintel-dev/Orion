"""UniTS

This primitive an implementation of "TIMESFM:
Time Series Foundation Model"
https://arxiv.org/abs/2310.10688

The model implementation can be found at
https://github.com/google-research/timesfm?tab=readme-ov-file
"""

import numpy as np
import timesfm as tf


def rolling_window_sequences(X, window_size, step_size):
    """Create rolling window sequences out of time series data.

    This function creates an array of sequences by rolling over the input sequence.

    Args:
        X (ndarray):
            The sequence to iterate over.
        window_size (int):
            Length of window.
        step_size (int):
            Indicating the number of steps to move the window forward each round.

    Returns:
        ndarray, ndarray:
            * rolling window sequences.
    """
    out_X = list()

    start = 0
    max_start = len(X) - window_size
    while start < max_start:
        end = start + window_size
        out_X.append(X[start:end])
        start = start + step_size

    return np.asarray(out_X)


class TimesFM:
    """TimesFM model for timeseries forecasting.

    Args:
        window_size (int):
            Window size of each sample. Must be multiple of 32. Default to 256.
        step (int):
            Stride length between samples. Default to 1.
        pred_len (int):
            Prediction horizon length. Default to 1.
        repo_id (str):
            Directory of the model checkpoint. Default to "google/timesfm-1.0-200m."
        freq (int):
            Frequency. TimesFM expects a categorical indicator valued in {0, 1, 2}.
            Default to 0
    """

    def __init__(self,
                 window_size=256,
                 step=1,
                 pred_len=1,
                 repo_id="google/timesfm-1.0-200m",
                 freq=0):

        self.window_size = window_size
        self.step_size = step
        self.pred_len = pred_len
        self.freq = freq

        self.model = tf.TimesFm(context_len=self.window_size, horizon_len=self.pred_len,
                                input_patch_len=32, output_patch_len=128,
                                num_layers=20, model_dims=1280)

        self.model.load_from_checkpoint(repo_id=repo_id)

    def predict(self, X, index):
        """Forecasting timeseries

        Args:
            X (ndarray):
                input timeseries.
            index (ndarray):
                timestamps array.
        Return:
            ndarray, ndarray, ndarray:
                * forecasted timeseries.
                * array of truncated ground truth with same size as forecasted timeseries.
                * array of timestamps with same size as forecasted timeseries.
        """
        X_windows = rolling_window_sequences(X,
                                             window_size=self.window_size,
                                             step_size=self.step_size)
        frequency_input = [self.freq for _ in range(len(X_windows))]
        y_hat, _ = self.model.forecast(X_windows[:, :, 0], freq=frequency_input)
        y_hat = y_hat[:, 0]
        y = X[self.window_size:]
        index = index[self.window_size:]
        return y_hat, y, index
