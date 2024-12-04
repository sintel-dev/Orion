"""UniTS

This primitive an implementation of "TIMESFM:
Time Series Foundation Model"
https://arxiv.org/abs/2310.10688

The model implementation can be found at
https://github.com/google-research/timesfm?tab=readme-ov-file
"""

import timesfm as tf


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
                 pred_len=1,
                 repo_id="google/timesfm-1.0-200m-pytorch",
                 freq=0):

        self.window_size = window_size
        self.pred_len = pred_len
        self.freq = freq

        self.model = tf.TimesFm(hparams=tf.TimesFmHparams(context_len=window_size,
                                                          per_core_batch_size=32,
                                                          horizon_len=pred_len),
                                checkpoint=tf.TimesFmCheckpoint(huggingface_repo_id=repo_id))
    def predict(self, X):
        """Forecasting timeseries

        Args:
            X (ndarray):
                input timeseries.
        Return:
            ndarray:
                forecasted timeseries.
        """
        frequency_input = [self.freq for _ in range(len(X))]
        y_hat, _ = self.model.forecast(X[:, :, 0], freq=frequency_input)
        return y_hat[:, 0]
