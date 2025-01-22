"""UniTS

This primitive an implementation of "TIMESFM:
Time Series Foundation Model"
https://arxiv.org/abs/2310.10688

The model implementation can be found at
https://github.com/google-research/timesfm?tab=readme-ov-file
"""

import sys

if sys.version_info < (3, 11):
    msg = (
        '`timesfm` requires Python >= 3.11 and your '
        f'python version is {sys.version}.\n'
        'Make sure you are using Python 3.11 or later.\n'
    )
    raise RuntimeError(msg)

try:
    import timesfm as tf
except ImportError as ie:
    ie.msg += (
        '\n\nIt seems like `timesfm` is not installed.\n'
        'Please install `timesfm` using:\n'
        '\n    pip install orion-ml[pretrained]'
    )
    raise

MAX_LENGTH = 93000


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
        batch_size(int):
            Size of one batch. Default to 32.
        freq (int):
            Frequency. TimesFM expects a categorical indicator valued in {0, 1, 2}.
            Default to 0.
        target (int):
            Index of target column in multivariate case. Default to 0.
    """

    def __init__(self,
                 window_size=256,
                 pred_len=1,
                 repo_id="google/timesfm-1.0-200m-pytorch",
                 batch_size=32,
                 freq=0,
                 target=0):

        self.window_size = window_size
        self.pred_len = pred_len
        self.freq = freq
        self.batch_size = batch_size
        self.target = target

        self.model = tf.TimesFm(hparams=tf.TimesFmHparams(context_len=window_size,
                                                          per_core_batch_size=batch_size,
                                                          horizon_len=pred_len),
                                checkpoint=tf.TimesFmCheckpoint(huggingface_repo_id=repo_id))

    def predict(self, X, force=False):
        """Forecasting timeseries

        Args:
            X (ndarray):
                input timeseries.
        Return:
            ndarray:
                forecasted timeseries.
        """
        frequency_input = [self.freq] * len(X)
        m, n, d = X.shape

        # does not support multivariate
        if d > 1:
            raise ValueError(f'Encountered X with too many channels (channels={d}).')

        # does not support long time series
        if not force and m > (MAX_LENGTH - self.window_size):
            msg = (
                f'`X` has {m} samples, which might result in out of memory issues.\n'
                'If you are sure you want to proceed, set `force=True`.'
            )

            raise MemoryError(msg)

        y_hat, _ = self.model.forecast(X[:, :, 0], freq=frequency_input)
        return y_hat[:, 0]
