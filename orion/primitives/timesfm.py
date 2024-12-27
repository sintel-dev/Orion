"""UniTS

This primitive an implementation of "TIMESFM:
Time Series Foundation Model"
https://arxiv.org/abs/2310.10688

The model implementation can be found at
https://github.com/google-research/timesfm?tab=readme-ov-file
"""

import numpy as np

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

    def predict(self, X):
        """Forecasting timeseries

        Args:
            X (ndarray):
                input timeseries.
        Return:
            ndarray:
                forecasted timeseries.
        """
        frequency_input = [self.freq]*len(X)
        d = X.shape[-1]
        
        #Univariate
        if d == 1:
            y_hat, _ = self.model.forecast(X[:, :, 0], freq=frequency_input)
            return y_hat[:, 0]
        
        #Multivariate
        covariates = list(range(d))
        covariates = covariates.remove(self.target)
        X_cont = X[:, :, self.target]
        X_cov = np.delete(X, self.target, axis=2)

        #Append covariates with future values
        m, n, k = X_cov.shape
        X_cov_new = np.zeros((m, n+self.pred_len, k))
        X_cov_new[:, :-self.pred_len, :] = X_cov
        X_cov_new[:-1, -self.pred_len:, :] = X_cov[1:, :self.pred_len, :]

        x_cov = {str(i): X_cov_new[:, :, i] for i in range(k)}
        y_hat, _ = self.model.forecast_with_covariates(
            inputs=X_cont,
            dynamic_numerical_covariates=x_cov,
            freq=frequency_input,
        )
        return np.concatenate(y_hat)
