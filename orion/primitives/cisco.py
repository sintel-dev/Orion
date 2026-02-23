"""
This primitive is an implementation of Cisco's Time Series
Foundation Model for timeseries forecasting.

The model implementation can be found at
https://arxiv.org/pdf/2511.19841
"""

import os

import numpy as np
import torch
from torch import nn

from huggingface_hub import snapshot_download
from timesfm import pytorch_patched_decoder as ppd
from timesfm.timesfm_base import linear_interpolation, strip_leading_nans


def build_coarse_context(series: np.ndarray, max_coarse_ctx: int = 512, block: int = 60):
    """
    Construct coarse (long-term) context by aggregating fine samples.

    Takes the rightmost (max_coarse_ctx * block) raw samples, partitions them
    into consecutive non-overlapping blocks, and computes the mean of each block.

    Args:
        series (ndarray):
            Array of fine-resolution time series data.
        max_coarse_ctx (int):
            Maximum number of coarse points to return (default: 512).
        block (int):
            Number of fine samples to aggregate into one coarse sample (default: 60).
            For example, block=60 aggregates 60 minutes into 1 hour.
    Returns:
        List of floats representing coarse means with length <= max_coarse_ctx.
    """
    needed_raw = max_coarse_ctx * block
    raw_slice = series[-needed_raw:]

    remainder = len(raw_slice) % block
    if remainder != 0:
        raw_slice = raw_slice[remainder:]

    coarse = []
    for i in range(0, len(raw_slice), block):
        block_vals = raw_slice[i:i + block]
        if len(block_vals) < block:
            break
        coarse.append(float(np.mean(block_vals)))

    return coarse[-max_coarse_ctx:]


def build_fine_context(series: np.ndarray, fine_len: int = 512):
    """
    Extract fine (short-term) context from the rightmost samples.

    Args:
        series (ndarray):
            Array of fine-resolution time series data.
        fine_len (int):
            Desired length of fine-level context to extract (default: 512).

    Returns:
        List of floats representing the fine-level context of length <= fine_len.
    """
    if isinstance(series, np.ndarray):
        series = series.tolist()
    return series[-fine_len:]


def build_multi_resolution_context(series: np.ndarray,
                                   agg_factor: int = 60,
                                   max_coarse_ctx: int = 512,
                                   max_fine_ctx: int = 512) -> tuple[list[float], list[float]]:
    """
    Build both coarse and fine resolution contexts from a time series.
    This is the main function for creating multi-resolution inputs for Cisco.

    Args:
        series: Array of fine-resolution time series data.
        agg_factor: Aggregation factor to form coarse context from fine context
            (default: 60, e.g., minutes to hours).
        max_coarse_ctx: Maximum number of coarse points (default: 512).
        max_fine_ctx: Maximum number of fine points (default: 512).

    Returns:
        Tuple of:
            - List of floats representing the coarse (long-term) context.
            - List of floats representing the fine (short-term) context.
    """
    coarse_ctx = build_coarse_context(
        series,
        max_coarse_ctx=max_coarse_ctx,
        block=agg_factor
    )
    fine_ctx = build_fine_context(series, fine_len=max_fine_ctx)

    return coarse_ctx, fine_ctx


class MinimalCiscoModel(ppd.PatchedTimeSeriesDecoder):
    """Multi-resolution TimesFM with resolution embeddings and special token."""

    def __init__(self, config):
        super().__init__(config)
        self.multi_resolution = nn.Embedding(2, config.hidden_size)
        self.special_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    @staticmethod
    def reverse_transform_two_segments(outputs, stats_coarse, stats_fine, N_coarse, N_fine):
        """
        Faster + simpler than mask-mapping.
        outputs: [B, N_total, H, Q]
        stats_*: (mu, sigma) from _preprocess_input (shapes vary: [B], [B,1], [B,1,1], ...)
        Applies y = y*sigma + mu on:
          - coarse tokens: [0 : N_coarse)
          - fine tokens:   [N_coarse+1 : N_coarse+1+N_fine)   (skips special token at N_coarse)
        """
        B = outputs.shape[0]
        dtype = outputs.dtype

        mu_c = stats_coarse[0].to(dtype).reshape(B, -1)[:, 0]
        sg_c = stats_coarse[1].to(dtype).reshape(B, -1)[:, 0]
        mu_f = stats_fine[0].to(dtype).reshape(B, -1)[:, 0]
        sg_f = stats_fine[1].to(dtype).reshape(B, -1)[:, 0]

        outputs[:, 0:N_coarse, :, :] = outputs[:, 0:N_coarse, :, :] * \
            sg_c.view(B, 1, 1, 1) + mu_c.view(B, 1, 1, 1)

        s1 = N_coarse + 1
        e1 = N_coarse + 1 + N_fine
        outputs[:, s1:e1, :, :] = outputs[:, s1:e1, :, :] * \
            sg_f.view(B, 1, 1, 1) + mu_f.view(B, 1, 1, 1)

        return outputs


def preprocess_series(series, agg_factor=60):
    """
    Return RAW (not normalized) padded contexts + pad masks.
    Let timesfm `_preprocess_input` do its own normalization and return stats.

    Args:
        series: 1D numpy array
        agg_factor: Aggregation factor (default 60 = minute->hour)

    Returns:
        coarse, coarse_mask, fine, fine_mask
    """
    series = np.asarray(series, dtype=np.float32)
    if not np.isfinite(series).all():
        series = np.where(np.isfinite(series), series, np.nan)
    series = strip_leading_nans(series)
    series = linear_interpolation(series)

    coarse_list, fine_list = build_multi_resolution_context(
        series,
        agg_factor=agg_factor,
        max_coarse_ctx=512,
        max_fine_ctx=512
    )

    coarse = torch.tensor(coarse_list, dtype=torch.float32)
    fine = torch.tensor(fine_list, dtype=torch.float32)

    if len(coarse) < 512:
        pad_len = 512 - len(coarse)
        coarse = torch.cat([torch.zeros(pad_len), coarse])
        coarse_mask = torch.cat([torch.ones(pad_len), torch.zeros(len(coarse_list))])
    else:
        coarse = coarse[-512:]
        coarse_mask = torch.zeros(512)

    if len(fine) < 512:
        pad_len = 512 - len(fine)
        fine = torch.cat([torch.zeros(pad_len), fine])
        fine_mask = torch.cat([torch.ones(pad_len), torch.zeros(len(fine_list))])
    else:
        fine = fine[-512:]
        fine_mask = torch.zeros(512)

    return coarse, coarse_mask, fine, fine_mask


class CiscoInference:
    """Minimal Cisco TimesFM inference."""

    def __init__(self, repo_id="cisco-ai/cisco-time-series-model-1.0-preview"):
        """Load model from HuggingFace."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Downloading from {repo_id}...")
        checkpoint_dir = snapshot_download(repo_id)
        checkpoint_path = os.path.join(checkpoint_dir, "torch_model.pt")

        self.config = ppd.TimesFMConfig(
            num_layers=50,
            num_heads=16,
            hidden_size=1280,
            intermediate_size=1280,
            patch_len=32,
            horizon_len=128,
            head_dim=80,
            quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            use_positional_embedding=False,
        )

        self.model = MinimalCiscoModel(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def forecast(self, series_list, horizon_len=128, agg_factor=60):
        """
        Forecast time series.

        Args:
            series_list: List of 1D numpy arrays
            horizon_len: Steps to forecast (default 128)
            agg_factor: Aggregation factor (default 60)

        Returns:
            List of dicts with 'mean' and 'quantiles'
        """
        assert horizon_len <= self.config.horizon_len, (
            f"horizon_len must be <= model horizon {self.config.horizon_len}"
        )

        results = []

        with torch.no_grad():
            for series in series_list:
                if not isinstance(series, np.ndarray):
                    series = np.array(series, dtype=np.float32)

                coarse_raw, c_padmask, fine_raw, f_padmask = preprocess_series(
                    series, agg_factor=agg_factor)

                coarse = coarse_raw.unsqueeze(0).unsqueeze(-1).to(self.device)
                c_pad = c_padmask.unsqueeze(0).unsqueeze(-1).to(self.device)
                fine = fine_raw.unsqueeze(0).unsqueeze(-1).to(self.device)
                f_pad = f_padmask.unsqueeze(0).unsqueeze(-1).to(self.device)

                coarse_proc, c_pad_proc, stats_coarse, _ = self.model._preprocess_input(
                    coarse, c_pad)
                fine_proc, f_pad_proc, stats_fine, _ = self.model._preprocess_input(fine, f_pad)

                B = coarse_proc.shape[0]
                N_coarse = coarse_proc.shape[1]
                N_fine = fine_proc.shape[1]
                D = coarse_proc.shape[2]

                # padding must be 2D [B,N] for stacked_transformer
                if c_pad_proc.ndim == 3:
                    c_pad_proc = c_pad_proc.squeeze(-1)
                if f_pad_proc.ndim == 3:
                    f_pad_proc = f_pad_proc.squeeze(-1)

                spec_token = self.model.special_token.expand(B, 1, D)
                spec_pad = torch.zeros(B, 1, device=self.device, dtype=c_pad_proc.dtype)

                model_input = torch.cat([coarse_proc, spec_token, fine_proc], dim=1)
                padding = torch.cat([c_pad_proc, spec_pad, f_pad_proc], dim=1)

                res_ids = torch.cat([
                    torch.zeros(N_coarse, dtype=torch.long, device=self.device),
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    torch.ones(N_fine, dtype=torch.long, device=self.device),
                ]).unsqueeze(0).expand(B, -1)

                model_input = model_input + self.model.multi_resolution(res_ids)

                freq = torch.zeros(B, 1, dtype=torch.long, device=self.device)
                model_input = model_input + self.model.freq_emb(freq)

                output = self.model.stacked_transformer(model_input, padding)
                logits = self.model.horizon_ff_layer(output)

                num_outputs = len(self.config.quantiles) + 1
                preds = logits.view(B, -1, self.config.horizon_len, num_outputs)

                preds = self.model.reverse_transform_two_segments(
                    preds, stats_coarse, stats_fine, N_coarse, N_fine
                )

                fine_token_idx = N_coarse + N_fine
                token_pred = preds[:, fine_token_idx, :horizon_len, :]

                token_np = token_pred[0].cpu().numpy()
                quantiles = {str(q): token_np[:, i + 1]
                             for i, q in enumerate(self.config.quantiles)}
                results.append({
                    "mean": token_np[:, 0],
                    "quantiles": quantiles,
                })

        return results


class Cisco:
    """Cisco model for timeseries forecasting (completely standalone implementation).

    This version includes all necessary code directly - no external module dependencies
    beyond TimesFM and standard libraries.

    Args:
        window_size (int):
            Window size of each sample. Default to 30720.
            Note: Cisco expects a large window size because it uses long term context.
        pred_len (int):
            Prediction horizon length. Default to 128.
        repo_id (str):
            HuggingFace repository ID. Default to "cisco-ai/cisco-time-series-model-1.0-preview"
        agg_factor (int):
            Aggregation factor for coarse context. Default to 60 (minute->hour).
        target (int):
            Index of target column in multivariate case. Default to 0.
        return_quantile (str or None):
            If specified, return this quantile instead of mean (e.g., "0.5" for median).
            Default to None (returns mean).
    """

    def __init__(
        self,
        window_size=30720,
        pred_len=128,
        repo_id="cisco-ai/cisco-time-series-model-1.0-preview",
        agg_factor=60,
        target=0,
        return_quantile=None,
    ):
        self.window_size = int(window_size)
        self.pred_len = int(pred_len)
        self.agg_factor = int(agg_factor)
        self.target = int(target)
        self.return_quantile = return_quantile

        self.model = CiscoInference(repo_id=repo_id)

    def predict(self, X):
        """Forecast.

        Args:
            X (ndarray): shape (n_windows, window_size, n_features)
                Each window is a time series with window_size timesteps
                and n_features variables.

        Returns:
            ndarray: shape (n_windows, pred_len)
                Predicted values for each window.
        """
        n_windows = X.shape[0]

        series_list = []
        for i in range(n_windows):
            series = X[i, :self.window_size, self.target].astype(np.float32)
            series_list.append(series)

        forecast_results = self.model.forecast(
            series_list,
            horizon_len=self.pred_len,
            agg_factor=self.agg_factor
        )

        if self.return_quantile is not None:
            preds = np.stack([
                f["quantiles"][self.return_quantile]
                for f in forecast_results
            ], axis=0)
        else:
            preds = np.stack([
                f["mean"]
                for f in forecast_results
            ], axis=0)

        return preds
