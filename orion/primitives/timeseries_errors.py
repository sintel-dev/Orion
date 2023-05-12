"""
Time Series error calculation functions.
"""

import math

import numpy as np
import pandas as pd
from pyts.metrics import dtw
from scipy import integrate


def regression_errors(y, y_hat, smoothing_window=0.01, smooth=True,
                      masking_window=0.01, mask=False):
    """Compute an array of absolute errors comparing predictions and expected output.

    If smooth is True, apply EWMA to the resulting array of errors.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.
        smoothing_window (float):
            Optional. Size of the smoothing window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed with EWMA.
            If not given, `True` is used.
        masking_window (float):
            Optional. Size of the masking window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        mask (bool):
            Optional. Mask the start of anomaly scores.
            If not given, `False` is used.

    Returns:
        ndarray:
            Array of errors.
    """
    errors = np.abs(y - y_hat)[:, 0]

    if not smooth:
        return errors

    smoothing_window = max(1, int(len(y) * smoothing_window))
    errors = pd.Series(errors).ewm(span=smoothing_window).mean().values

    if mask:
        mask_length = int(masking_window * len(errors))
        errors[:mask_length] = min(errors)
    return errors


def _point_wise_error(y, y_hat):
    """Compute point-wise error between predicted and expected values.

    The computed error is calculated as the difference between predicted
    and expected values with a rolling smoothing factor.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.

    Returns:
        ndarray:
            An array of smoothed point-wise error.
    """
    return abs(y - y_hat)


def _area_error(y, y_hat, score_window=10):
    """Compute area error between predicted and expected values.

    The computed error is calculated as the area difference between predicted
    and expected values with a smoothing factor.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.

    Returns:
        ndarray:
            An array of area error.
    """
    smooth_y = pd.Series(y).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)
    smooth_y_hat = pd.Series(y_hat).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)

    errors = abs(smooth_y - smooth_y_hat)

    return errors


def _dtw_error(y, y_hat, score_window=10):
    """Compute dtw error between predicted and expected values.

    The computed error is calculated as the dynamic time warping distance
    between predicted and expected values with a smoothing factor.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.

    Returns:
        ndarray:
            An array of dtw error.
    """
    length_dtw = (score_window // 2) * 2 + 1
    half_length_dtw = length_dtw // 2

    # add padding
    y_pad = np.pad(y, (half_length_dtw, half_length_dtw),
                   'constant', constant_values=(0, 0))
    y_hat_pad = np.pad(y_hat, (half_length_dtw, half_length_dtw),
                       'constant', constant_values=(0, 0))

    i = 0
    similarity_dtw = list()
    while i < len(y) - length_dtw:
        true_data = y_pad[i:i + length_dtw]
        true_data = true_data.flatten()

        pred_data = y_hat_pad[i:i + length_dtw]
        pred_data = pred_data.flatten()

        dist = dtw(true_data, pred_data)
        similarity_dtw.append(dist)
        i += 1

    errors = ([0] * half_length_dtw + similarity_dtw +
              [0] * (len(y) - len(similarity_dtw) - half_length_dtw))

    return errors


def reconstruction_errors(y, y_hat, step_size=1, score_window=10, smoothing_window=0.01,
                          smooth=True, rec_error_type='point'):
    """Compute an array of reconstruction errors.

    Compute the discrepancies between the expected and the
    predicted values according to the reconstruction error type.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        step_size (int):
            Optional. Indicating the number of steps between windows in the predicted values.
            If not given, 1 is used.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        smoothing_window (float or int):
            Optional. Size of the smoothing window, when float it is expressed as a proportion
            of the total length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. Reconstruction error types ``["point", "area", "dtw"]``.
            If not given, "point" is used.

    Returns:
        ndarray:
            Array of reconstruction errors.
    """
    if isinstance(smoothing_window, float):
        smoothing_window = min(math.trunc(len(y) * smoothing_window), 200)

    true = [item[0] for item in y.reshape((y.shape[0], -1))]
    for item in y[-1][1:]:
        true.extend(item)

    predictions = []
    predictions_vs = []

    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)

    for i in range(num_errors):
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])
        if intermediate:
            predictions.append(np.median(np.asarray(intermediate)))

            predictions_vs.append([[
                np.min(np.asarray(intermediate)),
                np.percentile(np.asarray(intermediate), 25),
                np.percentile(np.asarray(intermediate), 50),
                np.percentile(np.asarray(intermediate), 75),
                np.max(np.asarray(intermediate))
            ]])

    true = np.asarray(true)
    predictions = np.asarray(predictions)
    predictions_vs = np.asarray(predictions_vs)

    # Compute reconstruction errors
    if rec_error_type.lower() == "point":
        errors = _point_wise_error(true, predictions)

    elif rec_error_type.lower() == "area":
        errors = _area_error(true, predictions, score_window)

    elif rec_error_type.lower() == "dtw":
        errors = _dtw_error(true, predictions, score_window)

    # Apply smoothing
    if smooth:
        errors = pd.Series(errors).rolling(
            smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values

    return errors, predictions_vs
