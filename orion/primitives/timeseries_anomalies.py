"""
Time Series anomaly detection functions.
Some of the implementation is inspired by the paper https://arxiv.org/pdf/1802.04431.pdf
"""

import numpy as np
import pandas as pd
from scipy.optimize import fmin


def deltas(errors, epsilon, mean, std):
    """Compute mean and std deltas.

    delta_mean = mean(errors) - mean(all errors below epsilon)
    delta_std = std(errors) - std(all errors below epsilon)

    Args:
        errors (ndarray):
            Array of errors.
        epsilon (ndarray):
            Threshold value.
        mean (float):
            Mean of errors.
        std (float):
            Standard deviation of errors.

    Returns:
        float, float:
            * delta_mean.
            * delta_std.
    """
    below = errors[errors <= epsilon]
    if not len(below):
        return 0, 0

    return mean - below.mean(), std - below.std()


def count_above(errors, epsilon):
    """Count number of errors and continuous sequences above epsilon.

    Continuous sequences are counted by shifting and counting the number
    of positions where there was a change and the original value was true,
    which means that a sequence started at that position.

    Args:
        errors (ndarray):
            Array of errors.
        epsilon (ndarray):
            Threshold value.

    Returns:
        int, int:
            * Number of errors above epsilon.
            * Number of continuous sequences above epsilon.
    """
    above = errors > epsilon
    total_above = len(errors[above])

    above = pd.Series(above)
    shift = above.shift(1)
    change = above != shift

    total_consecutive = sum(above & change)

    return total_above, total_consecutive


def z_cost(z, errors, mean, std):
    """Compute how bad a z value is.

    The original formula is::

                 (delta_mean/mean) + (delta_std/std)
        ------------------------------------------------------
        number of errors above + (number of sequences above)^2

    which computes the "goodness" of `z`, meaning that the higher the value
    the better the `z`.

    In this case, we return this value inverted (we make it negative), to convert
    it into a cost function, as later on we will use scipy.fmin to minimize it.

    Args:
        z (ndarray):
            Value for which a cost score is calculated.
        errors (ndarray):
            Array of errors.
        mean (float):
            Mean of errors.
        std (float):
            Standard deviation of errors.

    Returns:
        float:
            Cost of z.
    """
    epsilon = mean + z * std

    delta_mean, delta_std = deltas(errors, epsilon, mean, std)
    above, consecutive = count_above(errors, epsilon)

    numerator = -(delta_mean / mean + delta_std / std)
    denominator = above + consecutive ** 2

    if denominator == 0:
        return np.inf

    return numerator / denominator


def _find_threshold(errors, z_range):
    """Find the ideal threshold.

    The ideal threshold is the one that minimizes the z_cost function. Scipy.fmin is used
    to find the minimum, using the values from z_range as starting points.

    Args:
        errors (ndarray):
            Array of errors.
        z_range (list):
            List of two values denoting the range out of which the start points for the
            scipy.fmin function are chosen.

    Returns:
        float:
            Calculated threshold value.
    """
    mean = errors.mean()
    std = errors.std()

    min_z, max_z = z_range
    best_z = min_z
    best_cost = np.inf
    for z in range(min_z, max_z):
        best = fmin(z_cost, z, args=(errors, mean, std), full_output=True, disp=False)
        z, cost = best[0:2]
        if cost < best_cost:
            best_z = z[0]
            best_cost = cost

    return mean + best_z * std


def _fixed_threshold(errors, k=4):
    """Calculate the threshold.

    The fixed threshold is defined as k standard deviations away from the mean.

    Args:
        errors (ndarray):
            Array of errors.

    Returns:
        float:
            Calculated threshold value.
    """
    mean = errors.mean()
    std = errors.std()

    return mean + k * std


def _find_sequences(errors, epsilon, anomaly_padding):
    """Find sequences of values that are above epsilon.

    This is done following this steps:

        * create a boolean mask that indicates which values are above epsilon.
        * mark certain range of errors around True values with a True as well.
        * shift this mask by one place, filing the empty gap with a False.
        * compare the shifted mask with the original one to see if there are changes.
        * Consider a sequence start any point which was true and has changed.
        * Consider a sequence end any point which was false and has changed.

    Args:
        errors (ndarray):
            Array of errors.
        epsilon (float):
            Threshold value. All errors above epsilon are considered an anomaly.
        anomaly_padding (int):
            Number of errors before and after a found anomaly that are added to the
            anomalous sequence.

    Returns:
        ndarray, float:
            * Array containing start, end of each found anomalous sequence.
            * Maximum error value that was not considered an anomaly.
    """
    above = pd.Series(errors > epsilon)
    index_above = np.argwhere(above.values)

    for idx in index_above.flatten():
        above[max(0, idx - anomaly_padding):min(idx + anomaly_padding + 1, len(above))] = True

    shift = above.shift(1).fillna(False)
    change = above != shift

    if above.all():
        max_below = 0
    else:
        max_below = max(errors[~above])

    index = above.index
    starts = index[above & change].tolist()
    ends = (index[~above & change] - 1).tolist()

    if len(ends) == len(starts) - 1:
        ends.append(len(above) - 1)

    return np.array([starts, ends]).T, max_below


def _get_max_errors(errors, sequences, max_below):
    """Get the maximum error for each anomalous sequence.

    Also add a row with the max error which was not considered anomalous.

    Table containing a ``max_error`` column with the maximum error of each
    sequence and the columns ``start`` and ``stop`` with the corresponding start and stop
    indexes, sorted descendingly by the maximum error.

    Args:
        errors (ndarray):
            Array of errors.
        sequences (ndarray):
            Array containing start, end of anomalous sequences
        max_below (float):
            Maximum error value that was not considered an anomaly.

    Returns:
        pandas.DataFrame:
            DataFrame object containing columns ``start``, ``stop`` and ``max_error``.
    """
    max_errors = [{
        'max_error': max_below,
        'start': -1,
        'stop': -1
    }]

    for sequence in sequences:
        start, stop = sequence
        sequence_errors = errors[start: stop + 1]
        max_errors.append({
            'start': start,
            'stop': stop,
            'max_error': max(sequence_errors)
        })

    max_errors = pd.DataFrame(max_errors).sort_values('max_error', ascending=False)
    return max_errors.reset_index(drop=True)


def _prune_anomalies(max_errors, min_percent):
    """Prune anomalies to mitigate false positives.

    This is done by following these steps:

        * Shift the errors 1 negative step to compare each value with the next one.
        * Drop the last row, which we do not want to compare.
        * Calculate the percentage increase for each row.
        * Find rows which are below ``min_percent``.
        * Find the index of the latest of such rows.
        * Get the values of all the sequences above that index.

    Args:
        max_errors (pandas.DataFrame):
            DataFrame object containing columns ``start``, ``stop`` and ``max_error``.
        min_percent (float):
            Percentage of separation the anomalies need to meet between themselves and the
            highest non-anomalous error in the window sequence.

    Returns:
        ndarray:
            Array containing start, end, max_error of the pruned anomalies.
    """
    next_error = max_errors['max_error'].shift(-1).iloc[:-1]
    max_error = max_errors['max_error'].iloc[:-1]

    increase = (max_error - next_error) / max_error
    too_small = increase < min_percent

    if too_small.all():
        last_index = -1
    else:
        last_index = max_error[~too_small].index[-1]

    return max_errors[['start', 'stop', 'max_error']].iloc[0: last_index + 1].values


def _compute_scores(pruned_anomalies, errors, threshold, window_start):
    """Compute the score of the anomalies.

    Calculate the score of the anomalies proportional to the maximum error in the sequence
    and add window_start timestamp to make the index absolute.

    Args:
        pruned_anomalies (ndarray):
            Array of anomalies containing the start, end and max_error for all anomalies in
            the window.
        errors (ndarray):
            Array of errors.
        threshold (float):
            Threshold value.
        window_start (int):
            Index of the first error value in the window.

    Returns:
        list:
            List of anomalies containing start-index, end-index, score for each anomaly.
    """
    anomalies = list()
    denominator = errors.mean() + errors.std()

    for row in pruned_anomalies:
        max_error = row[2]
        score = (max_error - threshold) / denominator
        anomalies.append([row[0] + window_start, row[1] + window_start, score])

    return anomalies


def _merge_sequences(sequences):
    """Merge consecutive and overlapping sequences.

    We iterate over a list of start, end, score triples and merge together
    overlapping or consecutive sequences.
    The score of a merged sequence is the average of the single scores,
    weighted by the length of the corresponding sequences.

    Args:
        sequences (list):
            List of anomalies, containing start-index, end-index, score for each anomaly.

    Returns:
        ndarray:
            Array containing start-index, end-index, score for each anomaly after merging.
    """
    if len(sequences) == 0:
        return np.array([])

    sorted_sequences = sorted(sequences, key=lambda entry: entry[0])
    new_sequences = [sorted_sequences[0]]
    score = [sorted_sequences[0][2]]
    weights = [sorted_sequences[0][1] - sorted_sequences[0][0]]

    for sequence in sorted_sequences[1:]:
        prev_sequence = new_sequences[-1]

        if sequence[0] <= prev_sequence[1] + 1:
            score.append(sequence[2])
            weights.append(sequence[1] - sequence[0])
            weighted_average = np.average(score, weights=weights)
            new_sequences[-1] = (prev_sequence[0], max(prev_sequence[1], sequence[1]),
                                 weighted_average)
        else:
            score = [sequence[2]]
            weights = [sequence[1] - sequence[0]]
            new_sequences.append(sequence)

    return np.array(new_sequences)


def _find_window_sequences(window, z_range, anomaly_padding, min_percent, window_start,
                           fixed_threshold):
    """Find sequences of values that are anomalous.

    We first find the threshold for the window, then find all sequences above that threshold.
    After that, we get the max errors of the sequences and prune the anomalies. Lastly, the
    score of the anomalies is computed.

    Args:
        window (ndarray):
            Array of errors in the window that is analyzed.
        z_range (list):
            List of two values denoting the range out of which the start points for the
            dynamic find_threshold function are chosen.
        anomaly_padding (int):
            Number of errors before and after a found anomaly that are added to the anomalous
            sequence.
        min_percent (float):
            Percentage of separation the anomalies need to meet between themselves and the
            highest non-anomalous error in the window sequence.
        window_start (int):
            Index of the first error value in the window.
        fixed_threshold (bool):
            Indicates whether to use fixed threshold or dynamic threshold.

    Returns:
        ndarray:
            Array containing the start-index, end-index, score for each anomalous sequence
            that was found in the window.
    """
    if fixed_threshold:
        threshold = _fixed_threshold(window)

    else:
        threshold = _find_threshold(window, z_range)

    window_sequences, max_below = _find_sequences(window, threshold, anomaly_padding)
    max_errors = _get_max_errors(window, window_sequences, max_below)
    pruned_anomalies = _prune_anomalies(max_errors, min_percent)
    window_sequences = _compute_scores(pruned_anomalies, window, threshold, window_start)

    return window_sequences


def find_anomalies(errors, index, z_range=(0, 10), window_size=None, window_size_portion=None,
                   window_step_size=None, window_step_size_portion=None, min_percent=0.1,
                   anomaly_padding=50, lower_threshold=False, fixed_threshold=None, inverse=False):
    """Find sequences of error values that are anomalous.

    We first define the window of errors, that we want to analyze. We then find the anomalous
    sequences in that window and store the start/stop index pairs that correspond to each
    sequence, along with its score. Optionally, we can flip the error sequence around the mean
    and apply the same procedure, allowing us to find unusually low error sequences.
    We then move the window and repeat the procedure.
    Lastly, we combine overlapping or consecutive sequences.

    Args:
        errors (ndarray):
            Array of errors.
        index (ndarray):
            Array of indices of the errors.
        z_range (list):
            Optional. List of two values denoting the range out of which the start points for
            the scipy.fmin function are chosen. If not given, (0, 10) is used.
        window_size (int):
            Optional. Size of the window for which a threshold is calculated. If not given,
            `None` is used, which finds one threshold for the entire sequence of errors.
        window_size_portion (float):
            Optional. Specify the size of the window to be a portion of the sequence of errors.
            If not given, `None` is used, and window size is used as is.
        window_step_size (int):
            Optional. Number of steps the window is moved before another threshold is
            calculated for the new window.
        window_step_size_portion (float):
            Optional. Specify the number of steps to be a portion of the window size. If not given,
            `None` is used, and window step size is used as is.
        min_percent (float):
            Optional. Percentage of separation the anomalies need to meet between themselves and
            the highest non-anomalous error in the window sequence. It nof given, 0.1 is used.
        anomaly_padding (int):
            Optional. Number of errors before and after a found anomaly that are added to the
            anomalous sequence. If not given, 50 is used.
        lower_threshold (bool):
            Optional. Indicates whether to apply a lower threshold to find unusually low errors.
            If not given, `False` is used.
        fixed_threshold (bool):
            Optional. Indicates whether to use fixed threshold or dynamic threshold. If not
            given, `False` is used.
        inverse (bool):
            Optional. Indicate whether to take the inverse of errors.

    Returns:
        ndarray:
            Array containing start-index, end-index, score for each anomalous sequence that
            was found.
    """
    window_size = window_size or len(errors)
    if window_size_portion:
        window_size = np.ceil(len(errors) * window_size_portion).astype('int')

    window_step_size = window_step_size or window_size
    if window_step_size_portion:
        window_step_size = np.ceil(window_size * window_step_size_portion).astype('int')

    if inverse:
        errors = max(errors) + (-1 * errors)

    window_start = 0
    window_end = 0
    sequences = list()

    while window_end < len(errors):
        window_end = window_start + window_size
        window = errors[window_start:window_end]
        window_sequences = _find_window_sequences(window, z_range, anomaly_padding, min_percent,
                                                  window_start, fixed_threshold)
        sequences.extend(window_sequences)

        if lower_threshold:
            # Flip errors sequence around mean
            mean = window.mean()
            inverted_window = mean - (window - mean)
            inverted_window_sequences = _find_window_sequences(inverted_window, z_range,
                                                               anomaly_padding, min_percent,
                                                               window_start, fixed_threshold)
            sequences.extend(inverted_window_sequences)

        window_start = window_start + window_step_size

    sequences = _merge_sequences(sequences)

    anomalies = list()

    for start, stop, score in sequences:
        anomalies.append([index[int(start)], index[int(stop)], score])

    return np.asarray(anomalies)
