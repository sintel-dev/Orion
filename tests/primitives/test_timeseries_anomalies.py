from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from orion.primitives.timeseries_anomalies import (
    _area_error, _dtw_error, _find_sequences, _get_max_errors, _merge_sequences, _point_wise_error,
    _prune_anomalies, find_anomalies, reconstruction_errors, regression_errors)


def test__point_wise_error():
    y = np.array([0.0, 0.1, 1.0, 0.5, 0.1, 0.1, 0.0, 0.5])
    y_hat = np.array([0.1, 2.0, 0.5, 0.0, 3.0, 0.1, 5.0, 0.5])

    expected = np.array([0.1, 1.9, 0.5, 0.5, 2.9, 0.0, 5.0, 0.0])
    returned = _point_wise_error(y, y_hat)

    np.testing.assert_array_equal(returned, expected)


def test__area_error():
    y = np.array([0.0, 0.1, 1.0, 0.5, 0.1, 0.1, 0.0, 0.5])
    y_hat = np.array([0.1, 2.0, 0.5, 0.0, 3.0, 0.1, 5.0, 0.5])
    score_window = 4

    expected = np.array([1.0, 1.7, 1.2, 1.4, 2.15, 5.15, 6.45, 5.0])
    returned = _area_error(y, y_hat, score_window)

    assert_allclose(returned, expected)


def test__dtw_error():
    y = np.array([0.0, 0.1, 1.0, 0.5])
    y_hat = np.array([0.1, 2.0, 0.5, 0.0])
    score_window = 2

    expected = np.array([0.0, 2.0, 0.0, 0.0])
    returned = _dtw_error(y, y_hat, score_window)

    assert_allclose(returned, expected)


class RegressionErrorsTest(TestCase):

    y = np.array([0.0, 0.1, 1.0, 0.5, 0.1, 0.1, 0.0, 0.5]).reshape(-1, 1)
    y_hat = np.array([0.1, 2.0, 0.5, 0.0, 3.0, 0.1, 5.0, 0.5]).reshape(-1, 1)

    def _run(self, smoothing_window, smooth, expected):
        sequences = regression_errors(self.y, self.y_hat, smoothing_window, smooth)

        assert_allclose(sequences, expected, rtol=1e-2)

    def test_no_smooth(self):
        smooth = False
        smoothing_window = 0
        expected = np.array([0.1, 1.9, 0.5, 0.5, 2.9, 0.0, 5.0, 0.0])
        self._run(smoothing_window, smooth, expected)

    def test_smooth(self):
        smooth = True
        smoothing_window = 0.125
        expected = np.array([0.1, 1.9, 0.5, 0.5, 2.9, 0.0, 5.0, 0.0])
        self._run(smoothing_window, smooth, expected)

    def test_smooth_span(self):
        smooth = True
        smoothing_window = 0.25
        expected = np.array([0.1, 1.45, 0.792, 0.595, 2.138, 0.71, 3.571, 1.19])
        self._run(smoothing_window, smooth, expected)


class ReconstructionErrorsTest(TestCase):

    y = np.array([
        [[0.0], [0.1]],
        [[1.0], [0.5]],
        [[0.1], [0.1]],
        [[0.0], [0.5]]
    ])

    y_hat = np.array([
        [[0.1], [2.0]],
        [[0.5], [0.0]],
        [[3.0], [0.1]],
        [[5.0], [0.5]]
    ])

    STEP_SIZE = 1

    def _run(self, score_window, smoothing_window, smooth, rec_error_type, expected):
        sequences, _ = reconstruction_errors(
            self.y, self.y_hat, self.STEP_SIZE, score_window, smoothing_window,
            smooth, rec_error_type)

        assert_allclose(sequences, expected)

    def test_no_smooth(self):
        smooth = False
        score_window = 0
        smoothing_window = 0
        rec_error_type = 'point'
        expected = np.array([0.1, 0.25, 1.4, 2.55, 0.0])
        self._run(score_window, smoothing_window, smooth, rec_error_type, expected)

    def test_smooth(self):
        smooth = True
        score_window = 0
        smoothing_window = 0.25
        rec_error_type = 'point'
        expected = np.array([0.1, 0.25, 1.4, 2.55, 0.0])
        self._run(score_window, smoothing_window, smooth, rec_error_type, expected)

    def test_area(self):
        smooth = False
        score_window = 4
        smoothing_window = 0
        rec_error_type = 'area'
        expected = np.array([0.175, 1.0, 2.975, 4.075, 3.25])
        self._run(score_window, smoothing_window, smooth, rec_error_type, expected)

    def test_dtw(self):
        smooth = False
        score_window = 2
        smoothing_window = 0
        rec_error_type = 'dtw'
        expected = np.array([0.0, 0.35, 1.75, 0.0, 0.0])
        self._run(score_window, smoothing_window, smooth, rec_error_type, expected)


class GetMaxErrorsTest(TestCase):

    MAX_BELOW = 0.1

    def _run(self, errors, sequences, expected):
        sequences = _get_max_errors(errors, sequences, self.MAX_BELOW)

        assert_frame_equal(sequences, expected)

    def test_no_anomalies(self):
        errors = np.array([0.1, 0.0, 0.1, 0.0])
        sequences = np.ndarray((0, 2))
        expected = pd.DataFrame([
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        self._run(errors, sequences, expected)

    def test_one_sequence(self):
        errors = np.array([0.1, 0.2, 0.2, 0.1])
        sequences = np.array([
            [1, 2]
        ])
        expected = pd.DataFrame([
            [0.2, 1, 2],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        self._run(errors, sequences, expected)

    def test_two_sequences(self):
        errors = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.2, 0.1])
        sequences = np.array([
            [1, 3],
            [5, 6]
        ])
        expected = pd.DataFrame([
            [0.3, 1, 3],
            [0.2, 5, 6],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        self._run(errors, sequences, expected)


class PruneAnomaliesTest(TestCase):

    MIN_PERCENT = 0.2

    def _run(self, max_errors, expected):
        sequences = _prune_anomalies(max_errors, self.MIN_PERCENT)

        assert_allclose(sequences, expected)

    def test_no_sequences(self):
        max_errors = pd.DataFrame([
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.ndarray((0, 3))
        self._run(max_errors, expected)

    def test_no_anomalies(self):
        max_errors = pd.DataFrame([
            [0.11, 1, 2],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.ndarray((0, 3))
        self._run(max_errors, expected)

    def test_one_anomaly(self):
        max_errors = pd.DataFrame([
            [0.2, 1, 2],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.array([
            [1, 2, 0.2]
        ])
        self._run(max_errors, expected)

    def test_two_anomalies(self):
        max_errors = pd.DataFrame([
            [0.3, 1, 2],
            [0.2, 4, 5],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.array([
            [1, 2, 0.3],
            [4, 5, 0.2]
        ])
        self._run(max_errors, expected)

    def test_two_out_of_three(self):
        max_errors = pd.DataFrame([
            [0.3, 1, 2],
            [0.22, 4, 5],
            [0.11, 7, 8],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.array([
            [1, 2, 0.3],
            [4, 5, 0.22]
        ])
        self._run(max_errors, expected)

    def test_two_with_a_gap(self):
        max_errors = pd.DataFrame([
            [0.3, 1, 2],
            [0.21, 4, 5],
            [0.2, 7, 8],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.array([
            [1, 2, 0.3],
            [4, 5, 0.21],
            [7, 8, 0.2]
        ])
        self._run(max_errors, expected)


class FindSequencesTest(TestCase):

    THRESHOLD = 0.5
    ANOMALY_PADDING = 1

    def _run(self, errors, expected, expected_max):
        found, max_below = _find_sequences(np.asarray(errors), self.THRESHOLD,
                                           self.ANOMALY_PADDING)

        np.testing.assert_array_equal(found, expected)
        assert max_below == expected_max

    def test__find_sequences_no_sequences(self):
        self._run([0.1, 0.2, 0.3, 0.4], np.ndarray((0, 2)), 0.4)

    def test__find_sequences_all_one_sequence(self):
        self._run([1, 1, 1, 1], [(0, 3)], 0)

    def test__find_sequences_open_end(self):
        self._run([0, 0, 0.4, 1, 1, 1], [(2, 5)], 0)

    def test__find_sequences_open_start(self):
        self._run([1, 1, 1, 0.4, 0, 0], [(0, 3)], 0)

    def test__find_sequences_middle(self):
        self._run([0, 0, 1, 1, 0, 0], [(1, 4)], 0)

    def test__find_sequences_stop(self):
        self._run([1, 0, 0, 0, 1, 1], [(0, 1), (3, 5)], 0)


class MergeSequencesTest(TestCase):

    def _run(self, sequences, expected):
        merged_sequences = _merge_sequences(sequences)

        np.testing.assert_array_equal(merged_sequences, expected)

    def test__merge_sequences_consecutive(self):
        self._run([(1, 2, 0.5), (3, 4, 0.5)], [(1, 4, 0.5)])

    def test__merge_sequences_start_overlap(self):
        self._run([(1, 3, 0.5), (2, 4, 0.5)], [(1, 4, 0.5)])

    def test__merge_sequences_start_end_overlap(self):
        self._run([(1, 4, 0.5), (2, 3, 0.5)], [(1, 4, 0.5)])

    def test__merge_sequences_non_consecutive(self):
        self._run([(1, 2, 0.5), (4, 5, 0.5)], [(1, 2, 0.5), (4, 5, 0.5)])

    def test__merge_sequences_consecutive_different_score(self):
        self._run([(1, 2, 1.0), (3, 4, 0.5)], [(1, 4, 0.75)])

    def test__merge_sequences_consecutive_different_score_and_length(self):
        self._run([(1, 2, 1.0), (3, 4, 0.5)], [(1, 4, 0.75)])


class FindAnomaliesTest(TestCase):

    THRESHOLD = 0.5
    INDEX_SHORT = [1, 2, 3, 4]
    INDEX_LONG = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ANOMALY_PADDING = 1

    def _run(self, errors, expected, index=INDEX_SHORT, window_size=None,
             window_step_size=None, lower_threshold=False, fixed_threshold=False):
        found = find_anomalies(np.asarray(errors), index=index,
                               anomaly_padding=self.ANOMALY_PADDING,
                               window_size=window_size, window_step_size=window_step_size,
                               lower_threshold=lower_threshold,
                               fixed_threshold=fixed_threshold)

        assert_allclose(found, expected)

    def test_find_anomalies_no_anomalies(self):
        self._run([0, 0, 0, 0], np.array([]))

    def test_find_anomalies_one_anomaly(self):
        self._run([0, 0.5, 0.5, 0], np.array([[1., 4., 0.025]]))

    def test_find_anomalies_open_start(self):
        self._run([0.5, 0.5, 0, 0], np.array([[1., 3., 0.025]]))

    def test_find_anomalies_open_end(self):
        self._run([0, 0, 0.5, 0.5], np.array([[2., 4., 0.025]]))

    def test_find_anomalies_two_anomalies(self):
        self._run([0.5, 0, 0.5, 0], np.array([[1., 4., 0.025]]))
        self._run([0, 0.5, 0, 0.5], np.array([[1., 4., 0.025]]))

    def test_find_anomalies_multiple_non_overlapping_thresholds(self):
        self._run([0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0],
                  np.array([[2., 4., 0.025], [6., 8., 0.025]]), index=self.INDEX_LONG,
                  window_size=4, window_step_size=4)

    def test_find_anomalies_multiple_overlapping_thresholds(self):
        self._run([0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0], np.array([[2., 9., 0.025]]),
                  index=self.INDEX_LONG, window_size=4, window_step_size=2)

    def test_find_anomalies_lower_threshold(self):
        self._run([0.5, 0.5, 0, 0], np.array([[1., 4., 0.025]]), lower_threshold=True)

    def test_find_anomalies_fixed_threshold(self):
        self._run([0.5, 0.5, 0, 0], np.array([]), fixed_threshold=True)
