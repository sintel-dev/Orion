from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

from orion.primitives.timeseries_errors import (
    _area_error, _dtw_error, _point_wise_error, reconstruction_errors, regression_errors)


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

    expected = np.array([0.0, 1.9, 0.0, 0.0])
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

        assert_allclose(sequences, expected, rtol=1e-2)

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
        expected = np.array([0.0, 0.27, 1.425, 0.0, 0.0])
        self._run(score_window, smoothing_window, smooth, rec_error_type, expected)
