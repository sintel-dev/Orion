import importlib
import unittest
from unittest.mock import patch

import numpy as np

import timesfm as tf
from orion.primitives.timesfm import MAX_LENGTH, TimesFM


class TestTimesFMImport(unittest.TestCase):

    @patch('sys.version_info', (3, 10))
    def test_runtime_error_python_version_less_than_3_11(self):
        with self.assertRaises(RuntimeError) as context:
            import orion.primitives.timesfm
            importlib.reload(orion.primitives.timesfm)

        self.assertIn('requires Python >= 3.11', str(context.exception))
        self.assertIn('python version is', str(context.exception))

    @patch('sys.version_info', (3, 11))
    @patch('builtins.__import__', side_effect=ImportError())
    def test_import_error_timesfm_not_installed(self, mock_import):
        # simulate Python version 3.11 and timesfm not installed
        with self.assertRaises(ImportError):
            import orion.primitives.timesfm  # noqa


class TestTimesFMPredict(unittest.TestCase):

    def setUp(self):
        self.model = TimesFM()

    def test_value_error_multivariate_input(self):
        # create a multivariate input with more than one channel
        X = np.random.rand(10, 5, 2)  # Shape (m, n, d) with d > 1

        with self.assertRaises(ValueError) as context:
            self.model.predict(X)

        self.assertIn('Encountered X with too many channels', str(context.exception))

    def test_memory_error_long_time_series(self):
        # create a long time series input
        m = MAX_LENGTH - self.model.window_size + 1
        X = np.random.rand(m, 5, 1)  # Shape (m, n, d) with d = 1

        with self.assertRaises(MemoryError) as context:
            self.model.predict(X)

        self.assertIn('might result in out of memory issues', str(context.exception))

    @patch.object(tf.TimesFm, 'forecast', return_value=(np.random.rand(10, 1), None))
    def test_no_memory_error_with_force(self, mock_forecast):
        # create a long time series input
        m = MAX_LENGTH - self.model.window_size + 1
        X = np.random.rand(m, 5, 1)  # Shape (m, n, d) with d = 1

        # should not raise MemoryError when force=True
        try:
            self.model.predict(X, force=True)
        except MemoryError:
            self.fail("predict() raised MemoryError unexpectedly with force=True")

        mock_forecast.assert_called_once()
