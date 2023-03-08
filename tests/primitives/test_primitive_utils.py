import numpy as np
import pytest

from orion.primitives.utils import aggregate_rolling_window

@pytest.fixture
def signal():
	return np.array([
		[1, 2, 3],
		[0, 0, 0],
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9]
	])

def test_aggregate_rolling_window(signal):
	# setup
	expected = np.array([1., 1., 1., 2., 5., 7., 9.])

	# run
	output = aggregate_rolling_window(signal)

	# assert
	np.testing.assert_allclose(output, expected)


def test_aggregate_rolling_window_mean(signal):
	# setup
	expected = np.array([1., 1., 4/3, 2., 5., 7., 9.])

	# run
	output = aggregate_rolling_window(signal, "mean")

	# assert
	np.testing.assert_allclose(output, expected)