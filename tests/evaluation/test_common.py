import numpy as np
import pytest

from orion.evaluation.common import _any_overlap, _overlap, _overlap_segment


@pytest.fixture
def expected():
    return (1, 5)


@pytest.fixture
def observed():
    return (2, 5)


def test__overlap(expected, observed):
    assert _overlap(expected, observed)


def test__any_overlap(expected, observed):
    part = expected
    interval = [observed]

    expected_return = 1
    returned = _any_overlap(part, interval)
    assert returned == expected_return


def test__overlap_segment(expected, observed):
    expected_return = (None, 0, 0, 1)
    np.testing.assert_array_equal(np.array(_overlap_segment([expected], [observed])),
                                  np.array(expected_return))
