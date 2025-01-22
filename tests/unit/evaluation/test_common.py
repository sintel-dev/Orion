import pytest

from orion.evaluation.common import _any_overlap, _overlap


@pytest.fixture
def expected():
    return {
        "true": [(1, 5), (1, 5), (1, 5)],
        "false": [(2, 4), (2, 4), (2, 4), (2, 2)]
    }


@pytest.fixture
def observed():
    return {
        "true": [(1, 4), (2, 5), (3, 4)],
        "false": [(1, 2), (4, 5), (5, 6), (2, 2)]
    }


def test__overlap_true(expected, observed):
    expected_true = expected["true"]
    observed_true = observed["true"]
    assert all(_overlap(ex, ob) for ex, ob in zip(expected_true, observed_true))


def test__overlap_false(expected, observed):
    expected_false = expected["false"]
    observed_false = observed["false"]
    assert not all(_overlap(ex, ob) for ex, ob in zip(expected_false, observed_false))


def test__any_overlap_true(expected, observed):
    part = expected["true"][0]
    interval = observed["true"]

    expected_return = 1
    returned = _any_overlap(part, interval)
    assert returned == expected_return


def test__any_overlap_false(expected, observed):
    part = expected["false"][0]
    interval = observed["false"]

    expected_return = 0
    returned = _any_overlap(part, interval)
    assert returned == expected_return
