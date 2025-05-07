"""Tests for globalign.random_angles() function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from globalign import random_angles

if TYPE_CHECKING:
    from numpy.typing import NDArray

TEST_SEED: Final = 42


@pytest.mark.parametrize(
    ("center", "radius"), [(0, 180), (90, 90), (45, 45), (0, 90), (0, 360)]
)
def test_random_angles_default_returns_32_angles(center: float, radius: float) -> None:
    result = random_angles(center, None, radius)

    assert isinstance(result, list)
    assert len(result) == 32


@pytest.mark.parametrize(
    ("center", "n"), [(0, 1), (10, 2), (-10, 3), (180, 5), (181, 5)]
)
def test_random_angles_zero_radius(center: float, n: int) -> None:
    result = random_angles(center, None, 0.0, n)

    # As all angles are expected to be equal, removing duplicates should
    # only keep one angle.
    assert len(set(result)) == 1


@pytest.mark.parametrize("radius", [-1.0, -10.0, -180.0])
def test_random_angles_negative_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="radius must be >= 0"):
        random_angles(0.0, None, radius)


@pytest.mark.parametrize("radius", [0.0, 10.0, 180.0])
@pytest.mark.parametrize("center", [0.0, -10.0, 10.0, 180.0])
def test_random_angles_single_angle_return(center: float, radius: float) -> None:
    result = random_angles(center, None, radius, n=1)

    assert len(result) == 1


@pytest.mark.parametrize("n", [0, -1, -10])
def test_random_angles_zero_n_returns_no_angles(n: int) -> None:
    result = random_angles(0.0, None, 180.0, n=n)

    assert len(result) == 0


@pytest.mark.parametrize("n", [1, 3, 10])
@pytest.mark.parametrize("radius", [0.0, 10.0, 180.0])
@pytest.mark.parametrize("center", [0.0, -10.0, 10.0, 180.0])
def test_random_angles_same_output_list_and_number(
    center: float, radius: float, n: int
) -> None:
    rng1 = np.random.default_rng(TEST_SEED)
    rng2 = np.random.default_rng(TEST_SEED)

    return_list_input = random_angles([center], None, radius, n, rng=rng1)
    return_number_input = random_angles(center, None, radius, n, rng=rng2)

    assert return_list_input == return_number_input


@pytest.mark.parametrize("n", [1, 3, 10])
@pytest.mark.parametrize(
    ("center", "center_probs"),
    [
        ([1.1, 2.2, 3.3], np.array([1.0, 0.0, 0.0])),
        ([1.1], np.array([1.0])),
        (2.0, np.array([1.0])),
        ([1.1, 5.4], np.array([0.0, 1.0])),
    ],
)
def test_random_angles_center_probabilities_affect_output(
    center: float | list[float], center_probs: list[NDArray], n: int
) -> None:
    # We set radius to zero to ensure that angles are equal to the
    # center input.
    result = random_angles(center, center_probs, radius=0.0, n=n)
    assert len(result) == n

    # With the probability distribution set to 1.0 for a single center
    # value, we expect all generated values to be equal to that center
    # value.
    max_probability_index = np.argmax(np.atleast_1d(center_probs))
    assert np.all(result == np.atleast_1d(center)[max_probability_index])


@pytest.mark.parametrize(
    ("center", "center_probs"),
    [
        ([1.1, 2.2, 3.3], np.array([1.0] * 10)),
        ([1.1, 2.2, 3.3], np.array([1.0])),
        ([1.1, 2.2, 3.3], np.array([])),
        ([5.0], np.array([1.0] * 10)),
        ([5.0], np.array([])),
        (2.0, np.array([1.0] * 10)),
        (2.0, np.array([])),
    ],
)
def test_random_angles_center_and_probabilities_size_mismatch(
    center: float | list[float], center_probs: list[NDArray]
) -> None:
    with pytest.raises(ValueError, match="centers and center_prob must have same size"):
        random_angles(center, center_probs, radius=180.0)


@given(
    center=st.floats(min_value=-1e6, max_value=1e6),
    radius=st.floats(min_value=0.1, max_value=1e6),
    n=st.integers(min_value=2, max_value=1000),
)
def test_random_angles_length(center: float, radius: float, n: int) -> None:
    result = random_angles(center, None, radius, n)

    assert len(result) == n


@given(
    center=st.floats(min_value=-1e6, max_value=1e6),
    radius=st.floats(min_value=0.1, max_value=1e6),
    n=st.integers(min_value=1, max_value=1000),
)
def test_random_angles_bounds(center: float, radius: float, n: int) -> None:
    result = random_angles(center, None, radius, n)
    min_angle = center - radius
    max_angle = center + radius

    assert all(min_angle <= ang <= max_angle for ang in result)
