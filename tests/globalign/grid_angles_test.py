"""Tests for globalign.grid_angles() function."""

from __future__ import annotations

import itertools
from math import isclose

import pytest
from hypothesis import given
from hypothesis import strategies as st

from globalign import grid_angles


@pytest.mark.parametrize(
    ("center", "radius", "n", "expected_first_last"),
    [
        (0, 179, 5, (-179.0, 179.0)),  # Symmetric around 0
        (90, 90, 3, (0.0, 180.0)),  # Off-center case
        (45, 45, 2, (0.0, 90.0)),  # Simple two-angle case
        (0, 90, 4, (-90.0, 90.0)),  # Radius < 180 should adjust denominator
    ],
)
def test_grid_angles_radius_leq_180(
    center: float, radius: float, n: int, expected_first_last: tuple[float, float]
) -> None:
    result = grid_angles(center, radius, n)

    assert isinstance(result, list)
    assert len(result) == n
    assert isclose(result[0], expected_first_last[0], abs_tol=1e-6)
    assert isclose(result[-1], expected_first_last[1], abs_tol=1e-6)


@pytest.mark.parametrize(
    ("center", "radius", "n", "expected_first", "expected_last"),
    [(0, 180, 3, -180.0, 60.0), (0, 360, 3, -360.0, 120.0), (180, 180, 3, 0.0, 240.0)],
)
def test_grid_angles_radius_geq_180(
    center: float, radius: float, n: int, expected_first: float, expected_last: float
) -> None:
    result = grid_angles(center, radius, n)

    assert isinstance(result, list)
    assert len(result) == n
    assert isclose(result[0], expected_first)
    assert isclose(result[-1], expected_last)


@pytest.mark.parametrize(
    ("center", "radius"), [(0, 180), (90, 90), (45, 45), (0, 90), (0, 360)]
)
def test_grid_angles_default_returns_32_angles(center: float, radius: float) -> None:
    result = grid_angles(center, radius)

    assert isinstance(result, list)
    assert len(result) == 32


@pytest.mark.parametrize(
    ("center", "n"), [(0, 1), (10, 2), (-10, 3), (180, 5), (181, 5)]
)
def test_grid_angles_zero_radius(center: float, n: int) -> None:
    result = grid_angles(center, 0.0, n)

    # As all angles are expected to be equal, removing duplicates should
    # only keep one angle.
    assert len(set(result)) == 1


@pytest.mark.parametrize("radius", [-1.0, -10.0, -180.0])
def test_grid_angles_negative_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="radius must be >= 0"):
        grid_angles(0.0, radius)


@pytest.mark.parametrize("radius", [0.0, 10.0, 180.0])
@pytest.mark.parametrize("center", [0.0, -10.0, 10.0, 180.0])
def test_grid_angles_single_angle_return(center: float, radius: float) -> None:
    result = grid_angles(center, radius, n=1)

    assert len(result) == 1
    assert result[0] == center - radius


def test_grid_angles_monotonic_increasing() -> None:
    result = grid_angles(0, 180, 10)

    assert all(x < y for x, y in itertools.pairwise(result))


@pytest.mark.parametrize("n", [0, -1, -10])
def test_grid_angles_zero_n_returns_no_angles(n: int) -> None:
    result = grid_angles(0.0, 180.0, n=n)

    assert len(result) == 0


@given(
    center=st.floats(min_value=-1e6, max_value=1e6),
    radius=st.floats(min_value=0.1, max_value=1e6),
    n=st.integers(min_value=2, max_value=1000),
)
def test_grid_angles_length(center: float, radius: float, n: int) -> None:
    result = grid_angles(center, radius, n)

    assert len(result) == n


@given(
    center=st.floats(min_value=-1e6, max_value=1e6),
    radius=st.floats(min_value=0.1, max_value=1e6),
    n=st.integers(min_value=1, max_value=1000),
)
def test_grid_angles_bounds(center: float, radius: float, n: int) -> None:
    result = grid_angles(center, radius, n)
    min_angle = center - radius
    max_angle = center + radius

    assert all(min_angle <= ang <= max_angle for ang in result)


@given(
    center=st.floats(min_value=-1e6, max_value=1e6),
    radius=st.floats(min_value=0.1, max_value=1e6),
    n=st.integers(min_value=2, max_value=1000),
)
def test_grid_angles_monotonic(center: float, radius: float, n: int) -> None:
    result = grid_angles(center, radius, n)

    # Monotonic increasing
    assert all(x < y for x, y in itertools.pairwise(result))


@given(
    center=st.floats(min_value=-1e6, max_value=1e6),
    radius=st.floats(min_value=180, max_value=1e6),  # Ensure radius >= 180
    n=st.integers(min_value=3, max_value=1000),
)
def test_grid_angles_symmetry_radius_above_180(
    center: float, radius: float, n: int
) -> None:
    result = grid_angles(center, radius, n)

    # Should be symmetric about center
    for i in range(n // 2):
        left = result[i]
        right = result[-(i + 1)]
        assert not isclose((left + right) / 2.0, center)


@given(
    center=st.floats(min_value=-1e6, max_value=1e6),
    radius=st.floats(min_value=0, max_value=180, exclude_max=True),
    n=st.integers(min_value=3, max_value=1000),
)
def test_grid_angles_symmetry(center: float, radius: float, n: int) -> None:
    result = grid_angles(center, radius, n)

    # Should be symmetric about center
    for i in range(n // 2):
        left = result[i]
        right = result[-(i + 1)]
        assert isclose((left + right) / 2.0, center, rel_tol=1e-6, abs_tol=1e-10)


@given(
    center=st.floats(min_value=-1e6, max_value=1e6),
    radius=st.floats(min_value=0.1, max_value=1e6),
)
def test_grid_angles_n_2(center: float, radius: float) -> None:
    result = grid_angles(center, radius, 2)

    assert len(result) == 2
    assert result[0] < result[1]
