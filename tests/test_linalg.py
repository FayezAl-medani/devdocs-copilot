"""Comprehensive test suite for linear algebra operations."""

from __future__ import annotations

import warnings
from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from devdocs.linalg import (
    Matrix,
    Vector,
    cosine_sim,
    cosine_sim_matrix,
    matmul_naive,
    matmul_vec,
)

# -----------------------------------------------------------------------------
# Matmul tests
# -----------------------------------------------------------------------------


def test_matmul_naive_matches_vec() -> None:
    """Check naive and vectorized matmul match on various matrix shapes."""
    rng = np.random.default_rng(seed=101)

    shapes = [
        ((3, 5), (5, 2)),
        ((4, 4), (4, 4)),
        ((10, 3), (3, 8)),
        ((1, 1), (1, 1)),
        ((1, 7), (7, 1)),
        ((7, 1), (1, 5)),
    ]

    for (n, k), (_, m) in shapes:
        a: Matrix = rng.standard_normal((n, k))
        b: Matrix = rng.standard_normal((k, m))

        naive_res = matmul_naive(a, b)
        vec_res = matmul_vec(a, b)

        assert naive_res.shape == (n, m)
        assert vec_res.shape == (n, m)
        assert np.allclose(naive_res, vec_res, atol=1e-10)


def test_matmul_both_match_numpy() -> None:
    """Verify both implementations match NumPy's built-in @ operator."""
    rng = np.random.default_rng(seed=202)
    a: Matrix = rng.standard_normal((7, 4))
    b: Matrix = rng.standard_normal((4, 6))

    expected = a @ b

    assert np.allclose(matmul_naive(a, b), expected, atol=1e-10)
    assert np.allclose(matmul_vec(a, b), expected, atol=1e-10)


def test_matmul_shape_mismatch_raises() -> None:
    """Check ValueError is raised on shape mismatches and invalid dimensions."""
    rng = np.random.default_rng(seed=303)
    a: Matrix = rng.standard_normal((3, 4))
    b: Matrix = rng.standard_normal((5, 2))

    with pytest.raises(ValueError, match="Inner dimensions do not match"):
        matmul_naive(a, b)

    with pytest.raises(ValueError, match="Inner dimensions do not match"):
        matmul_vec(a, b)


def test_matmul_non_2d_raises() -> None:
    """Check ValueError is raised when either input to matmul is not 2-dimensional."""
    mat_2d: Matrix = np.ones((3, 3), dtype=np.float64)
    arr_1d: Vector = np.ones(3, dtype=np.float64)
    arr_3d: Matrix = np.ones((2, 3, 3), dtype=np.float64)

    # 1D inputs
    with pytest.raises(ValueError, match="Expected 2D matrix"):
        matmul_naive(arr_1d, mat_2d)

    with pytest.raises(ValueError, match="Expected 2D matrix"):
        matmul_naive(mat_2d, arr_1d)

    with pytest.raises(ValueError, match="Expected 2D matrix"):
        matmul_vec(arr_1d, mat_2d)

    with pytest.raises(ValueError, match="Expected 2D matrix"):
        matmul_vec(mat_2d, arr_1d)

    # 3D inputs
    with pytest.raises(ValueError, match="Expected 2D matrix"):
        matmul_naive(arr_3d, mat_2d)

    with pytest.raises(ValueError, match="Expected 2D matrix"):
        matmul_vec(arr_3d, mat_2d)


# -----------------------------------------------------------------------------
# Cosine similarity (1D) tests
# -----------------------------------------------------------------------------


def test_cosine_identical_is_one() -> None:
    """Check cosine similarity of identical vectors equals 1.0."""
    v: Vector = np.array([1.0, 2.0, 3.0, -4.0], dtype=np.float64)
    assert cosine_sim(v, v) == pytest.approx(1.0, abs=1e-9)


def test_cosine_orthogonal_is_zero() -> None:
    """Check cosine similarity of perpendicular/orthogonal vectors equals 0.0."""
    v1: Vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v2: Vector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    assert cosine_sim(v1, v2) == pytest.approx(0.0, abs=1e-9)

    v3: Vector = np.array([2.0, 3.0], dtype=np.float64)
    v4: Vector = np.array([-3.0, 2.0], dtype=np.float64)
    assert cosine_sim(v3, v4) == pytest.approx(0.0, abs=1e-9)


def test_cosine_opposite_is_minus_one() -> None:
    """Check cosine similarity of opposing vectors equals -1.0."""
    v1: Vector = np.array([3.0, -1.0, 4.0], dtype=np.float64)
    v2: Vector = -1.0 * v1
    assert cosine_sim(v1, v2) == pytest.approx(-1.0, abs=1e-9)


def test_cosine_scale_invariant() -> None:
    """Check cosine similarity is scale invariant (magnitude does not affect direction)."""
    v: Vector = np.array([1.5, -2.5, 3.5], dtype=np.float64)
    scaled: Vector = 5.0 * v
    assert cosine_sim(v, scaled) == pytest.approx(1.0, abs=1e-9)

    neg_scaled: Vector = -3.0 * v
    assert cosine_sim(v, neg_scaled) == pytest.approx(-1.0, abs=1e-9)


def test_cosine_zero_vector_returns_zero() -> None:
    """Check cosine similarity returns 0.0 when either vector is a zero vector."""
    zero: Vector = np.zeros(4, dtype=np.float64)
    nonzero: Vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    assert cosine_sim(zero, nonzero) == 0.0
    assert cosine_sim(nonzero, zero) == 0.0
    assert cosine_sim(zero, zero) == 0.0


def test_cosine_dim_mismatch_raises() -> None:
    """Check ValueError is raised when vector lengths do not match."""
    v1: Vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v2: Vector = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Vector lengths do not match"):
        cosine_sim(v1, v2)


def test_cosine_non_1d_raises() -> None:
    """Check ValueError is raised when either input to cosine_sim is not 1D."""
    mat_2d: Matrix = np.ones((2, 2), dtype=np.float64)
    v_1d: Vector = np.ones(2, dtype=np.float64)

    with pytest.raises(ValueError, match="Expected 1D vector"):
        cosine_sim(mat_2d, v_1d)

    with pytest.raises(ValueError, match="Expected 1D vector"):
        cosine_sim(v_1d, mat_2d)


# -----------------------------------------------------------------------------
# Batch Cosine similarity (2D matrix) tests
# -----------------------------------------------------------------------------


def test_matrix_shape() -> None:
    """Check matrix batch cosine similarity returns expected (n, m) shape."""
    rng = np.random.default_rng(seed=404)
    q: Matrix = rng.standard_normal((3, 768))
    d: Matrix = rng.standard_normal((1000, 768))

    res = cosine_sim_matrix(q, d)
    assert res.shape == (3, 1000)


def test_matrix_matches_pairwise() -> None:
    """Check batch matrix cosine similarity matches element-wise cosine_sim."""
    rng = np.random.default_rng(seed=505)
    q: Matrix = rng.standard_normal((4, 16))
    d: Matrix = rng.standard_normal((6, 16))

    batch_sim = cosine_sim_matrix(q, d)

    for i in range(q.shape[0]):
        for j in range(d.shape[0]):
            expected = cosine_sim(q[i], d[j])
            assert batch_sim[i, j] == pytest.approx(expected, abs=1e-9)


def test_matrix_zero_row_no_nan() -> None:
    """Check zero rows produce 0.0 similarity without NaNs, Infs, or warnings."""
    q: Matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -3.0],
        ],
        dtype=np.float64,
    )
    d: Matrix = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = cosine_sim_matrix(q, d)

    assert np.isfinite(res).all()
    assert not np.isnan(res).any()
    # Check that zero query row produces zeros
    assert np.all(res[1, :] == 0.0)
    # Check that zero doc row produces zeros
    assert np.all(res[:, 0] == 0.0)


def test_matrix_dimension_mismatch_raises() -> None:
    """Check dimension mismatch between queries and documents raises ValueError."""
    rng = np.random.default_rng(seed=707)
    q: Matrix = rng.standard_normal((5, 128))
    d: Matrix = rng.standard_normal((10, 256))

    with pytest.raises(ValueError, match="Embedding dimensions do not match"):
        cosine_sim_matrix(q, d)


def test_matrix_non_2d_raises() -> None:
    """Check ValueError is raised when either input to cosine_sim_matrix is not 2D."""
    mat_2d: Matrix = np.ones((3, 3), dtype=np.float64)
    arr_1d: Vector = np.ones(3, dtype=np.float64)
    arr_3d: Matrix = np.ones((2, 3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="Expected 2D matrix"):
        cosine_sim_matrix(arr_1d, mat_2d)

    with pytest.raises(ValueError, match="Expected 2D matrix"):
        cosine_sim_matrix(mat_2d, arr_1d)

    with pytest.raises(ValueError, match="Expected 2D matrix"):
        cosine_sim_matrix(arr_3d, mat_2d)


# -----------------------------------------------------------------------------
# Precision, Range, and Clip Guardrail Demonstration
# -----------------------------------------------------------------------------


def test_clip_guardrail_prevents_precision_overshoot() -> None:
    """Demonstrate why np.clip is strictly necessary to prevent floating-point overshoot.

    In IEEE 754 arithmetic, the dot product of a normalized unit vector with itself
    can evaluate to > 1.0 (e.g. 1.0000001 in float32 or 1.0000000000000002 in float64).
    Without clipping, downstream consumers (such as np.arccos) raise RuntimeWarning and return NaN.
    """
    rng = np.random.default_rng(seed=5)
    # The first 64-dim standard normal vector generated by seed 5 overshoots 1.0 without clip
    v = rng.standard_normal(64).astype(np.float32)

    # Recreate unclipped calculation directly:
    v_norm = v / np.linalg.norm(v)
    unclipped_dot = float(np.dot(v_norm, v_norm))

    # Verify that unclipped arithmetic genuinely overshoots 1.0:
    assert unclipped_dot > 1.0, f"Expected float32 rounding > 1.0, got {unclipped_dot}"

    # Verify that arccos on unclipped dot product fails with NaN
    with pytest.warns(RuntimeWarning, match="invalid value encountered in arccos"):
        nan_angle = np.arccos(unclipped_dot)
        assert np.isnan(nan_angle)

    # Now verify that cosine_sim safely clips to exactly 1.0 and allows arccos without error
    clipped_sim = cosine_sim(v, v)
    assert clipped_sim == 1.0
    valid_angle = np.arccos(clipped_sim)
    assert valid_angle == 0.0


def test_main_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify package main entrypoint executes and prints confirmation."""
    from devdocs import main as devdocs_main

    devdocs_main()
    captured = capsys.readouterr()
    assert "devdocs package initialized" in captured.out


# -----------------------------------------------------------------------------
# Extreme Floats & Numerical Stability Tests
# -----------------------------------------------------------------------------


def test_numerical_stability_extreme_large_and_tiny_values() -> None:
    """Verify stability on extreme values without overflow (inf), underflow (zero), or NaNs.

    - Values like 1e200 (float64) or 1e30 (float32) would overflow to inf under naive x**2.
    - Values like 1e-200 (float64) or 1e-30 (float32) would underflow to 0.0 under naive x**2,
      causing non-zero vectors to be treated as zero vectors silently.
    Our implementation scales by max(abs(x)) first, preserving exact angular direction.
    """
    # Extreme float64 cases
    huge_64: Vector = np.array([1e200, 1e200], dtype=np.float64)
    tiny_64: Vector = np.array([1e-200, 1e-200], dtype=np.float64)

    # They have identical directional ratios, so cosine similarity must be exactly 1.0
    sim_huge_tiny = cosine_sim(huge_64, tiny_64)
    assert sim_huge_tiny == pytest.approx(1.0, abs=1e-9)

    sim_huge_self = cosine_sim(huge_64, huge_64)
    assert sim_huge_self == pytest.approx(1.0, abs=1e-9)

    sim_tiny_self = cosine_sim(tiny_64, tiny_64)
    assert sim_tiny_self == pytest.approx(1.0, abs=1e-9)

    # Extreme float32 cases
    huge_32: Vector = np.array([1e30, 1e30], dtype=np.float32)
    tiny_32: Vector = np.array([1e-30, 1e-30], dtype=np.float32)

    assert cosine_sim(huge_32, huge_32) == pytest.approx(1.0, abs=1e-5)
    assert cosine_sim(tiny_32, tiny_32) == pytest.approx(1.0, abs=1e-5)

    # Matrix batch extreme cases
    q_extreme = np.array([[1e200, 1e200], [0.0, 0.0], [1e-200, 1e-200]], dtype=np.float64)
    res_matrix = cosine_sim_matrix(q_extreme, q_extreme)

    assert np.isfinite(res_matrix).all()
    assert res_matrix[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert res_matrix[0, 2] == pytest.approx(1.0, abs=1e-9)  # 1e200 collinear with 1e-200
    assert res_matrix[1, 1] == 0.0  # Zero row is 0.0


# -----------------------------------------------------------------------------
# Dtype Preservation Tests (float32 & float64)
# -----------------------------------------------------------------------------


def test_dtype_preservation() -> None:
    """Ensure matmul and cosine_sim_matrix preserve float32 and float64 dtypes."""
    # float32 inputs
    a32 = np.ones((2, 3), dtype=np.float32)
    b32 = np.ones((3, 4), dtype=np.float32)

    assert matmul_naive(a32, b32).dtype == np.float32
    assert matmul_vec(a32, b32).dtype == np.float32
    assert cosine_sim_matrix(a32, a32).dtype == np.float32

    # float64 inputs
    a64 = np.ones((2, 3), dtype=np.float64)
    b64 = np.ones((3, 4), dtype=np.float64)

    assert matmul_naive(a64, b64).dtype == np.float64
    assert matmul_vec(a64, b64).dtype == np.float64
    assert cosine_sim_matrix(a64, a64).dtype == np.float64

    # Mixed inputs promoted to float64
    assert matmul_naive(a32, b64).dtype == np.float64
    assert matmul_vec(a32, b64).dtype == np.float64
    assert cosine_sim_matrix(a32, a64).dtype == np.float64


# -----------------------------------------------------------------------------
# Property-Based Testing (Hypothesis with arrays strategy)
# -----------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=6),
    k=st.integers(min_value=1, max_value=6),
    m=st.integers(min_value=1, max_value=6),
    dtype_choice=st.sampled_from([np.float32, np.float64]),
    data=st.data(),
)
def test_hypothesis_matmul_naive_matches_vec(
    n: int, k: int, m: int, dtype_choice: type[np.floating[Any]], data: st.DataObject
) -> None:
    """Property-based test: verify naive and vectorized matmul match across arbitrary array data."""
    float_strat = st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        width=32 if dtype_choice == np.float32 else 64,
    )
    a = data.draw(arrays(dtype=dtype_choice, shape=(n, k), elements=float_strat))
    b = data.draw(arrays(dtype=dtype_choice, shape=(k, m), elements=float_strat))

    naive = matmul_naive(a, b)
    vec = matmul_vec(a, b)

    assert naive.dtype == dtype_choice
    assert vec.dtype == dtype_choice

    rtol = 1e-3 if dtype_choice == np.float32 else 1e-7
    atol = 1e-3 if dtype_choice == np.float32 else 1e-7
    assert np.allclose(naive, vec, rtol=rtol, atol=atol)


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=5),
    m=st.integers(min_value=1, max_value=5),
    dim=st.integers(min_value=1, max_value=8),
    dtype_choice=st.sampled_from([np.float32, np.float64]),
    data=st.data(),
)
def test_hypothesis_cosine_sim_matrix_bounded(
    n: int, m: int, dim: int, dtype_choice: type[np.floating[Any]], data: st.DataObject
) -> None:
    """Property-based test: verify cosine_sim_matrix is always in [-1, 1].

    Tests arbitrary float arrays across shapes and dtypes.
    """
    float_strat = st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32 if dtype_choice == np.float32 else 64,
    )
    q = data.draw(arrays(dtype=dtype_choice, shape=(n, dim), elements=float_strat))
    d = data.draw(arrays(dtype=dtype_choice, shape=(m, dim), elements=float_strat))

    sim = cosine_sim_matrix(q, d)

    assert sim.dtype == dtype_choice
    assert sim.shape == (n, m)
    assert np.isfinite(sim).all()
    assert np.all(sim >= -1.0)
    assert np.all(sim <= 1.0)


@settings(max_examples=40, deadline=None)
@given(
    dim=st.integers(min_value=1, max_value=12),
    scale=st.floats(min_value=1e-2, max_value=1e3, allow_nan=False, allow_infinity=False),
    dtype_choice=st.sampled_from([np.float32, np.float64]),
    data=st.data(),
)
def test_hypothesis_cosine_scale_invariant(
    dim: int, scale: float, dtype_choice: type[np.floating[Any]], data: st.DataObject
) -> None:
    """Property-based test: verify cosine_sim is scale-invariant.

    Tests positive scalar multiplication on non-zero arbitrary vectors.
    """
    float_strat = st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        width=32 if dtype_choice == np.float32 else 64,
    )
    v = data.draw(arrays(dtype=dtype_choice, shape=dim, elements=float_strat))

    if float(np.max(np.abs(v))) == 0.0:
        assert cosine_sim(v, v) == 0.0
    else:
        scaled = (scale * v).astype(dtype_choice)
        sim = cosine_sim(v, scaled)
        assert np.isclose(sim, 1.0, atol=1e-4)
