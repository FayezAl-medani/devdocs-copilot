"""Comprehensive test suite for linear algebra operations."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from devdocs.linalg import (
    Matrix,
    Vector,
    cosine_sim,
    cosine_sim_matrix,
    matmul_naive,
    matmul_vec,
)


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

    # 1D or 3D inputs should also raise ValueError
    invalid_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Expected 2D"):
        matmul_naive(invalid_1d, a)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected 2D"):
        matmul_vec(invalid_1d, a)  # type: ignore[arg-type]


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


def test_matrix_values_in_range() -> None:
    """Check all computed similarity values fall strictly within [-1.0, 1.0]."""
    rng = np.random.default_rng(seed=606)
    q: Matrix = rng.standard_normal((20, 64))
    d: Matrix = rng.standard_normal((30, 64))

    res = cosine_sim_matrix(q, d)

    assert np.all(res >= -1.0)
    assert np.all(res <= 1.0)


def test_matrix_dimension_mismatch_raises() -> None:
    """Check dimension mismatch between queries and documents raises ValueError."""
    rng = np.random.default_rng(seed=707)
    q: Matrix = rng.standard_normal((5, 128))
    d: Matrix = rng.standard_normal((10, 256))

    with pytest.raises(ValueError, match="Embedding dimensions do not match"):
        cosine_sim_matrix(q, d)


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=8),
    k=st.integers(min_value=1, max_value=8),
    m=st.integers(min_value=1, max_value=8),
)
def test_hypothesis_matmul_naive_matches_vec(n: int, k: int, m: int) -> None:
    """Property-based test: verify matmul_naive equals matmul_vec for arbitrary shapes."""
    rng = np.random.default_rng(seed=n * 100 + k * 10 + m)
    a: Matrix = rng.uniform(-10.0, 10.0, size=(n, k)).astype(np.float64)
    b: Matrix = rng.uniform(-10.0, 10.0, size=(k, m)).astype(np.float64)

    naive = matmul_naive(a, b)
    vec = matmul_vec(a, b)

    assert np.allclose(naive, vec, atol=1e-8)
