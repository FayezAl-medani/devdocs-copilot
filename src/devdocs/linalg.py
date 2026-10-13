"""Linear algebra operations implemented from scratch and with NumPy.

This module provides basic linear algebra operations:
- matmul_naive: 3-loop matrix multiplication without high-level library functions
- matmul_vec: vectorized matrix multiplication using @ operator
- cosine_sim: cosine similarity between two 1D vectors with numerical stability
- cosine_sim_matrix: fully vectorized batch cosine similarity between embedding matrices
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatType = np.floating[Any]
Matrix = NDArray[FloatType]
Vector = NDArray[FloatType]


def _require_2d(arr: Matrix, name: str = "array") -> tuple[int, int]:
    """Validate that input array is 2D and return its shape (rows, cols).

    Parameters
    ----------
    arr : Matrix
        Input array to validate.
    name : str
        Variable name for descriptive error messaging.

    Returns
    -------
    tuple[int, int]
        Shape of the 2D matrix (rows, cols).

    Raises
    ------
    ValueError
        If the array is not 2-dimensional.
    """
    if arr.ndim != 2:
        msg = f"Expected 2D matrix for {name}, but got shape {arr.shape} (ndim={arr.ndim})"
        raise ValueError(msg)
    return arr.shape[0], arr.shape[1]


def _require_1d(arr: Vector, name: str = "vector") -> int:
    """Validate that input vector is 1D and return its length.

    Parameters
    ----------
    arr : Vector
        Input vector to validate.
    name : str
        Variable name for descriptive error messaging.

    Returns
    -------
    int
        Length of the 1D vector.

    Raises
    ------
    ValueError
        If the vector is not 1-dimensional.
    """
    if arr.ndim != 1:
        msg = f"Expected 1D vector for {name}, but got shape {arr.shape} (ndim={arr.ndim})"
        raise ValueError(msg)
    return arr.shape[0]


def matmul_naive(a: Matrix, b: Matrix) -> Matrix:
    """Compute matrix multiplication using 3 nested loops without vectorization.

    Forbidden: @, np.dot, np.matmul, np.einsum.

    Parameters
    ----------
    a : Matrix of shape (n, k)
        First input matrix.
    b : Matrix of shape (k, m)
        Second input matrix.

    Returns
    -------
    Matrix of shape (n, m)
        Resulting matrix product, preserving promoted input float dtype (e.g. float32 or float64).

    Raises
    ------
    ValueError
        If inputs are not 2D or inner dimensions do not match.
    """
    n, k_a = _require_2d(a, name="a")
    k_b, m = _require_2d(b, name="b")

    if k_a != k_b:
        msg = f"Inner dimensions do not match: {k_a} != {k_b} (shapes: {a.shape} and {b.shape})"
        raise ValueError(msg)

    target_dtype = np.result_type(a.dtype, b.dtype)
    out: Matrix = np.zeros((n, m), dtype=target_dtype)

    for i in range(n):
        for j in range(m):
            cell_sum = 0.0
            for k in range(k_a):
                cell_sum += float(a[i, k] * b[k, j])
            out[i, j] = cell_sum

    return out


def matmul_vec(a: Matrix, b: Matrix) -> Matrix:
    """Compute vectorized matrix multiplication using NumPy's @ operator with dimension checks.

    Parameters
    ----------
    a : Matrix of shape (n, k)
        First input matrix.
    b : Matrix of shape (k, m)
        Second input matrix.

    Returns
    -------
    Matrix of shape (n, m)
        Resulting matrix product, preserving input float dtype.

    Raises
    ------
    ValueError
        If inputs are not 2D or inner dimensions do not match.
    """
    _, k_a = _require_2d(a, name="a")
    k_b, _ = _require_2d(b, name="b")

    if k_a != k_b:
        msg = f"Inner dimensions do not match: {k_a} != {k_b} (shapes: {a.shape} and {b.shape})"
        raise ValueError(msg)

    result = a @ b
    return np.asarray(result, dtype=result.dtype)


def cosine_sim(a: Vector, b: Vector) -> float:
    """Compute cosine similarity between two 1D vectors: cos(a, b) = (a·b) / (‖a‖ ‖b‖).

    Engineering Decisions & Numerical Stability:
    1. Zero Vectors:
       If either vector is zero (norm == 0.0 or max absolute value == 0.0), the angle is
       mathematically undefined. We return 0.0 by convention to represent absence of similarity
       without producing NaNs or warnings.
    2. Numerical Stability (Extreme Floats):
       To prevent catastrophic intermediate overflow (e.g. elements ~ 1e200 squared produce inf)
       and underflow (e.g. elements ~ 1e-200 squared produce 0.0), vectors are scaled by their
       maximum absolute component before computing Euclidean norm. Because max(|scaled|) == 1.0,
       1.0 <= ||scaled||^2 <= dim, mathematically guaranteeing that squaring neither overflows
       nor underflows.
    3. IEEE 754 Floating-Point Guardrail:
       Due to floating-point rounding errors in norm division and dot products, unclipped results
       for collinear vectors can evaluate to 1.0000000000000002 or -1.0000000000000002.
       We clip to [-1.0, 1.0] to preserve mathematical validity and prevent NaNs in downstream
       arccos.

    Parameters
    ----------
    a : Vector of shape (dim,)
        First input vector.
    b : Vector of shape (dim,)
        Second input vector.

    Returns
    -------
    float
        Cosine similarity in range [-1.0, 1.0], or 0.0 if either vector is zero.

    Raises
    ------
    ValueError
        If inputs are not 1D vectors or vector lengths do not match.
    """
    dim_a = _require_1d(a, name="a")
    dim_b = _require_1d(b, name="b")

    if dim_a != dim_b:
        msg = f"Vector lengths do not match: {dim_a} != {dim_b}"
        raise ValueError(msg)

    max_a = float(np.max(np.abs(a)))
    max_b = float(np.max(np.abs(b)))

    if max_a == 0.0 or max_b == 0.0:
        return 0.0

    scaled_a = a / max_a
    scaled_b = b / max_b

    norm_a = float(np.linalg.norm(scaled_a))
    norm_b = float(np.linalg.norm(scaled_b))

    dot_product = float(np.dot(scaled_a, scaled_b))
    sim = dot_product / (norm_a * norm_b)

    return float(np.clip(sim, -1.0, 1.0))


def cosine_sim_matrix(q: Matrix, d: Matrix) -> Matrix:
    """Compute fully vectorized batch cosine similarity between embedding matrices.

    q: (n, dim)  d: (m, dim)  ->  (n, m)

    Engineering Constraints & Numerical Stability:
    1. Zero Loops: Fully vectorized without any for, while, or comprehension constructs.
    2. Numerical Stability (Extreme Floats):
       Each row is scaled by its maximum absolute component before computing norm.
       This prevents overflow on huge magnitudes (e.g. 1e200 / 1e30 in float32)
       and underflow on tiny magnitudes (e.g. 1e-200 / 1e-30 in float32).
    3. Safe Zero-Row Handling:
       For zero rows, max absolute value is 0.0. We substitute 1.0 in the divisor
       via np.where, yielding exactly 0.0 without generating NaNs or RuntimeWarnings.
    4. Dtype Preservation:
       Output preserves the promoted floating-point dtype of the inputs (e.g. float32 -> float32).
    5. IEEE 754 Floating-Point Guardrail:
       Values are clipped to [-1.0, 1.0] to safeguard against precision overshoots.

    Parameters
    ----------
    q : Matrix of shape (n, dim)
        Query / source embedding matrix.
    d : Matrix of shape (m, dim)
        Document / target embedding matrix.

    Returns
    -------
    Matrix of shape (n, m)
        Pairwise cosine similarity matrix with values bounded to [-1.0, 1.0].

    Raises
    ------
    ValueError
        If inputs are not 2D matrices or feature dimensions do not match.
    """
    _, dim_q = _require_2d(q, name="q")
    _, dim_d = _require_2d(d, name="d")

    if dim_q != dim_d:
        msg = f"Embedding dimensions do not match: {dim_q} != {dim_d}"
        raise ValueError(msg)

    target_dtype = np.result_type(q.dtype, d.dtype)

    # 1. Scale each row by its max absolute value for numerical stability
    q_max = np.max(np.abs(q), axis=1, keepdims=True)
    d_max = np.max(np.abs(d), axis=1, keepdims=True)

    q_safe_max = np.where(q_max == 0.0, 1.0, q_max)
    d_safe_max = np.where(d_max == 0.0, 1.0, d_max)

    q_scaled = q / q_safe_max
    d_scaled = d / d_safe_max

    # 2. Compute Euclidean norm of scaled vectors (guaranteed no overflow/underflow)
    q_scaled_norms = np.linalg.norm(q_scaled, axis=1, keepdims=True)
    d_scaled_norms = np.linalg.norm(d_scaled, axis=1, keepdims=True)

    # 3. Safe normalization (zero rows mapped to 0 / 1 = 0)
    q_safe_norms = np.where(q_max == 0.0, 1.0, q_scaled_norms)
    d_safe_norms = np.where(d_max == 0.0, 1.0, d_scaled_norms)

    q_norm = q_scaled / q_safe_norms
    d_norm = d_scaled / d_safe_norms

    # 4. Batch dot product
    sim_matrix = q_norm @ d_norm.T

    # 5. Guardrail clipping
    clipped = np.clip(sim_matrix, -1.0, 1.0)
    return np.asarray(clipped, dtype=target_dtype)
