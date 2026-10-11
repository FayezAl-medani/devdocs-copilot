"""Linear algebra operations implemented from scratch and with NumPy.

This module provides basic linear algebra operations:
- matmul_naive: 3-loop matrix multiplication without high-level library functions
- matmul_vec: vectorized matrix multiplication using @ operator
- cosine_sim: cosine similarity between two 1D vectors
- cosine_sim_matrix: fully vectorized batch cosine similarity between embedding matrices
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]


def matmul_naive(a: Matrix, b: Matrix) -> Matrix:
    """ضرب مصفوفتين بثلاث حلقات for فقط.

    ممنوع: @ ، np.dot ، np.matmul ، np.einsum.

    Parameters
    ----------
    a : Matrix of shape (n, k)
        First matrix.
    b : Matrix of shape (k, m)
        Second matrix.

    Returns
    -------
    Matrix of shape (n, m)
        Resulting matrix product.

    Raises
    ------
    ValueError
        If inputs are not 2D or inner dimensions do not match.
    """
    if a.ndim != 2 or b.ndim != 2:
        msg = (
            f"Expected 2D matrices, but got shapes {a.shape} (ndim={a.ndim}) "
            f"and {b.shape} (ndim={b.ndim})"
        )
        raise ValueError(msg)

    n, k_a = a.shape
    k_b, m = b.shape

    if k_a != k_b:
        msg = f"Inner dimensions do not match: {k_a} != {k_b} (shapes: {a.shape} and {b.shape})"
        raise ValueError(msg)

    out: Matrix = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            cell_sum = 0.0
            for k in range(k_a):
                cell_sum += float(a[i, k] * b[k, j])
            out[i, j] = cell_sum

    return out


def matmul_vec(a: Matrix, b: Matrix) -> Matrix:
    """ضرب مصفوفتين بعملية NumPy المتجهة (@) مع التحقق من الأبعاد.

    Parameters
    ----------
    a : Matrix of shape (n, k)
        First matrix.
    b : Matrix of shape (k, m)
        Second matrix.

    Returns
    -------
    Matrix of shape (n, m)
        Resulting matrix product.

    Raises
    ------
    ValueError
        If inputs are not 2D or inner dimensions do not match.
    """
    if a.ndim != 2 or b.ndim != 2:
        msg = (
            f"Expected 2D matrices, but got shapes {a.shape} (ndim={a.ndim}) "
            f"and {b.shape} (ndim={b.ndim})"
        )
        raise ValueError(msg)

    if a.shape[1] != b.shape[0]:
        msg = (
            f"Inner dimensions do not match: {a.shape[1]} != {b.shape[0]} "
            f"(shapes: {a.shape} and {b.shape})"
        )
        raise ValueError(msg)

    result = a @ b
    return np.asarray(result, dtype=np.float64)


def cosine_sim(a: Vector, b: Vector) -> float:
    """حساب تشابه جيب التمام بين متجهين: cos(a, b) = (a·b) / (‖a‖ ‖b‖).

    القرار الهندسي:
    إذا كان أحد المتجهين صفرياً (الـ norm يساوي صفراً)، فإن الزاوية تكون غير معرفة رياضياً.
    لتجنب القسمة على صفر أو إنتاج قيمة غير محددة (NaN)، نعيد 0.0 اصطلاحاً
    (باعتبار المتجه الصفري لا يملك اتجاهاً ولا يتشابه مع أي متجه آخر).

    Parameters
    ----------
    a : Vector of shape (dim,)
        First vector.
    b : Vector of shape (dim,)
        Second vector.

    Returns
    -------
    float
        Cosine similarity in range [-1.0, 1.0], or 0.0 if either vector is zero.

    Raises
    ------
    ValueError
        If inputs are not 1D vectors or vector lengths do not match.
    """
    if a.ndim != 1 or b.ndim != 1:
        msg = f"Expected 1D vectors, got shapes {a.shape} and {b.shape}"
        raise ValueError(msg)

    if a.shape[0] != b.shape[0]:
        msg = f"Vector lengths do not match: {a.shape[0]} != {b.shape[0]}"
        raise ValueError(msg)

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot_product = float(np.dot(a, b))
    sim = dot_product / (norm_a * norm_b)

    # قص القيمة لتفادي أخطاء التقريب العشري الطفيفة خارج النطاق [-1, 1]
    return float(np.clip(sim, -1.0, 1.0))


def cosine_sim_matrix(q: Matrix, d: Matrix) -> Matrix:
    """حساب مصفوفة تشابه جيب التمام بين مصفوفتي embeddings بشكل متجه بالكامل.

    q: (n, dim)  d: (m, dim)  ->  (n, m)

    الشروط الهندسية:
    1. بدون أي حلقة for أو while أو list comprehension.
    2. تطبيع كل صف بالقسمة على الـ norm مع keepdims=True ثم ضرب q_norm @ d_norm.T.
    3. التعامل مع الصفوف الصفرية بالكامل بدون إنتاج NaN وبدون أي تحذيرات حسابية.

    Parameters
    ----------
    q : Matrix of shape (n, dim)
        Queries / source embeddings.
    d : Matrix of shape (m, dim)
        Documents / target embeddings.

    Returns
    -------
    Matrix of shape (n, m)
        Similarity matrix where entry (i, j) is the cosine similarity between q[i] and d[j].

    Raises
    ------
    ValueError
        If inputs are not 2D matrices or feature dimensions do not match.
    """
    if q.ndim != 2 or d.ndim != 2:
        msg = f"Expected 2D matrices, got shapes {q.shape} and {d.shape}"
        raise ValueError(msg)

    if q.shape[1] != d.shape[1]:
        msg = f"Embedding dimensions do not match: {q.shape[1]} != {d.shape[1]}"
        raise ValueError(msg)

    # حساب الـ Euclidean Norm لكل صف مع الحفاظ على الأبعاد: shape (n, 1) و (m, 1)
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    d_norms = np.linalg.norm(d, axis=1, keepdims=True)

    # لتجنب القسمة على صفر وظهور تحذيرات RuntimeWarning، نستبدل القيمة 0 بـ 1 في المقام
    # بما أن البسط للصفوف الصفرية هو 0، فإن 0 / 1 = 0 تلقائياً وبأمان تام
    q_safe_norms = np.where(q_norms == 0.0, 1.0, q_norms)
    d_safe_norms = np.where(d_norms == 0.0, 1.0, d_norms)

    q_norm = q / q_safe_norms
    d_norm = d / d_safe_norms

    # ضرب المصفوفات المتجهة للحصول على كل التشابهات الزوجية دفعة واحدة
    sim_matrix = q_norm @ d_norm.T

    # ضمان عدم خروج القيم عن المجال المسموح [-1.0, 1.0] بسبب دقة الفاصلة العائمة
    clipped = np.clip(sim_matrix, -1.0, 1.0)
    return np.asarray(clipped, dtype=np.float64)
