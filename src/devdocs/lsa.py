"""Latent Semantic Analysis (LSA) implemented via Singular Value Decomposition."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

MatrixF32 = NDArray[np.float32]


class LsaModel:
    """Dimensionality reduction model via truncated Singular Value Decomposition (SVD).

    Decomposes TF-IDF term-document matrix X into X ≈ U_k Σ_k V_k^T, projecting
    documents into a dense k-dimensional latent semantic space where semantically
    associated terms cluster together.

    Parameters
    ----------
    k : int
        Number of latent dimensions to retain.
    """

    def __init__(self, k: int = 128) -> None:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        self.k: int = k
        self.components_: NDArray[np.float32] = np.empty((0, 0), dtype=np.float32)  # V_k^T: (k, D)
        self.singular_values_: NDArray[np.float32] = np.empty(0, dtype=np.float32)  # Σ_k: (k,)
        self.is_fitted: bool = False

    def fit(self, X: MatrixF32, k: int | None = None) -> LsaModel:
        """Fit LSA model on document matrix X using truncated SVD.

        Parameters
        ----------
        X : MatrixF32 of shape (n_documents, vocab_size)
            Input TF-IDF document matrix.
        k : int or None
            Optional override for latent dimensions.

        Returns
        -------
        LsaModel
            Fitted model instance.
        """
        if k is not None:
            if k <= 0:
                raise ValueError(f"k must be a positive integer, got {k}")
            self.k = k

        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix for X, got shape {X.shape}")

        n_docs, vocab_size = X.shape
        max_possible_k = min(n_docs, vocab_size)
        if max_possible_k == 0:
            raise ValueError("Cannot fit LSA on an empty matrix.")

        actual_k = min(self.k, max_possible_k)
        self.k = actual_k

        # Compute economy/thin SVD: X = U Σ V^T
        # U: (N, min(N, D)), S: (min(N, D),), Vt: (min(N, D), D)
        _, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Truncate to top-k components
        self.singular_values_ = np.asarray(S[:actual_k], dtype=np.float32)
        self.components_ = np.asarray(Vt[:actual_k, :], dtype=np.float32)
        self.is_fitted = True
        return self

    def project(self, X: MatrixF32) -> MatrixF32:
        """Project documents into the dense k-dimensional latent semantic space.

        The projection coordinates are given by:
            X_k = X @ V_k = X @ components_.T
        and rows are L2-normalized so that cosine similarity can be computed via dot product.

        Parameters
        ----------
        X : MatrixF32 of shape (m, vocab_size)
            Matrix to project.

        Returns
        -------
        MatrixF32 of shape (m, k)
            L2-normalized latent representation in k dimensions.
        """
        if not self.is_fitted:
            raise ValueError("LsaModel must be fitted before calling project().")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix for X, got shape {X.shape}")

        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X has {X.shape[1]} features, "
                f"but model was fitted on {self.components_.shape[1]} features."
            )

        # Projection: (m, D) @ (D, k) -> (m, k)
        projected = X @ self.components_.T

        # L2-normalize rows for cosine similarity equivalence
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        normalized = projected / safe_norms

        return np.asarray(normalized, dtype=np.float32)

    def reconstruct(self, X: MatrixF32) -> MatrixF32:
        """Reconstruct rank-k approximation of X in original feature space.

        Formula: X_hat = (X @ V_k) @ V_k^T.
        """
        if not self.is_fitted:
            raise ValueError("LsaModel must be fitted before calling reconstruct().")
        projected = X @ self.components_.T
        reconstructed = projected @ self.components_
        return np.asarray(reconstructed, dtype=np.float32)

    def reconstruction_error(self, X: MatrixF32) -> float:
        """Compute Frobenius norm reconstruction error ||X - X_hat||_F."""
        x_hat = self.reconstruct(X)
        return float(np.linalg.norm(X - x_hat))
