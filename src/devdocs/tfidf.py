"""Pure NumPy TF-IDF Vectorizer implemented from first principles."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from devdocs.corpus import Chunk
from devdocs.text import tokenize

MatrixF32 = NDArray[np.float32]


class TfidfVectorizer:
    """Computes Term Frequency - Inverse Document Frequency matrices using pure NumPy.

    Parameters
    ----------
    min_df : int
        Minimum document frequency required to include a term in vocabulary.
    max_features : int or None
        Maximum vocabulary size. If vocabulary exceeds this threshold,
        terms are ranked by document frequency and truncated.
    """

    def __init__(self, min_df: int = 1, max_features: int | None = 8000) -> None:
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_: dict[str, int] = {}
        self.idf_: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self.is_fitted: bool = False

    def _extract_texts(self, raw_documents: Sequence[Chunk | str]) -> list[str]:
        """Extract text strings from either Chunk instances or plain strings."""
        return [doc.text if isinstance(doc, Chunk) else doc for doc in raw_documents]

    def fit(self, raw_documents: Sequence[Chunk | str]) -> TfidfVectorizer:
        """Fit vocabulary and compute inverse document frequencies (IDF).

        Formula:
            idf(t) = ln((1 + N) / (1 + df(t))) + 1.0

        Parameters
        ----------
        raw_documents : Sequence[Chunk | str]
            Training collection of documents.

        Returns
        -------
        TfidfVectorizer
            Fitted instance.
        """
        texts = self._extract_texts(raw_documents)
        n_docs = len(texts)
        if n_docs == 0:
            raise ValueError("Cannot fit TfidfVectorizer on an empty document collection.")

        # Compute document frequency df(t) for every term
        df_counts: Counter[str] = Counter()
        for text in texts:
            unique_terms = set(tokenize(text))
            for term in unique_terms:
                df_counts[term] += 1

        # Filter by min_df
        filtered_terms = [
            (term, count) for term, count in df_counts.items() if count >= self.min_df
        ]

        # Rank by df descending, then alphabetically for deterministic ordering
        filtered_terms.sort(key=lambda item: (-item[1], item[0]))

        if self.max_features is not None and len(filtered_terms) > self.max_features:
            filtered_terms = filtered_terms[: self.max_features]

        # Build vocabulary mapping: term -> column index
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(filtered_terms)}
        vocab_size = len(self.vocabulary_)

        # Compute smoothed IDF vector
        idf_values = np.empty(vocab_size, dtype=np.float32)
        for term, col_idx in self.vocabulary_.items():
            df = df_counts[term]
            idf_values[col_idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        self.idf_ = idf_values
        self.is_fitted = True
        return self

    def transform(self, raw_documents: Sequence[Chunk | str]) -> MatrixF32:
        """Transform documents into L2-normalized TF-IDF matrix.

        Out-of-vocabulary (OOV) terms are silently ignored.
        Documents containing only OOV terms produce zero vectors without NaNs.

        Parameters
        ----------
        raw_documents : Sequence[Chunk | str]
            Documents to vectorize.

        Returns
        -------
        MatrixF32
            Matrix of shape (n_documents, vocab_size) with dtype float32.
        """
        if not self.is_fitted:
            raise ValueError("TfidfVectorizer must be fitted before calling transform().")

        texts = self._extract_texts(raw_documents)
        n_docs = len(texts)
        vocab_size = len(self.vocabulary_)

        if n_docs == 0 or vocab_size == 0:
            return np.zeros((n_docs, vocab_size), dtype=np.float32)

        matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)

        for row_idx, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                col_idx = self.vocabulary_.get(term)
                if col_idx is not None:
                    # Raw Term Frequency multiplied by IDF
                    matrix[row_idx, col_idx] = count * self.idf_[col_idx]

        # L2 row-normalization for cosine similarity equivalence:
        # ||x||_2 = sqrt(sum(x_i^2))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        normalized_matrix = matrix / safe_norms

        return np.asarray(normalized_matrix, dtype=np.float32)

    def fit_transform(self, raw_documents: Sequence[Chunk | str]) -> MatrixF32:
        """Fit vectorizer and transform documents in a single call."""
        return self.fit(raw_documents).transform(raw_documents)
