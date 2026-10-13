"""Semantic search engine using TF-IDF and Latent Semantic Analysis (LSA)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from devdocs.corpus import Chunk
from devdocs.linalg import cosine_sim_matrix
from devdocs.lsa import LsaModel
from devdocs.text import tokenize
from devdocs.tfidf import TfidfVectorizer

MatrixF32 = NDArray[np.float32]


@dataclass(frozen=True)
class Hit:
    """Represents a scored search result."""

    chunk: Chunk
    score: float


class SemanticIndex:
    """Search index supporting TF-IDF and LSA dense retrieval.

    Parameters
    ----------
    method : {"tfidf", "lsa"}
        Indexing and retrieval method.
    k : int
        Number of latent semantic dimensions if method is "lsa".
    min_df : int
        Minimum document frequency for TF-IDF vocabulary.
    max_features : int or None
        Maximum vocabulary features.
    """

    def __init__(
        self,
        method: Literal["tfidf", "lsa"] = "tfidf",
        k: int = 128,
        min_df: int = 1,
        max_features: int | None = 8000,
    ) -> None:
        if method not in ("tfidf", "lsa"):
            raise ValueError(f"Unknown indexing method '{method}'. Must be 'tfidf' or 'lsa'.")
        self.method: Literal["tfidf", "lsa"] = method
        self.k: int = k
        self.min_df: int = min_df
        self.max_features: int | None = max_features

        self.chunks: list[Chunk] = []
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features)
        self.lsa_model: LsaModel | None = None
        self.doc_embeddings_: MatrixF32 = np.empty((0, 0), dtype=np.float32)
        self.is_built: bool = False

    def build(
        self,
        chunks: list[Chunk],
        method: Literal["tfidf", "lsa"] | None = None,
        k: int | None = None,
    ) -> SemanticIndex:
        """Build search index over a collection of document chunks.

        Pre-computes and caches normalized representations so search execution
        requires only a single query projection and batch dot product.

        Parameters
        ----------
        chunks : list[Chunk]
            Document corpus chunks to index.
        method : {"tfidf", "lsa"} or None
            Optional method override.
        k : int or None
            Optional latent dimension override.

        Returns
        -------
        SemanticIndex
            Built search index instance.
        """
        if not chunks:
            raise ValueError("Cannot build SemanticIndex on an empty collection of chunks.")

        if method is not None:
            if method not in ("tfidf", "lsa"):
                raise ValueError(f"Unknown method '{method}'.")
            self.method = method

        if k is not None:
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            self.k = k

        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(min_df=self.min_df, max_features=self.max_features)

        # 1. Compute L2-normalized TF-IDF matrix for all chunks
        tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

        # 2. Compute embeddings based on chosen method
        if self.method == "tfidf":
            self.doc_embeddings_ = tfidf_matrix
            self.lsa_model = None
        else:
            self.lsa_model = LsaModel(k=self.k)
            self.lsa_model.fit(tfidf_matrix)
            self.doc_embeddings_ = self.lsa_model.project(tfidf_matrix)

        self.is_built = True
        return self

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        """Search the index for the top-k most relevant chunks.

        Engineering & Validation Behaviors:
        1. Empty query (or whitespace) raises ValueError.
        2. Query with only out-of-vocabulary (OOV) terms returns empty list []
           (avoids surfacing ungrounded arbitrary top-k results with zero scores).
        3. Efficient Top-K: uses np.argpartition in O(m) rather than O(m log m) argsort.
        4. Gracefully bounds top_k <= len(chunks).

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int
            Number of top results to return.

        Returns
        -------
        list[Hit]
            Ranked list of Hit instances sorted by descending similarity score.
        """
        if not self.is_built:
            raise ValueError("SemanticIndex has not been built yet. Call build() first.")

        query_stripped = query.strip()
        if not query_stripped:
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            return []

        # Tokenize query and check for in-vocabulary terms
        query_tokens = tokenize(query_stripped)
        in_vocab_tokens = [tok for tok in query_tokens if tok in self.vectorizer.vocabulary_]

        # Decision: Out-of-vocabulary queries carry zero semantic signal -> return empty list
        if not in_vocab_tokens:
            return []

        # Vectorize query
        q_tfidf = self.vectorizer.transform([query_stripped])

        if self.method == "tfidf":
            q_emb = q_tfidf
        else:
            assert self.lsa_model is not None
            q_emb = self.lsa_model.project(q_tfidf)

        # Compute similarity against all pre-cached document embeddings
        # shape: (1, n_chunks) -> flatten to (n_chunks,)
        sim_scores = cosine_sim_matrix(q_emb, self.doc_embeddings_)[0]

        n_docs = len(self.chunks)
        effective_k = min(top_k, n_docs)

        # O(m) top-k selection via argpartition
        if effective_k >= n_docs:
            sorted_indices = np.argsort(-sim_scores)
        else:
            partition_indices = np.argpartition(sim_scores, -effective_k)[-effective_k:]
            sorted_within_partition = np.argsort(-sim_scores[partition_indices])
            sorted_indices = partition_indices[sorted_within_partition]

        hits: list[Hit] = []
        for idx in sorted_indices:
            score = float(sim_scores[idx])
            # Only return hits with non-zero similarity
            if score > 0.0:
                hits.append(Hit(chunk=self.chunks[int(idx)], score=score))

        return hits

    def save(self, path: Path | str) -> None:
        """Persist index state to disk using compressed numpy archive and metadata JSON.

        Parameters
        ----------
        path : Path or str
            Destination filepath prefix (creates .npz and .json files).
        """
        if not self.is_built:
            raise ValueError("Cannot save an unbuilt index.")

        base_path = Path(path).with_suffix("")
        npz_file = base_path.with_suffix(".npz")
        json_file = base_path.with_suffix(".json")

        # Save matrices in compressed npz
        if self.lsa_model is not None:
            np.savez_compressed(
                npz_file,
                doc_embeddings=self.doc_embeddings_,
                idf=self.vectorizer.idf_,
                lsa_components=self.lsa_model.components_,
                lsa_singular_values=self.lsa_model.singular_values_,
            )
        else:
            np.savez_compressed(
                npz_file,
                doc_embeddings=self.doc_embeddings_,
                idf=self.vectorizer.idf_,
            )

        # Save metadata and chunks in JSON
        metadata = {
            "method": self.method,
            "k": self.k,
            "min_df": self.min_df,
            "max_features": self.max_features,
            "vocabulary": self.vectorizer.vocabulary_,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        json_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> SemanticIndex:
        """Load persisted index from disk.

        Parameters
        ----------
        path : Path or str
            Filepath prefix pointing to saved index (.npz / .json).

        Returns
        -------
        SemanticIndex
            Reconstructed SemanticIndex instance with exact identical parameters and state.
        """
        base_path = Path(path).with_suffix("")
        npz_file = base_path.with_suffix(".npz")
        json_file = base_path.with_suffix(".json")

        if not npz_file.exists() or not json_file.exists():
            raise FileNotFoundError(f"Missing index files for base path: {base_path}")

        meta = json.loads(json_file.read_text(encoding="utf-8"))

        index = cls(
            method=meta["method"],
            k=meta["k"],
            min_df=meta["min_df"],
            max_features=meta["max_features"],
        )

        index.chunks = [Chunk(**c) for c in meta["chunks"]]
        index.vectorizer.vocabulary_ = meta["vocabulary"]
        index.vectorizer.is_fitted = True

        npz_data = np.load(npz_file)
        index.vectorizer.idf_ = np.asarray(npz_data["idf"], dtype=np.float32)
        index.doc_embeddings_ = np.asarray(npz_data["doc_embeddings"], dtype=np.float32)

        if meta["method"] == "lsa":
            index.lsa_model = LsaModel(k=meta["k"])
            index.lsa_model.components_ = np.asarray(npz_data["lsa_components"], dtype=np.float32)
            index.lsa_model.singular_values_ = np.asarray(
                npz_data["lsa_singular_values"], dtype=np.float32
            )
            index.lsa_model.is_fitted = True

        index.is_built = True
        return index
