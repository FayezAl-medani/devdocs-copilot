"""devdocs package initialization."""

from devdocs.corpus import Chunk, load_markdown
from devdocs.linalg import (
    Matrix,
    Vector,
    cosine_sim,
    cosine_sim_matrix,
    matmul_naive,
    matmul_vec,
)
from devdocs.lsa import LsaModel
from devdocs.metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank
from devdocs.search import Hit, SemanticIndex
from devdocs.text import tokenize
from devdocs.tfidf import TfidfVectorizer

__all__ = [
    "Chunk",
    "Hit",
    "LsaModel",
    "Matrix",
    "SemanticIndex",
    "TfidfVectorizer",
    "Vector",
    "cosine_sim",
    "cosine_sim_matrix",
    "load_markdown",
    "matmul_naive",
    "matmul_vec",
    "mean_reciprocal_rank",
    "recall_at_k",
    "reciprocal_rank",
    "tokenize",
]


def main() -> None:
    """CLI entry point for devdocs."""
    print("devdocs package initialized successfully.")
