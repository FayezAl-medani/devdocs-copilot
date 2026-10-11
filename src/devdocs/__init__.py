"""devdocs package initialization."""

from devdocs.linalg import (
    Matrix,
    Vector,
    cosine_sim,
    cosine_sim_matrix,
    matmul_naive,
    matmul_vec,
)

__all__ = [
    "Matrix",
    "Vector",
    "cosine_sim",
    "cosine_sim_matrix",
    "matmul_naive",
    "matmul_vec",
]


def main() -> None:
    """CLI entry point for devdocs."""
    print("devdocs package initialized successfully.")
