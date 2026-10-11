"""Benchmark comparing naive 3-loop matmul with NumPy vectorized matmul."""

from __future__ import annotations

import sys
import time

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from devdocs.linalg import Matrix, matmul_naive, matmul_vec


def benchmark_size(size: int, naive_runs: int = 1, vec_runs: int = 3) -> dict[str, float]:
    """Benchmark matmul_naive and matmul_vec for square matrices of given size."""
    rng = np.random.default_rng(seed=42)
    a: Matrix = rng.standard_normal((size, size)).astype(np.float64)
    b: Matrix = rng.standard_normal((size, size)).astype(np.float64)

    # Measure naive
    naive_times: list[float] = []
    for _ in range(naive_runs):
        start = time.perf_counter()
        _ = matmul_naive(a, b)
        naive_times.append(time.perf_counter() - start)
    best_naive = min(naive_times)

    # Measure vectorized
    vec_times: list[float] = []
    for _ in range(vec_runs):
        start = time.perf_counter()
        _ = matmul_vec(a, b)
        vec_times.append(time.perf_counter() - start)
    best_vec = min(vec_times)

    return {
        "naive_time": best_naive,
        "vec_time": best_vec,
        "speedup": best_naive / best_vec if best_vec > 0 else 0.0,
    }


def main() -> None:
    """Run benchmarks and print markdown table."""
    sizes = [200, 500]

    print("=" * 70)
    print("Running Matrix Multiplication Benchmark (Naive vs Vectorized)")
    print("=" * 70)

    results: list[dict[str, str]] = []

    for sz in sizes:
        print(f"Benchmarking size {sz}x{sz}...")
        # For 500x500 naive has 1.25e8 iterations, 1 run is sufficient.
        metrics = benchmark_size(sz, naive_runs=1, vec_runs=3)

        naive_sec = metrics["naive_time"]
        vec_sec = metrics["vec_time"]
        speedup = metrics["speedup"]

        results.append(
            {
                "func": "matmul_naive",
                "size": f"{sz}×{sz}",
                "time": f"{naive_sec:.4f} s ({naive_sec * 1000:.1f} ms)",
                "speedup": "1.0×",
            }
        )
        results.append(
            {
                "func": "matmul_vec",
                "size": f"{sz}×{sz}",
                "time": f"{vec_sec:.6f} s ({vec_sec * 1000:.3f} ms)",
                "speedup": f"{speedup:.1f}×",
            }
        )

    print("\nBenchmark Results Summary Table:")
    print("| الدالة | الحجم | الزمن | التسريع |")
    print("|---|---|---|---|")
    for row in results:
        print(f"| {row['func']} | {row['size']} | {row['time']} | {row['speedup']} |")


if __name__ == "__main__":
    main()
