import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devdocs.autograd.nn import MLP as ValueMLP
from devdocs.autograd.tensor import Tensor, softmax_cross_entropy
from devdocs.autograd.value import Value
from scripts.train_moons import make_moons


def benchmark_value(X: np.ndarray, y: np.ndarray, iterations: int = 5) -> float:
    """Benchmark 1 training step of scalar Value MLP on 200 samples."""
    # Convert labels from {0, 1} to {-1, 1} for hinge loss
    y_hinge = np.where(y == 1, 1.0, -1.0)
    model = ValueMLP(2, [16, 16, 1])

    def one_step() -> None:
        model.zero_grad()
        total_loss = Value(0.0)
        for xi, yi in zip(X, y_hinge, strict=True):
            inputs = [Value(xi[0]), Value(xi[1])]
            pred = model(inputs)
            assert isinstance(pred, Value)
            loss_i = (Value(1.0) - Value(yi) * pred).relu()
            total_loss = total_loss + loss_i
        mean_loss = total_loss * (1.0 / len(X))
        mean_loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad

    # Warmup
    one_step()

    start = time.perf_counter()
    for _ in range(iterations):
        one_step()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0  # in ms


def benchmark_tensor(X: np.ndarray, y: np.ndarray, iterations: int = 50) -> float:
    """Benchmark 1 training step of pure NumPy Tensor MLP on 200 samples."""
    rng = np.random.default_rng(42)
    # Architecture: 2 -> 16 -> 16 -> 2
    w1 = Tensor(rng.standard_normal((2, 16)).astype(np.float32) * np.sqrt(2.0 / 2.0))
    b1 = Tensor(np.zeros(16, dtype=np.float32))
    w2 = Tensor(rng.standard_normal((16, 16)).astype(np.float32) * np.sqrt(2.0 / 16.0))
    b2 = Tensor(np.zeros(16, dtype=np.float32))
    w3 = Tensor(rng.standard_normal((16, 2)).astype(np.float32) * np.sqrt(2.0 / 16.0))
    b3 = Tensor(np.zeros(2, dtype=np.float32))
    params = [w1, b1, w2, b2, w3, b3]

    X_t = Tensor(X.astype(np.float32))

    def one_step() -> None:
        for p in params:
            p.zero_grad()

        h1 = (X_t @ w1 + b1).tanh()
        h2 = (h1 @ w2 + b2).tanh()
        logits = h2 @ w3 + b3
        loss = softmax_cross_entropy(logits, y)
        loss.backward()

        for p in params:
            if p.grad is not None:
                p.data -= 0.05 * p.grad

    # Warmup
    for _ in range(3):
        one_step()

    start = time.perf_counter()
    for _ in range(iterations):
        one_step()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0  # in ms


def benchmark_pytorch(X: np.ndarray, y: np.ndarray, iterations: int = 50) -> float:
    """Benchmark 1 training step of PyTorch CPU MLP on 200 samples."""
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 2),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()

    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.long)

    def one_step() -> None:
        optimizer.zero_grad()
        out = model(X_torch)
        loss = criterion(out, y_torch)
        loss.backward()
        optimizer.step()

    # Warmup
    for _ in range(3):
        one_step()

    start = time.perf_counter()
    for _ in range(iterations):
        one_step()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0  # in ms


def main() -> None:
    print("=" * 70)
    print("BENCHMARK: 1 Training Step Latency (Batch Size = 200, 2-16-16-2 MLP)")
    print("=" * 70)

    X, y = make_moons(n_samples=200, noise=0.1, seed=42)

    print("Running PyTorch CPU benchmark...")
    time_torch = benchmark_pytorch(X, y, iterations=100)

    print("Running NumPy Tensor benchmark...")
    time_tensor = benchmark_tensor(X, y, iterations=100)

    print("Running Scalar Value benchmark (this may take a few seconds)...")
    time_value = benchmark_value(X, y, iterations=5)

    print("\n" + "=" * 70)
    header = f"{'Implementation':<24} | {'Latency (ms)':<14} | {'vs Value':<12} | {'vs Tensor':<12}"
    print(header)
    print("-" * 70)
    sp_v_t = f"{time_value / time_tensor:.1f}x slower"
    print(f"{'Scalar Value (micrograd)':<24} | {time_value:<14.2f} | {'1.00x':<12} | {sp_v_t:<12}")
    sp_t_v = f"{time_value / time_tensor:.1f}x"
    print(f"{'NumPy Tensor (ours)':<24} | {time_tensor:<14.2f} | {sp_t_v:<12} | {'1.00x':<12}")
    sp_p_v = f"{time_value / time_torch:.1f}x"
    sp_p_t = f"{time_tensor / time_torch:.2f}x"
    print(f"{'PyTorch (CPU)':<24} | {time_torch:<14.2f} | {sp_p_v:<12} | {sp_p_t:<12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
