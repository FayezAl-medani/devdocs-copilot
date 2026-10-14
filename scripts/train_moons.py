"""Train Value-based MLP and Tensor-based MLP on Two Moons with decision boundary plots."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from devdocs.autograd.nn import MLP
from devdocs.autograd.tensor import Tensor, softmax_cross_entropy
from devdocs.autograd.value import no_grad

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def make_moons(
    n_samples: int = 200, noise: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two-interleaving-moons 2D classification dataset in pure NumPy."""
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2

    # Top moon
    t1 = rng.uniform(0, np.pi, n_half)
    x1 = np.cos(t1) + rng.normal(0, noise, n_half)
    y1 = np.sin(t1) + rng.normal(0, noise, n_half)

    # Bottom moon
    t2 = rng.uniform(0, np.pi, n_half)
    x2 = 1.0 - np.cos(t2) + rng.normal(0, noise, n_half)
    y2 = 1.0 - np.sin(t2) - 0.5 + rng.normal(0, noise, n_half)

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])]).astype(np.float32)
    y = np.hstack([np.zeros(n_half, dtype=np.int64), np.ones(n_half, dtype=np.int64)])

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def train_value_mlp() -> tuple[MLP, float]:
    """Train scalar Value-based MLP on Two Moons using Hinge Loss."""
    X, y_int = make_moons(n_samples=200, noise=0.1, seed=42)
    y = np.where(y_int == 0, -1.0, 1.0).astype(np.float32)

    model = MLP(2, [16, 16, 1], activation="tanh", seed=42)
    n_steps = 100
    loss_history: list[float] = []

    print("\n--- Training Scalar Value MLP (2 -> 16 -> 16 -> 1) ---", flush=True)
    start_time = time.perf_counter()

    for step in range(n_steps):
        # Forward pass
        scores = [model([float(x[0]), float(x[1])]) for x in X]
        # Hinge loss: max(0, 1 - y * score)
        losses = [(1.0 + -yi * si).relu() for yi, si in zip(y, scores, strict=True)]
        data_loss = sum(losses) * (1.0 / len(losses))
        reg_loss = sum(p * p for p in model.parameters()) * 1e-4
        total_loss = data_loss + reg_loss
        loss_history.append(float(total_loss.data))

        # Backward pass
        model.zero_grad()
        total_loss.backward()

        # Parameter update with linear learning rate decay
        lr = 0.5 * (1.0 - 0.7 * step / n_steps)
        for p in model.parameters():
            p.data -= lr * p.grad

        if (step + 1) % 20 == 0 or step == 0:
            print(f"Step {step + 1:3d}/{n_steps} | Loss: {total_loss.data:.4f}", flush=True)

    train_duration = time.perf_counter() - start_time
    # Compute final accuracy
    with no_grad():
        final_scores = [model([float(x[0]), float(x[1])]) for x in X]
        acc = sum((si.data > 0) == (yi > 0) for si, yi in zip(final_scores, y, strict=True)) / len(
            y
        )
    print(
        f"Value MLP trained in {train_duration:.2f}s | Final Loss: {loss_history[-1]:.4f} | "
        f"Accuracy: {acc * 100:.1f}%",
        flush=True,
    )

    # Plot decision boundary
    plot_decision_boundary_value(model, X, y_int, DOCS_DIR / "decision_boundary_value.png")
    return model, acc


def plot_decision_boundary_value(
    model: MLP, X: np.ndarray, y: np.ndarray, output_path: Path
) -> None:
    """Plot 2D decision boundary for Value MLP and save image."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))

    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    with no_grad():
        preds: list[float] = [float(model([float(pt[0]), float(pt[1])]).data) for pt in grid_pts]
    Z = np.array(preds).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors="k")
    plt.title("Value MLP Decision Boundary (Accuracy >= 95%)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}", flush=True)


def train_tensor_mlp() -> tuple[list[Tensor], float]:
    """Train Tensor-based MLP on Two Moons with Softmax Cross-Entropy."""
    X, y = make_moons(n_samples=200, noise=0.1, seed=42)
    rng = np.random.default_rng(42)

    W1 = Tensor(rng.standard_normal((2, 32), dtype=np.float32) * np.sqrt(2.0 / 2))
    b1 = Tensor(np.zeros(32, dtype=np.float32))
    W2 = Tensor(rng.standard_normal((32, 32), dtype=np.float32) * np.sqrt(2.0 / 32))
    b2 = Tensor(np.zeros(32, dtype=np.float32))
    W3 = Tensor(rng.standard_normal((32, 2), dtype=np.float32) * np.sqrt(2.0 / 32))
    b3 = Tensor(np.zeros(2, dtype=np.float32))

    params = [W1, b1, W2, b2, W3, b3]
    X_t = Tensor(X)
    n_steps = 100
    loss_history: list[float] = []

    print("\n--- Training Tensor MLP (2 -> 32 -> 32 -> 2) ---")
    start_time = time.perf_counter()

    for step in range(n_steps):
        # Forward pass
        h1 = (X_t @ W1 + b1).tanh()
        h2 = (h1 @ W2 + b2).tanh()
        logits = h2 @ W3 + b3
        loss = softmax_cross_entropy(logits, y)
        loss_history.append(float(loss.data))

        # Backward pass
        for p in params:
            p.grad.fill(0.0)
        loss.backward()

        # Parameter update with decay
        lr = 0.5 * (1.0 - 0.7 * step / n_steps)
        for p in params:
            p.data -= lr * p.grad

    train_duration = time.perf_counter() - start_time
    # Final accuracy
    h1 = (X_t @ W1 + b1).tanh()
    h2 = (h1 @ W2 + b2).tanh()
    final_logits = h2 @ W3 + b3
    preds = np.argmax(final_logits.data, axis=1)
    acc = float(np.mean(preds == y))

    print(
        f"Tensor MLP trained in {train_duration:.3f}s | Final Loss: {loss_history[-1]:.4f} | "
        f"Accuracy: {acc * 100:.1f}%"
    )

    plot_decision_boundary_tensor(params, X, y, DOCS_DIR / "decision_boundary_tensor.png")
    return params, acc


def plot_decision_boundary_tensor(
    params: list[Tensor], X: np.ndarray, y: np.ndarray, output_path: Path
) -> None:
    """Plot 2D decision boundary for Tensor MLP and save image."""
    W1, b1, W2, b2, W3, b3 = params
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    grid_pts = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_t = Tensor(grid_pts)
    h1 = (grid_t @ W1 + b1).tanh()
    h2 = (h1 @ W2 + b2).tanh()
    logits = h2 @ W3 + b3
    Z = np.argmax(logits.data, axis=1).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors="k")
    plt.title("Tensor MLP Decision Boundary (Accuracy >= 95%)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Run training for both Value and Tensor MLPs and produce artifacts."""
    _, acc_val = train_value_mlp()
    _, acc_ten = train_tensor_mlp()

    assert acc_val >= 0.95, f"Value MLP accuracy {acc_val:.2%} < 95%"
    assert acc_ten >= 0.95, f"Tensor MLP accuracy {acc_ten:.2%} < 95%"
    print("\nBoth models achieved >= 95% accuracy! Artifacts generated successfully in docs/.")


if __name__ == "__main__":
    main()
