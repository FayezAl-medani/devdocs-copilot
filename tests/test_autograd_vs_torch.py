# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false
"""Tests validating Value and Tensor autograd engines against PyTorch and numerical gradients."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from devdocs.autograd.nn import MLP, Layer, Neuron
from devdocs.autograd.tensor import Tensor, softmax_cross_entropy, unbroadcast
from devdocs.autograd.value import Value, no_grad


def check_relative_error(anal_grad: np.ndarray, num_grad: np.ndarray, tol: float = 1e-4) -> None:
    """Validate that analytical and numerical gradients agree within relative error tolerance."""
    denom = np.maximum(np.maximum(np.abs(anal_grad), np.abs(num_grad)), 1e-7)
    rel_error = np.abs(anal_grad - num_grad) / denom
    assert np.all(rel_error < tol), (
        f"Gradient check failed! Max rel error: {np.max(rel_error)}\n"
        f"Analytical:\n{anal_grad}\nNumerical:\n{num_grad}"
    )


def compute_numerical_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Compute central difference numerical gradient (f(x+h) - f(x-h)) / (2h)."""
    grad = np.zeros_like(x, dtype=np.float64)
    x_work = x.copy().astype(np.float64)

    it = np.nditer(x_work, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x_work[idx]

        x_work[idx] = orig + eps
        pos = f(x_work)

        x_work[idx] = orig - eps
        neg = f(x_work)

        x_work[idx] = orig
        grad[idx] = (pos - neg) / (2.0 * eps)
        it.iternext()

    return grad.astype(np.float32)


# ==============================================================================
# 1. Scalar Value vs PyTorch
# ==============================================================================


class TestValueAutograd:
    """Verify scalar Value forward and backward operations against PyTorch."""

    def test_basic_arithmetic(self) -> None:
        # devdocs Value
        a = Value(2.5)
        b = Value(-1.5)
        c = (a * b + a) / b - (a**2.0)
        c.backward()

        # PyTorch
        ta = torch.tensor(2.5, requires_grad=True, dtype=torch.float64)
        tb = torch.tensor(-1.5, requires_grad=True, dtype=torch.float64)
        tc = (ta * tb + ta) / tb - (ta**2.0)
        tc.backward()

        assert np.isclose(c.data, tc.item(), atol=1e-6)
        assert np.isclose(a.grad, ta.grad.item(), atol=1e-6)
        assert np.isclose(b.grad, tb.grad.item(), atol=1e-6)

    def test_activations_and_math(self) -> None:
        # devdocs Value
        x = Value(0.7)
        y = x.tanh().exp() + (x.relu() * 2.0) - x.log()
        y.backward()

        # PyTorch
        tx = torch.tensor(0.7, requires_grad=True, dtype=torch.float64)
        ty = tx.tanh().exp() + (torch.relu(tx) * 2.0) - tx.log()
        ty.backward()

        assert np.isclose(y.data, ty.item(), atol=1e-6)
        assert np.isclose(x.grad, tx.grad.item(), atol=1e-6)

    def test_node_reuse_accumulation(self) -> None:
        """Verify the += accumulation rule when a node is used multiple times (diamond graph)."""
        x = Value(3.0)
        y = x * x + x  # dy/dx = 2*x + 1 = 7.0
        y.backward()

        tx = torch.tensor(3.0, requires_grad=True, dtype=torch.float64)
        ty = tx * tx + tx
        ty.backward()

        assert np.isclose(y.data, ty.item())
        assert np.isclose(x.grad, 7.0)
        assert np.isclose(x.grad, tx.grad.item())

    def test_no_grad_context(self) -> None:
        x = Value(2.0)
        with no_grad():
            y = x * 3.0 + 4.0
        assert len(y.prev) == 0
        y.backward()
        assert x.grad == 0.0

    def test_deep_graph_no_recursion_error(self) -> None:
        """Ensure iterative topo sort handles deep graphs (> 1200 nodes) without RecursionError."""
        x = Value(1.001)
        curr = x
        for _ in range(1200):
            curr = curr + curr * 0.001
        # Should execute cleanly without hitting Python recursion limit (1000)
        curr.backward()
        assert curr.grad == 1.0
        assert x.grad > 0.0


# ==============================================================================
# 2. Tensor Autograd vs PyTorch
# ==============================================================================


class TestTensorAutograd:
    """Verify Tensor operations and broadcasting against PyTorch."""

    def test_matmul_and_linear(self) -> None:
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 3)).astype(np.float32)
        w_np = rng.standard_normal((3, 5)).astype(np.float32)
        b_np = rng.standard_normal((5,)).astype(np.float32)

        # Tensor
        x = Tensor(x_np)
        w = Tensor(w_np)
        b = Tensor(b_np)
        out = (x @ w + b).relu()
        loss = out.sum()
        loss.backward()

        # PyTorch
        tx = torch.tensor(x_np, requires_grad=True)
        tw = torch.tensor(w_np, requires_grad=True)
        tb = torch.tensor(b_np, requires_grad=True)
        tout = torch.relu(tx @ tw + tb)
        tloss = tout.sum()
        tloss.backward()

        assert np.allclose(out.data, tout.detach().numpy(), atol=1e-5)
        assert np.allclose(loss.data, tloss.detach().numpy(), atol=1e-5)
        assert np.allclose(x.grad, tx.grad.numpy(), atol=1e-5)
        assert np.allclose(w.grad, tw.grad.numpy(), atol=1e-5)
        assert np.allclose(b.grad, tb.grad.numpy(), atol=1e-5)

    def test_tensor_elementwise_and_reductions(self) -> None:
        rng = np.random.default_rng(123)
        a_np = np.abs(rng.standard_normal((3, 4))).astype(np.float32) + 0.5
        b_np = np.abs(rng.standard_normal((3, 4))).astype(np.float32) + 0.5

        # Tensor
        a = Tensor(a_np)
        b = Tensor(b_np)
        c = (a * b - a / b).tanh() + a.exp() - b.log()
        loss = c.mean()
        loss.backward()

        # PyTorch
        ta = torch.tensor(a_np, requires_grad=True)
        tb = torch.tensor(b_np, requires_grad=True)
        tc = torch.tanh(ta * tb - ta / tb) + torch.exp(ta) - torch.log(tb)
        tloss = tc.mean()
        tloss.backward()

        assert np.allclose(c.data, tc.detach().numpy(), atol=1e-5)
        assert np.allclose(loss.data, tloss.detach().numpy(), atol=1e-5)
        assert np.allclose(a.grad, ta.grad.numpy(), atol=1e-5)
        assert np.allclose(b.grad, tb.grad.numpy(), atol=1e-5)

    def test_broadcasting_unbroadcast(self) -> None:
        """Test gradient shape reduction when tensors broadcast."""
        # Case 1: (3, 4) + (4,) -> target (4,)
        grad = np.ones((3, 4), dtype=np.float32)
        reduced = unbroadcast(grad, (4,))
        assert reduced.shape == (4,)
        assert np.allclose(reduced, np.full((4,), 3.0, dtype=np.float32))

        # Case 2: (2, 3, 4) + (1, 4) -> target (1, 4)
        grad2 = np.ones((2, 3, 4), dtype=np.float32)
        reduced2 = unbroadcast(grad2, (1, 4))
        assert reduced2.shape == (1, 4)
        assert np.allclose(reduced2, np.full((1, 4), 6.0, dtype=np.float32))

        # Tensor operation test with broadcasting
        x = Tensor(np.ones((2, 3), dtype=np.float32))
        b = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        y = (x + b) * b
        y.sum().backward()

        tx = torch.tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
        tb = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        ty = (tx + tb) * tb
        ty.sum().backward()

        assert np.allclose(x.grad, tx.grad.numpy(), atol=1e-5)
        assert np.allclose(b.grad, tb.grad.numpy(), atol=1e-5)

    def test_tensor_deep_chain_non_recursive(self) -> None:
        """Ensure Tensor backward handles deep graphs (> 1200 nodes) iteratively."""
        x = Tensor(np.array([1.5], dtype=np.float32))
        curr = x
        for _ in range(1200):
            curr = curr + curr * 0.001
        loss = curr.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (1,)
        assert x.grad[0] > 0.0


# ==============================================================================
# 3. Softmax Cross-Entropy Numerical Stability & Equivalence
# ==============================================================================


class TestSoftmaxCrossEntropy:
    """Verify softmax cross-entropy against PyTorch and test stability on extreme values."""

    def test_matches_pytorch_cross_entropy(self) -> None:
        rng = np.random.default_rng(999)
        logits_np = rng.standard_normal((10, 4)).astype(np.float32)
        y_np = rng.integers(0, 4, size=(10,)).astype(np.int64)

        # Tensor
        logits = Tensor(logits_np)
        loss = softmax_cross_entropy(logits, y_np)
        loss.backward()

        # PyTorch
        tlogits = torch.tensor(logits_np, requires_grad=True)
        ty = torch.tensor(y_np, dtype=torch.long)
        tloss = torch.nn.functional.cross_entropy(tlogits, ty)
        tloss.backward()

        assert np.allclose(loss.data, tloss.detach().numpy(), atol=1e-5)
        assert np.allclose(logits.grad, tlogits.grad.numpy(), atol=1e-5)

    def test_numerical_stability_extreme_logits(self) -> None:
        """Huge logits (e.g. 1000.0) cause naive exp() to overflow into inf and NaN."""
        extreme_logits = np.array(
            [[1000.0, 999.0, -500.0], [-1000.0, 0.0, 1000.0]], dtype=np.float32
        )
        y = np.array([0, 2], dtype=np.int64)

        t = Tensor(extreme_logits)
        loss = softmax_cross_entropy(t, y)
        loss.backward()

        assert not np.isnan(loss.data).any()
        assert not np.isinf(loss.data).any()
        assert t.grad is not None
        assert not np.isnan(t.grad).any()
        assert not np.isinf(t.grad).any()


# ==============================================================================
# 4. Identical Weight & Batch Training Step
# ==============================================================================


def test_identical_mlp_step_vs_pytorch() -> None:
    """Run an identical training step with exact same initial weights and batch.

    Verifies forward loss and all parameter gradients match between DevDocs Tensor and PyTorch.
    """
    rng = np.random.default_rng(2026)
    batch_size = 16
    in_dim, h1, h2, out_dim = 2, 8, 8, 2

    # Inputs and targets
    X_np = rng.standard_normal((batch_size, in_dim)).astype(np.float32)
    y_np = rng.integers(0, out_dim, size=(batch_size,)).astype(np.int64)

    # Weights
    w1_np = rng.standard_normal((in_dim, h1)).astype(np.float32) * 0.1
    b1_np = np.zeros(h1, dtype=np.float32)
    w2_np = rng.standard_normal((h1, h2)).astype(np.float32) * 0.1
    b2_np = np.zeros(h2, dtype=np.float32)
    w3_np = rng.standard_normal((h2, out_dim)).astype(np.float32) * 0.1
    b3_np = np.zeros(out_dim, dtype=np.float32)

    # DevDocs Forward + Backward
    X_t = Tensor(X_np)
    W1 = Tensor(w1_np)
    B1 = Tensor(b1_np)
    W2 = Tensor(w2_np)
    B2 = Tensor(b2_np)
    W3 = Tensor(w3_np)
    B3 = Tensor(b3_np)

    h1_t = (X_t @ W1 + B1).tanh()
    h2_t = (h1_t @ W2 + B2).tanh()
    logits_t = h2_t @ W3 + B3
    loss_t = softmax_cross_entropy(logits_t, y_np)
    loss_t.backward()

    # PyTorch Forward + Backward
    tX = torch.tensor(X_np)
    tW1 = torch.tensor(w1_np, requires_grad=True)
    tB1 = torch.tensor(b1_np, requires_grad=True)
    tW2 = torch.tensor(w2_np, requires_grad=True)
    tB2 = torch.tensor(b2_np, requires_grad=True)
    tW3 = torch.tensor(w3_np, requires_grad=True)
    tB3 = torch.tensor(b3_np, requires_grad=True)

    th1 = torch.tanh(tX @ tW1 + tB1)
    th2 = torch.tanh(th1 @ tW2 + tB2)
    tlogits = th2 @ tW3 + tB3
    tloss = torch.nn.functional.cross_entropy(tlogits, torch.tensor(y_np, dtype=torch.long))
    tloss.backward()

    # Assert forward match
    assert np.allclose(loss_t.data, tloss.detach().numpy(), atol=1e-5)

    # Assert all gradients match
    assert np.allclose(W1.grad, tW1.grad.numpy(), atol=1e-5)
    assert np.allclose(B1.grad, tB1.grad.numpy(), atol=1e-5)
    assert np.allclose(W2.grad, tW2.grad.numpy(), atol=1e-5)
    assert np.allclose(B2.grad, tB2.grad.numpy(), atol=1e-5)
    assert np.allclose(W3.grad, tW3.grad.numpy(), atol=1e-5)
    assert np.allclose(B3.grad, tB3.grad.numpy(), atol=1e-5)


# ==============================================================================
# 5. Independent Numerical Gradient Checks (No PyTorch)
# ==============================================================================


class TestNumericalGradCheck:
    """Verify analytical gradients directly against finite differences without PyTorch."""

    def test_tensor_linear_layer_numerical(self) -> None:
        rng = np.random.default_rng(777)
        x_init = rng.standard_normal((3, 4)).astype(np.float32)
        w_init = rng.standard_normal((4, 2)).astype(np.float32)
        b_init = rng.standard_normal((2,)).astype(np.float32)

        def eval_loss(w_arr: np.ndarray) -> float:
            x_t = Tensor(x_init)
            w_t = Tensor(w_arr)
            b_t = Tensor(b_init)
            out = x_t @ w_t + b_t
            loss = (out * out).sum()
            return float(loss.data)

        # Analytical grad
        x_t = Tensor(x_init)
        w_t = Tensor(w_init)
        b_t = Tensor(b_init)
        out = x_t @ w_t + b_t
        loss = (out * out).sum()
        loss.backward()
        analytical_grad = w_t.grad

        # Numerical grad
        numerical_grad = compute_numerical_gradient(eval_loss, w_init, eps=1e-3)

        assert analytical_grad is not None
        check_relative_error(analytical_grad, numerical_grad, tol=1e-2)

    def test_tensor_tanh_numerical(self) -> None:
        rng = np.random.default_rng(888)
        x_init = rng.standard_normal((2, 3)).astype(np.float32)

        def eval_loss(x_arr: np.ndarray) -> float:
            t = Tensor(x_arr)
            return float(t.tanh().sum().data)

        t = Tensor(x_init)
        t.tanh().sum().backward()
        analytical_grad = t.grad

        numerical_grad = compute_numerical_gradient(eval_loss, x_init, eps=1e-3)

        assert analytical_grad is not None
        check_relative_error(analytical_grad, numerical_grad, tol=1e-2)

    def test_softmax_cross_entropy_numerical(self) -> None:
        rng = np.random.default_rng(555)
        logits_init = rng.standard_normal((4, 3)).astype(np.float32)
        y = np.array([0, 1, 2, 0], dtype=np.int64)

        def eval_loss(l_arr: np.ndarray) -> float:
            t = Tensor(l_arr)
            return float(softmax_cross_entropy(t, y).data)

        t = Tensor(logits_init)
        loss = softmax_cross_entropy(t, y)
        loss.backward()
        analytical_grad = t.grad

        numerical_grad = compute_numerical_gradient(eval_loss, logits_init, eps=1e-3)

        assert analytical_grad is not None
        check_relative_error(analytical_grad, numerical_grad, tol=5e-2)


# ==============================================================================
# 6. Neural Net Modules (Neuron, Layer, MLP)
# ==============================================================================


class TestNeuralNetModules:
    """Verify initialization, forward evaluation, and error checks for neural net components."""

    def test_neuron_activations(self) -> None:
        n_tanh = Neuron(3, nonlin=True, activation="tanh", seed=42)
        n_relu = Neuron(3, nonlin=True, activation="relu", seed=42)
        n_linear = Neuron(3, nonlin=False, activation="linear", seed=42)

        x = [Value(1.0), Value(-0.5), Value(0.2)]
        out_tanh = n_tanh(x)
        out_relu = n_relu(x)
        out_linear = n_linear(x)

        assert isinstance(out_tanh, Value)
        assert isinstance(out_relu, Value)
        assert isinstance(out_linear, Value)
        assert -1.0 <= out_tanh.data <= 1.0
        assert out_relu.data >= 0.0

    def test_neuron_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="nin must be positive"):
            Neuron(0)

        n = Neuron(2, activation="unknown")
        with pytest.raises(ValueError, match="Unsupported activation"):
            n([1.0, 2.0])

        with pytest.raises(ValueError, match="Input dimension mismatch"):
            n([1.0])

    def test_layer_forward_and_params(self) -> None:
        layer_multi = Layer(2, 4, seed=1)
        layer_single = Layer(2, 1, seed=2)

        out_multi = layer_multi([0.5, -0.5])
        assert isinstance(out_multi, list)
        assert len(out_multi) == 4

        out_single = layer_single([0.5, -0.5])
        assert isinstance(out_single, Value)

        params = layer_multi.parameters()
        assert len(params) == 4 * (2 + 1)  # 4 neurons * (2 weights + 1 bias)

        with pytest.raises(ValueError, match="nout must be positive"):
            Layer(2, 0)

    def test_mlp_forward_and_backward(self) -> None:
        mlp = MLP(2, [4, 4, 1], activation="tanh", seed=42)
        x = [Value(0.3), Value(-0.4)]
        out = mlp(x)
        assert isinstance(out, Value)

        out.backward()
        for p in mlp.parameters():
            assert p.grad != 0.0

        mlp.zero_grad()
        for p in mlp.parameters():
            assert p.grad == 0.0

    def test_mlp_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="nouts must specify at least one layer size"):
            MLP(2, [])
