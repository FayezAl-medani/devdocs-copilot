"""Tensor-level automatic differentiation engine implemented in pure NumPy."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

ArrayF32 = NDArray[np.float32]


def _unbroadcast(grad: ArrayF32, target_shape: tuple[int, ...]) -> ArrayF32:
    """Sum out gradient dimensions introduced by NumPy broadcasting.

    Parameters
    ----------
    grad : ArrayF32
        Accumulated downstream gradient array.
    target_shape : tuple[int, ...]
        Original shape of the operand before broadcasting.

    Returns
    -------
    ArrayF32
        Gradient reduced and reshaped to match target_shape.
    """
    if grad.shape == target_shape:
        return grad

    if target_shape == () or target_shape == (1,):
        return np.asarray(np.sum(grad).reshape(target_shape), dtype=np.float32)

    # 1. Sum over leading dimensions prepended by broadcasting
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = np.sum(grad, axis=tuple(range(ndim_diff)))

    # 2. Sum over dimensions where target was 1 but grad was broadcast to > 1
    axes_to_sum = tuple(
        idx
        for idx, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape, strict=True))
        if t_dim == 1 and g_dim > 1
    )
    if axes_to_sum:
        grad = np.sum(grad, axis=axes_to_sum, keepdims=True)

    return np.asarray(grad.reshape(target_shape), dtype=np.float32)


# Public alias for gradient unbroadcasting
unbroadcast = _unbroadcast


class Tensor:
    """Multidimensional tensor supporting automatic reverse-mode differentiation.

    Parameters
    ----------
    data : ArrayLike
        Numerical array or convertible object (stored as float32).
    _children : tuple[Tensor, ...]
        Parent operand nodes that created this tensor.
    _op : str
        Elementary operation name for computation graph inspection.
    """

    _grad_enabled: bool = True

    @classmethod
    def set_grad_enabled(cls, enabled: bool) -> None:
        """Globally enable or disable gradient computation and graph retention."""
        cls._grad_enabled = enabled

    @classmethod
    def is_grad_enabled(cls) -> bool:
        """Check whether gradient computation is currently enabled."""
        return cls._grad_enabled

    def __init__(
        self,
        data: ArrayLike,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data: ArrayF32 = np.asarray(data, dtype=np.float32)
        self.grad: ArrayF32 = np.zeros_like(self.data, dtype=np.float32)
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Tensor] = set(_children) if Tensor._grad_enabled else set()
        self._op: str = _op

    @property
    def prev(self) -> set[Tensor]:
        """Parent operand tensors in the computation graph."""
        return self._prev

    def set_backward(self, fn: Callable[[], None]) -> None:
        """Assign gradient backpropagation closure to this tensor."""
        self._backward = fn

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype[np.float32]:
        return self.data.dtype

    def zero_grad(self) -> None:
        """Reset gradients to zeros."""
        self.grad.fill(0.0)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, shape={self.shape}, op='{self._op}')"

    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other_t.data, (self, other_t), "+")

        if Tensor._grad_enabled:

            def _backward() -> None:
                self.grad += _unbroadcast(out.grad, self.shape)
                other_t.grad += _unbroadcast(out.grad, other_t.shape)

            out._backward = _backward

        return out

    def __radd__(self, other: ArrayLike) -> Tensor:
        return self + other

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other_t.data, (self, other_t), "-")

        if Tensor._grad_enabled:

            def _backward() -> None:
                self.grad += _unbroadcast(out.grad, self.shape)
                other_t.grad += _unbroadcast(-out.grad, other_t.shape)

            out._backward = _backward

        return out

    def __rsub__(self, other: ArrayLike) -> Tensor:
        return Tensor(other) - self

    def __mul__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other_t.data, (self, other_t), "*")

        if Tensor._grad_enabled:

            def _backward() -> None:
                self.grad += _unbroadcast(other_t.data * out.grad, self.shape)
                other_t.grad += _unbroadcast(self.data * out.grad, other_t.shape)

            out._backward = _backward

        return out

    def __rmul__(self, other: ArrayLike) -> Tensor:
        return self * other

    def __truediv__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other_t.data, (self, other_t), "/")

        if Tensor._grad_enabled:

            def _backward() -> None:
                self.grad += _unbroadcast(out.grad / other_t.data, self.shape)
                other_t.grad += _unbroadcast(
                    -out.grad * self.data / (other_t.data**2), other_t.shape
                )

            out._backward = _backward

        return out

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        return Tensor(other) / self

    def __matmul__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        if self.ndim < 2 or other_t.ndim < 2:
            raise ValueError(
                f"Matmul requires 2D or higher tensors, got shapes {self.shape} and {other_t.shape}"
            )
        if self.shape[-1] != other_t.shape[-2]:
            raise ValueError(f"Inner matrix dimensions mismatch: {self.shape} @ {other_t.shape}")

        out = Tensor(self.data @ other_t.data, (self, other_t), "@")

        if Tensor._grad_enabled:

            def _backward() -> None:
                # dA = dC @ B^T
                grad_a = out.grad @ other_t.data.swapaxes(-1, -2)
                # dB = A^T @ dC
                grad_b = self.data.swapaxes(-1, -2) @ out.grad
                self.grad += _unbroadcast(grad_a, self.shape)
                other_t.grad += _unbroadcast(grad_b, other_t.shape)

            out._backward = _backward

        return out

    def relu(self) -> Tensor:
        """Element-wise Rectified Linear Unit activation: max(0, x)."""
        out = Tensor(np.maximum(self.data, 0.0), (self,), "ReLU")

        if Tensor._grad_enabled:

            def _backward() -> None:
                self.grad += np.where(self.data > 0.0, out.grad, 0.0).astype(np.float32)

            out._backward = _backward

        return out

    def tanh(self) -> Tensor:
        """Element-wise hyperbolic tangent activation."""
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        if Tensor._grad_enabled:

            def _backward() -> None:
                self.grad += (1.0 - t**2) * out.grad

            out._backward = _backward

        return out

    def exp(self) -> Tensor:
        """Element-wise exponential e^x."""
        out = Tensor(np.exp(self.data), (self,), "exp")

        if Tensor._grad_enabled:

            def _backward() -> None:
                self.grad += out.data * out.grad

            out._backward = _backward

        return out

    def log(self) -> Tensor:
        """Element-wise natural logarithm ln(x)."""
        if np.any(self.data <= 0.0):
            raise ValueError("Logarithm domain error: inputs must be strictly positive.")
        out = Tensor(np.log(self.data), (self,), "log")

        if Tensor._grad_enabled:

            def _backward() -> None:
                self.grad += (1.0 / self.data) * out.grad

            out._backward = _backward

        return out

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Sum tensor elements across specified axis/axes."""
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), "sum")

        if Tensor._grad_enabled:

            def _backward() -> None:
                grad = out.grad
                if axis is not None and not keepdims:
                    axes_tuple = (axis,) if isinstance(axis, int) else axis
                    # Reinsert reduced axes as size 1 dimensions for broadcasting
                    expanded_shape = list(self.shape)
                    for a in axes_tuple:
                        expanded_shape[a] = 1
                    grad = grad.reshape(expanded_shape)
                self.grad += np.broadcast_to(grad, self.shape)

            out._backward = _backward

        return out

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Compute arithmetic mean of tensor elements across specified axis/axes."""
        if axis is None:
            numel = float(self.size)
        elif isinstance(axis, int):
            numel = float(self.shape[axis])
        else:
            numel = float(np.prod([self.shape[a] for a in axis]))

        summed = self.sum(axis=axis, keepdims=keepdims)
        return summed * (1.0 / numel)

    def backward(self) -> None:
        """Compute reverse-mode gradients for ancestors via iterative topological sort."""
        topo: list[Tensor] = []
        visited: set[Tensor] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]

        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                for child in node._prev:
                    if child not in visited:
                        stack.append((child, False))

        self.grad = np.ones_like(self.data, dtype=np.float32)
        for node in reversed(topo):
            node._backward()


def softmax_cross_entropy(logits: Tensor, y: NDArray[np.int64] | Sequence[int]) -> Tensor:
    """Compute numerically stable Softmax Cross-Entropy loss.

    Formula:
        loss = - (1/N) * sum_i ln(softmax(logits_i)[y_i])

    Numerical Stability:
        Subtracts row-wise maximum logit m = max_j(z_j) before computing exponentials.
        This guarantees exp(z_j - m) <= 1.0 (eliminating overflow risk) and computes
        log-sum-exp directly without standalone log(0.0) underflow risks.

    Gradient:
        dLoss / dLogits = (1/N) * (softmax_probabilities - one_hot_labels)

    Parameters
    ----------
    logits : Tensor of shape (N, num_classes)
        Unnormalized logit predictions.
    y : NDArray[np.int64] or Sequence[int] of shape (N,)
        Ground-truth integer class labels.

    Returns
    -------
    Tensor
        Scalar average cross-entropy loss.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits tensor of shape (N, C), got {logits.shape}")

    n_samples = logits.shape[0]
    y_arr = np.asarray(y, dtype=np.int64)

    if y_arr.shape != (n_samples,):
        raise ValueError(f"Target labels shape {y_arr.shape} must match batch size ({n_samples},)")

    # 1. Numerically stable Log-Sum-Exp
    max_logits = np.max(logits.data, axis=-1, keepdims=True)
    shifted = logits.data - max_logits
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = shifted - np.log(sum_exp)

    # 2. Extract log probabilities of true classes
    sample_indices = np.arange(n_samples)
    loss_val: float = -float(np.mean(log_probs[sample_indices, y_arr]))  # pyright: ignore[reportUnknownArgumentType]

    out = Tensor(loss_val, (logits,), "CrossEntropy")

    # 3. Softmax probabilities for gradient calculation
    probs = exp_shifted / sum_exp

    if Tensor.is_grad_enabled():

        def _backward() -> None:
            grad = probs.copy()
            grad[sample_indices, y_arr] -= 1.0
            grad /= float(n_samples)
            logits.grad += grad * out.grad

        out.set_backward(_backward)

    return out
