"""Scalar automatic differentiation engine (micrograd-style)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


class no_grad:
    """Context manager to temporarily disable gradient tracking and graph construction."""

    _prev_state: bool = True

    def __enter__(self) -> None:
        self._prev_state = Value.is_grad_enabled()
        Value.set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        Value.set_grad_enabled(self._prev_state)


class Value:
    """Represents a scalar value with automatic gradient calculation via dynamic DAG tracking.

    Parameters
    ----------
    data : float
        Numerical scalar value.
    _children : tuple[Value, ...]
        Parent operand nodes that produced this value in the computational graph.
    _op : str
        Elementary operation symbol for debugging and graph visualization.
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
        data: float | int,
        _children: tuple[Value, ...] = (),
        _op: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Value] = set(_children) if Value._grad_enabled else set()
        self._op: str = _op

    @property
    def prev(self) -> set[Value]:
        """Parent nodes in the computational graph."""
        return self._prev

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other: Value | float | int) -> Value:
        other_val = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other_val.data, (self, other_val), "+")

        if Value._grad_enabled:

            def _backward() -> None:
                self.grad += out.grad
                other_val.grad += out.grad

            out._backward = _backward

        return out

    def __radd__(self, other: float | int) -> Value:
        return self + other

    def __mul__(self, other: Value | float | int) -> Value:
        other_val = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other_val.data, (self, other_val), "*")

        if Value._grad_enabled:

            def _backward() -> None:
                self.grad += other_val.data * out.grad
                other_val.grad += self.data * out.grad

            out._backward = _backward

        return out

    def __rmul__(self, other: float | int) -> Value:
        return self * other

    def __pow__(self, n: float | int) -> Value:
        power = float(n)
        out = Value(self.data**power, (self,), f"**{power}")

        if Value._grad_enabled:

            def _backward() -> None:
                self.grad += (power * (self.data ** (power - 1.0))) * out.grad

            out._backward = _backward

        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float | int) -> Value:
        return self + (-other)

    def __rsub__(self, other: float | int) -> Value:
        return Value(other) - self

    def __truediv__(self, other: Value | float | int) -> Value:
        other_val = other if isinstance(other, Value) else Value(other)
        return self * (other_val**-1.0)

    def __rtruediv__(self, other: float | int) -> Value:
        return Value(other) / self

    def tanh(self) -> Value:
        """Compute hyperbolic tangent activation: tanh(x) = (e^2x - 1) / (e^2x + 1)."""
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        if Value._grad_enabled:

            def _backward() -> None:
                self.grad += (1.0 - t**2) * out.grad

            out._backward = _backward

        return out

    def relu(self) -> Value:
        """Compute Rectified Linear Unit activation: max(0, x)."""
        out = Value(max(0.0, self.data), (self,), "ReLU")

        if Value._grad_enabled:

            def _backward() -> None:
                self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

            out._backward = _backward

        return out

    def exp(self) -> Value:
        """Compute exponential function e^x."""
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        if Value._grad_enabled:

            def _backward() -> None:
                self.grad += out.data * out.grad

            out._backward = _backward

        return out

    def log(self) -> Value:
        """Compute natural logarithm ln(x)."""
        if self.data <= 0.0:
            raise ValueError(f"Logarithm domain error: expected positive input, got {self.data}")
        out = Value(math.log(self.data), (self,), "log")

        if Value._grad_enabled:

            def _backward() -> None:
                self.grad += (1.0 / self.data) * out.grad

            out._backward = _backward

        return out

    def backward(self) -> None:
        """Compute exact gradients of all ancestor nodes via iterative topological backpropagation.

        Design Principles:
        1. Non-recursive (Iterative DFS): uses an explicit call stack to eliminate Python
           recursion limit errors on deep computational graphs (e.g. 50+ layer networks).
        2. Seed Gradient: root node starts backpropagation with grad = 1.0 (dL/dL = 1).
        3. Accumulation (+=): gradients from multiple downstream consumers sum correctly via
           the multivariable calculus chain rule.
        """
        topo: list[Value] = []
        visited: set[Value] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]

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

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
