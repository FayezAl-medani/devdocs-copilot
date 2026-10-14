"""Neural network modules built on top of scalar Value autograd engine."""

from __future__ import annotations

import random
from collections.abc import Sequence

from devdocs.autograd.value import Value


class Module:
    """Base neural network module interface."""

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to 0.0."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        """Return list of trainable parameters."""
        return []


class Neuron(Module):
    """A single artificial neuron with linear combination and non-linear activation.

    Parameters
    ----------
    nin : int
        Number of incoming input features.
    nonlin : bool
        Whether to apply non-linear tanh activation.
    activation : {"tanh", "relu", "linear"}
        Activation function choice.
    seed : int or None
        Optional random seed for reproducible weight initialization.
    """

    def __init__(
        self,
        nin: int,
        nonlin: bool = True,
        activation: str = "tanh",
        seed: int | None = None,
    ) -> None:
        if nin <= 0:
            raise ValueError(f"nin must be positive, got {nin}")
        rng = random.Random(seed)
        # Xavier/Glorot initialization scale
        scale = (2.0 / nin) ** 0.5
        self.w: list[Value] = [Value(rng.uniform(-scale, scale)) for _ in range(nin)]
        self.b: Value = Value(0.0)
        self.nonlin: bool = nonlin
        self.activation: str = activation

    def __call__(self, x: Sequence[Value | float | int]) -> Value:
        if len(x) != len(self.w):
            raise ValueError(
                f"Input dimension mismatch: expected {len(self.w)} features, got {len(x)}"
            )
        # sum(w_i * x_i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        if not self.nonlin or self.activation == "linear":
            return act
        if self.activation == "tanh":
            return act.tanh()
        if self.activation == "relu":
            return act.relu()
        raise ValueError(f"Unsupported activation: {self.activation}")

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    """A fully connected (dense) neural network layer.

    Parameters
    ----------
    nin : int
        Input dimensionality.
    nout : int
        Output dimensionality (number of neurons).
    nonlin : bool
        Whether neurons apply non-linear activation.
    activation : str
        Activation function ("tanh", "relu", "linear").
    seed : int or None
        Random seed for parameter initialization.
    """

    def __init__(
        self,
        nin: int,
        nout: int,
        nonlin: bool = True,
        activation: str = "tanh",
        seed: int | None = None,
    ) -> None:
        if nout <= 0:
            raise ValueError(f"nout must be positive, got {nout}")
        self.neurons: list[Neuron] = [
            Neuron(
                nin,
                nonlin=nonlin,
                activation=activation,
                seed=seed + i if seed is not None else None,
            )
            for i in range(nout)
        ]

    def __call__(self, x: Sequence[Value | float | int]) -> list[Value] | Value:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class MLP(Module):
    """Multi-Layer Perceptron architecture.

    Parameters
    ----------
    nin : int
        Input feature dimension.
    nouts : Sequence[int]
        Number of units in each layer (hidden layers and output layer).
    activation : str
        Activation function for hidden layers ("tanh" or "relu").
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        nin: int,
        nouts: Sequence[int],
        activation: str = "tanh",
        seed: int | None = None,
    ) -> None:
        if not nouts:
            raise ValueError("nouts must specify at least one layer size.")
        sz = [nin] + list(nouts)
        self.layers: list[Layer] = []
        for i in range(len(nouts)):
            is_output_layer = i == len(nouts) - 1
            layer_seed = seed + i * 100 if seed is not None else None
            self.layers.append(
                Layer(
                    sz[i],
                    sz[i + 1],
                    nonlin=not is_output_layer,
                    activation=activation,
                    seed=layer_seed,
                )
            )

    def __call__(self, x: Sequence[Value | float | int]) -> list[Value] | Value:
        current: list[Value] | Value = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            current_seq: Sequence[Value | float | int] = (
                current if isinstance(current, list) else [current]
            )
            current = layer(current_seq)
        return current

    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
