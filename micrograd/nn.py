"""
Neural network primitives built on top of the scalar autograd engine.

This module provides the three layers of composition needed to express a
multilayer perceptron in terms of Value scalars:

    Neuron : a single unit — weighted sum of inputs through a tanh nonlinearity
    Layer  : a collection of neurons that share the same input
    MLP    : a stack of layers, threaded forward via __call__

There are no matrix operations here. Every multiplication, addition, and
nonlinearity is a Value-level operation, which means every operation also
registers a backward closure. Calling backward() on the final output is
sufficient to populate gradients on every parameter — no separate Module /
Parameter / autograd-tape abstraction is needed.

This is the entire surface area required to reproduce the demonstrations in
the original micrograd lecture: classification, regression, arbitrary depth.
The engine does not know it is being used for neural networks; it works for
any composition of differentiable scalar operations.
"""

import random

from micrograd.engine import Value


class Neuron:
    """A single neuron: tanh(Σ w_i * x_i + b).

    The neuron holds a vector of weights w and a scalar bias b, all of which
    are Value instances so that gradients flow into them during backward().
    Forward evaluation is the standard linear-then-nonlinear unit; backward
    is delegated entirely to the engine.

    Initialization:
        Weights and bias are drawn uniformly from [-1, 1]. This range is
        small enough that the pre-activation Σ w_i x_i + b stays in the
        non-saturating region of tanh for normalized inputs (where the
        gradient is largest), but large enough to break symmetry between
        neurons in a layer so they learn different features.

        This is the same init used in Karpathy's original micrograd. It is
        not Xavier or Kaiming init — those scale by fan-in and matter when
        depths grow. At the depths we care about (2-3 layers, single-digit
        widths), uniform(-1, 1) is sufficient and instructive.

    Nonlinearity choice:
        tanh is used here for three reasons: (1) it is symmetric and
        zero-centered, which keeps activations from drifting away from
        zero across layers; (2) it has a clean closed-form derivative
        (1 - t^2) using the forward value, making backprop efficient;
        (3) it is the nonlinearity Karpathy used in the source lecture,
        and matching it makes gradient checks against that reference
        meaningful. ReLU would also work and could be added with a few
        lines (Value would need a relu() method).
    """

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """Forward pass: tanh(Σ w_i x_i + b).

        We seed ``sum`` with ``self.b`` rather than the default 0 so the
        first accumulator is already a Value — this avoids relying on
        __radd__ for the very first step (it still works either way; this
        is just clearer).
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        """All trainable Values in this neuron: weights followed by bias."""
        return self.w + [self.b]


class Layer:
    """A layer of ``nout`` neurons, each receiving the same ``nin`` inputs.

    The layer holds a list of neurons and produces a list of outputs. There
    is no weight sharing within a layer — each neuron has its own independent
    parameters.

    Interface convention:
        When ``nout == 1`` the layer unwraps and returns a single Value rather
        than a one-element list. This makes downstream code work uniformly
        whether the final layer produces a scalar (regression, binary
        classification) or a vector (multiclass classification) — the
        caller does not need to special-case the trailing dimension.
    """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """Flatten parameters across all neurons in the layer."""
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Multilayer perceptron: a stack of fully connected Layers.

    The architecture is specified by an input width ``nin`` and a list
    ``nouts`` of layer widths. For example, ``MLP(2, [16, 16, 1])`` builds:

        Layer(2, 16) -> Layer(16, 16) -> Layer(16, 1)

    Construction:
        The size sequence ``sz = [nin] + nouts`` lets us pair adjacent
        elements ``(sz[i], sz[i+1])`` to specify each layer's (in, out)
        dimensions. This is a clean idiom for "convert a list of widths
        into a list of adjacent-pair tuples."

    parameters():
        Returns a *flat* list of every Value in the network. Gradient
        descent operates on individual scalar parameters, not on structured
        objects (layers, neurons) — so a flat list is exactly the right
        granularity for the training loop's ``for p in model.parameters()``
        pattern.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        """Forward pass: thread x through every layer in order."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Flat list of every trainable Value in the network."""
        return [p for layer in self.layers for p in layer.parameters()]
