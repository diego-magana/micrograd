"""Neural-network primitives over the scalar engine.

Three levels of composition — Neuron, Layer, MLP — are enough to express a
multilayer perceptron entirely in ``Value`` scalars. There are no matrices
here; every weight, product, and sum is a scalar op that registers its own
backward closure, so a single ``loss.backward()`` populates gradients on every
parameter. No Module/Parameter/tape machinery needed.
"""

import random

from micrograd.engine import Value


class Neuron:
    """A single unit: optionally-activated affine map ``f(Σ w_i x_i + b)``.

    ``nonlin=True`` applies a tanh nonlinearity; ``nonlin=False`` leaves the
    unit linear. The linear form is what you want for an output that feeds a
    sigmoid/softmax — a squashed pre-logit would cap confidence and inflate the
    loss (the reason a classification head must be linear, not tanh).

    Weights and bias start from ``uniform(-1, 1)``: small enough that the
    pre-activation stays in tanh's non-saturating region for normalized inputs,
    large enough to break symmetry so neurons in a layer learn different
    features. That's the same init as Karpathy's micrograd — not fan-in-scaled
    Xavier/Kaiming, which only starts to matter at greater depth than this.
    """

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        # Seed the sum with the bias so the accumulator is a Value from step one.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer:
    """``nout`` neurons over the same inputs. Returns a bare Value when
    ``nout == 1`` (regression / binary heads) and a list otherwise, so callers
    never special-case the trailing dimension."""

    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        n = self.neurons[0]
        kind = 'tanh' if n.nonlin else 'linear'
        return f"Layer({len(n.w)}->{len(self.neurons)}, {kind})"


class MLP:
    """A stack of Layers. ``MLP(2, [16, 16, 3])`` builds 2→16→16→3.

    The output layer is linear by default (``nonlin=False``); every hidden layer
    is tanh-activated. That default is the whole point of the ``nonlin`` flag:
    hidden layers need a nonlinearity to compose, but the head must stay linear
    so it can emit unbounded logits for a sigmoid or softmax.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        last = len(nouts) - 1
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != last))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        """Reset every parameter's gradient. Call before each backward pass —
        gradients accumulate with ``+=``, so a skipped reset silently sums this
        step's gradient onto the last one's. It's the most common hand-rolled
        training bug; this method exists so the loop can't forget."""
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        sizes = [len(self.layers[0].neurons[0].w)] + [len(l.neurons) for l in self.layers]
        chain = ' -> '.join(str(s) for s in sizes)
        desc = "linear" if len(self.layers) == 1 else "hidden tanh, linear head"
        return f"MLP({chain}; {desc})"
