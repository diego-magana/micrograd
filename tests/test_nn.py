"""Structural and training checks for the nn layer.

The engine tests prove gradients are correct; these prove the network built on
top wires up correctly — parameter counts, output shapes, the linear head, and
that a real training loop actually drives loss down.
"""

import random

from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP


def test_parameter_counts():
    # Neuron(nin): nin weights + 1 bias.
    assert len(Neuron(5).parameters()) == 6
    # MLP(2,[16,16,1]): (2+1)*16 + (16+1)*16 + (16+1)*1 = 48 + 272 + 17 = 337.
    assert len(MLP(2, [16, 16, 1]).parameters()) == 337
    # MLP(2,[16,16,3]) softmax head: 48 + 272 + (16+1)*3 = 371.
    assert len(MLP(2, [16, 16, 3]).parameters()) == 371


def test_output_head_is_linear():
    """The MLP head must be linear so it can emit unbounded logits; hidden
    layers must be nonlinear so depth composes."""
    m = MLP(2, [16, 16, 3])
    assert all(n.nonlin for n in m.layers[0].neurons)       # hidden: tanh
    assert all(not n.nonlin for n in m.layers[-1].neurons)  # head: linear
    # A linear head can exceed tanh's (-1, 1) range.
    random.seed(0)
    m = MLP(2, [8, 1])
    assert any(abs(m([Value(3.0), Value(-3.0)]).data) > 1.0 for _ in range(1))


def test_layer_return_shape():
    # nout == 1 unwraps to a bare Value; nout > 1 returns a list.
    assert isinstance(Layer(3, 1)([Value(1.0)] * 3), Value)
    out = Layer(3, 4)([Value(1.0)] * 3)
    assert isinstance(out, list) and len(out) == 4


def test_zero_grad():
    m = MLP(2, [4, 1])
    out = m([Value(1.0), Value(-1.0)])
    out.backward()
    assert any(p.grad != 0.0 for p in m.parameters())
    m.zero_grad()
    assert all(p.grad == 0.0 for p in m.parameters())


def test_training_reduces_loss():
    """A minimal training smoke test: a few SGD steps on a trivially separable
    pair of points must reduce a squared-error loss. Guards against a loop that
    builds gradients but never moves the parameters the right way."""
    random.seed(1)
    m = MLP(2, [4, 1])
    data = [([2.0, 3.0], 1.0), ([-2.0, -1.0], -1.0)]

    def loss():
        return sum((m([Value(a), Value(b)]) - y) ** 2 for (a, b), y in data)

    first = loss().data
    for _ in range(20):
        L = loss()
        m.zero_grad()
        L.backward()
        for p in m.parameters():
            p.data -= 0.05 * p.grad
    assert loss().data < first * 0.5   # at least halved
