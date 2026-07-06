"""Gradient checks for the autograd engine, against PyTorch.

Two engines that agree on the forward value can still disagree on gradients — a
sign slip or a missing accumulation trains catastrophically while looking fine.
So the real test is gradient-by-gradient agreement on the same computation.
PyTorch is the reference, cast to ``.double()`` (its float32 default would floor
the comparison at ~1e-7; we want to assert 1e-5).
"""

import torch

from micrograd.engine import Value


def test_sanity_check():
    """A single tanh neuron — the canonical micrograd expression. A failure here
    is almost certainly in add, mul, tanh, or the topological traversal."""
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.8813735870195432)
    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()

    x1_t = torch.tensor([2.0]).double().requires_grad_(True)
    x2_t = torch.tensor([0.0]).double().requires_grad_(True)
    w1_t = torch.tensor([-3.0]).double().requires_grad_(True)
    w2_t = torch.tensor([1.0]).double().requires_grad_(True)
    b_t = torch.tensor([6.8813735870195432]).double().requires_grad_(True)
    o_t = torch.tanh(x1_t * w1_t + x2_t * w2_t + b_t)
    o_t.backward()

    assert abs(o.data - o_t.item()) < 1e-5
    for ours, ref in [(x1, x1_t), (x2, x2_t), (w1, w1_t), (w2, w2_t), (b, b_t)]:
        assert abs(ours.grad - ref.grad.item()) < 1e-5


def test_more_ops():
    """A tangled expression mixing +, -, *, /, **, exp, log, and — crucially —
    reusing the same nodes, so gradient accumulation across multiple paths is
    actually exercised. Catches a stray ``=`` where a ``+=`` belongs."""
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).exp()
    d = d + 3 * d + (b - a).exp()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g = g + f.log()
    g.backward()

    a_t = torch.tensor([-4.0]).double().requires_grad_(True)
    b_t = torch.tensor([2.0]).double().requires_grad_(True)
    c_t = a_t + b_t
    d_t = a_t * b_t + b_t ** 3
    c_t = c_t + c_t + 1
    c_t = c_t + 1 + c_t + (-a_t)
    d_t = d_t + d_t * 2 + (b_t + a_t).exp()
    d_t = d_t + 3 * d_t + (b_t - a_t).exp()
    e_t = c_t - d_t
    f_t = e_t ** 2
    g_t = f_t / 2.0
    g_t = g_t + 10.0 / f_t
    g_t = g_t + f_t.log()
    g_t.backward()

    assert abs(g.data - g_t.item()) < 1e-5
    assert abs(a.grad - a_t.grad.item()) < 1e-5
    assert abs(b.grad - b_t.grad.item()) < 1e-5


def test_relu_and_sigmoid():
    """Gradient checks for the two activations added for the classifier head:
    ReLU (kinked at 0) and the stable sigmoid (grad s*(1-s))."""
    for xval in (-2.0, -0.5, 0.5, 3.0):
        x = Value(xval)
        y = x.relu().sigmoid() + x.sigmoid()
        y.backward()
        x_t = torch.tensor([xval]).double().requires_grad_(True)
        y_t = torch.sigmoid(torch.relu(x_t)) + torch.sigmoid(x_t)
        y_t.backward()
        assert abs(y.data - y_t.item()) < 1e-5, xval
        assert abs(x.grad - x_t.grad.item()) < 1e-5, xval


def test_sigmoid_stable_on_large_inputs():
    """The stable sigmoid must not overflow at extreme magnitudes (the failure
    mode of a naive 1/(1+exp(-x)) with large negative x, and of the old
    algebraic tanh with large positive x)."""
    assert Value(-1000.0).sigmoid().data == 0.0
    assert abs(Value(1000.0).sigmoid().data - 1.0) < 1e-12
    assert abs(Value(1000.0).tanh().data - 1.0) < 1e-12   # would have overflowed pre-fix


def test_softmax_nll():
    """Softmax + NLL, built only from exp/div/log. The engine doesn't know the
    closed form dL/dlogit_i = p_i - 1{i=label}; backward() rederives it from the
    chain rule through the shared denominator (a diamond — accumulation must work).
    """
    logits = [Value(0.0), Value(3.0), Value(-2.0), Value(1.0)]
    counts = [l.exp() for l in logits]
    denom = sum(counts)
    probs = [c / denom for c in counts]
    loss = -probs[3].log()   # true class = 3
    loss.backward()
    grads_ours = [l.grad for l in logits]

    logits_t = torch.tensor([0.0, 3.0, -2.0, 1.0]).double().requires_grad_(True)
    loss_t = -torch.softmax(logits_t, dim=0)[3].log()
    loss_t.backward()

    assert abs(loss.data - loss_t.item()) < 1e-5
    for i in range(4):
        assert abs(grads_ours[i] - logits_t.grad[i].item()) < 1e-5
