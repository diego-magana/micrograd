"""
Gradient verification tests for the Value autograd engine.

Gradient checking is the epistemological foundation of any autograd
implementation. Two autograd systems that compute the same forward value
need not be applying the same gradient rules — a subtle sign error or a
missing accumulation can produce a result that looks fine but trains
catastrophically. The only reliable check is to compare gradients, node
by node, against an independent implementation on the same computation.

PyTorch is the reference. We mirror each computation in PyTorch with
``.double()`` precision (PyTorch defaults to 32-bit float, which would
otherwise floor the comparison precision at ~1e-7), set
``requires_grad=True`` so PyTorch tracks the graph, and assert that every
gradient agrees to within 1e-5 of the corresponding gradient our engine
produces.

What passing these tests proves:
    1. The chain rule is applied correctly at every primitive (forward and
       backward closures match).
    2. The graph is constructed correctly (parents, topological order,
       accumulation across multiple consumers).
    3. Derived operators (subtraction, division, negation) propagate
       gradient correctly through their primitive decompositions.

What it does *not* prove: numerical stability under extreme inputs, or
correctness of any operation not exercised by the tests. Coverage here is
sized to the operations needed to build an MLP and a softmax+NLL classifier.
"""

import torch

from micrograd.engine import Value


def test_sanity_check():
    """Verify a single-neuron computation against PyTorch.

    The expression is the canonical one from the micrograd lecture:
    a tanh-activated 2-input neuron with explicit weights and bias. If
    this test fails, the failure is almost certainly in __add__, __mul__,
    tanh, or the topological-sort traversal — every other primitive is
    untouched.
    """
    # Our engine.
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b = Value(6.8813735870195432)
    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()
    o.backward()

    # PyTorch reference. .double() forces float64 to match Python's native
    # float precision; without it, PyTorch's default float32 caps the
    # achievable comparison at ~1e-7 (and we want to assert tighter).
    x1_t = torch.tensor([2.0]).double().requires_grad_(True)
    x2_t = torch.tensor([0.0]).double().requires_grad_(True)
    w1_t = torch.tensor([-3.0]).double().requires_grad_(True)
    w2_t = torch.tensor([1.0]).double().requires_grad_(True)
    b_t = torch.tensor([6.8813735870195432]).double().requires_grad_(True)
    n_t = x1_t * w1_t + x2_t * w2_t + b_t
    o_t = torch.tanh(n_t)
    o_t.backward()

    # Forward values must agree.
    assert abs(o.data - o_t.item()) < 1e-5

    # Backward gradients must agree on every leaf.
    assert abs(x1.grad - x1_t.grad.item()) < 1e-5
    assert abs(x2.grad - x2_t.grad.item()) < 1e-5
    assert abs(w1.grad - w1_t.grad.item()) < 1e-5
    assert abs(w2.grad - w2_t.grad.item()) < 1e-5
    assert abs(b.grad - b_t.grad.item()) < 1e-5


def test_more_ops():
    """Verify a more complex expression that exercises every derived operator.

    The expression mixes ``+``, ``-``, ``*``, ``/``, ``**``, ``exp``, and ``log``,
    and crucially *reuses* the same Value multiple times so the multivariate
    chain rule (gradient accumulation across multiple downstream paths) is
    actually exercised. A failure here that did not show up in test_sanity_check
    most likely lives in __sub__, __truediv__, __neg__, exp, log, or in the
    accumulation logic (i.e. ``=`` somewhere it should have been ``+=``).
    """
    # Our engine.
    a = Value(-4.0)
    b = Value(2.0)
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
    a_grad_ours, b_grad_ours = a.grad, b.grad

    # PyTorch reference.
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
    assert abs(a_grad_ours - a_t.grad.item()) < 1e-5
    assert abs(b_grad_ours - b_t.grad.item()) < 1e-5


def test_softmax_nll():
    """Verify softmax + negative-log-likelihood loss against PyTorch.

    Softmax + NLL is the standard classification objective. The interesting
    property of this loss is that its gradient with respect to logit i has
    a remarkably clean analytical form:

        dL/dlogit_i = p_i - 1   if i is the true class
        dL/dlogit_i = p_i       otherwise

    where p = softmax(logits). Our engine does not know this identity — it
    just composes exp, division, and log primitives and lets backward()
    discover the same numbers from the chain rule. If this test passes,
    we've verified not only the individual primitives but also that they
    *compose correctly* into one of the most important loss functions in
    deep learning. If it fails, the most likely culprits are log() (which
    is not exercised by test_sanity_check) or accumulation through the
    shared denominator in softmax (every logit's exp() feeds the same sum
    — diamond pattern, accumulation must work).
    """
    # Our engine.
    logits = [Value(0.0), Value(3.0), Value(-2.0), Value(1.0)]
    counts = [logit.exp() for logit in logits]
    denominator = sum(counts)
    probs = [c / denominator for c in counts]
    loss = -probs[3].log()  # dim 3 is the true class label
    loss.backward()
    grads_ours = [logit.grad for logit in logits]

    # PyTorch reference.
    logits_t = torch.tensor([0.0, 3.0, -2.0, 1.0]).double().requires_grad_(True)
    probs_t = torch.softmax(logits_t, dim=0)
    loss_t = -probs_t[3].log()
    loss_t.backward()
    grads_torch = logits_t.grad.tolist()

    assert abs(loss.data - loss_t.item()) < 1e-5
    for i in range(4):
        assert abs(grads_ours[i] - grads_torch[i]) < 1e-5, \
            f"logit[{i}]: ours={grads_ours[i]}, torch={grads_torch[i]}"
