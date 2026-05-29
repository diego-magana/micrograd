"""
Scalar-valued reverse-mode automatic differentiation engine.

This module implements a complete autograd system over scalar values. Every
arithmetic operation between Value instances implicitly extends a directed acyclic
computation graph; calling backward() on any node performs a topologically-ordered
reverse traversal, applying the chain rule at each node via local gradient rules
registered at forward time.

Design (three-phase per operation):

    1. Compute the forward value (out.data).
    2. Record the graph edges (out._prev = {self, other}).
    3. Register a closure (out._backward) that knows how to push gradient
       back through this operation when called.

This is the same conceptual design as PyTorch's autograd, restricted to scalars.
Three primitive operations (__add__, __mul__, __pow__) plus three elementary
functions (tanh, exp, log) are sufficient — every other operator in the public
interface (subtraction, division, negation, the right-hand reflections) composes
from these primitives and inherits gradient correctness for free.
"""

import math


class Value:
    """A scalar value tracked in an autograd computation graph.

    Each Value carries four pieces of state beyond its numeric data:

        grad      : the partial derivative dL/d(self) accumulated during backward()
        _backward : a closure that, when called, distributes self.grad to its
                    children using the local derivative rule of the operation
                    that produced this node
        _prev     : the set of immediate parents in the computation graph
                    (i.e. the operands of the operation that produced self)
        _op       : a short string label for the op, used by visualization

    The graph builds itself implicitly: writing ``a * b + c`` constructs three
    Value nodes (a*b, then (a*b)+c), each pointing to its operands and each
    carrying its own _backward closure. No explicit graph builder is needed.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        """Initialize a Value node.

        Public parameters:
            data  : the scalar this node represents (int or float)
            label : optional human-readable name, used only by visualization

        Internal parameters (set by operator methods, not by users directly):
            _children : tuple of parent Value nodes — the operands of the op
                        that produced this node
            _op       : string identifier for the op ('+', '*', 'tanh', ...)

        Gradient is initialized to 0.0, not None — backward() relies on being
        able to do ``self.grad += contribution`` unconditionally without first
        checking for None. The output node's gradient is set to 1.0 by
        backward() itself (dL/dL = 1) before the reverse traversal begins.

        _backward is initialized to a no-op for leaves (inputs and parameters)
        because leaves have no children to propagate gradient to. Operator
        methods overwrite this closure with the appropriate local rule.
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # ------------------------------------------------------------------
    # Primitive operations: every other operator composes from these three.
    # ------------------------------------------------------------------

    def __add__(self, other):
        """Elementwise addition with gradient registration.

        Math:
            d/da (a + b) = 1
            d/db (a + b) = 1

        Both operands receive the full upstream gradient unchanged — addition
        is a "gradient distributor." The local Jacobian is identity, so the
        chain rule reduces to copying out.grad to each input.

        Why it matters for neural networks: every linear layer is a sum of
        products (Σ w_i x_i + b). Without correct gradient distribution
        through summation, no biased linear combination can be trained.

        Implementation notes:
            - ``other`` is wrapped if it's a plain int/float so that
              expressions like ``a + 1.0`` work. This is the symmetric
              case to __radd__ below.
            - Gradient accumulation uses ``+=`` not ``=``: a single Value
              may feed multiple downstream consumers, and all contributions
              must be summed (multivariate chain rule). Using ``=`` here
              would silently drop gradients in any expression that reuses
              a variable, e.g. ``y = x * x``.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """Elementwise multiplication with gradient registration.

        Math:
            d/da (a * b) = b
            d/db (a * b) = a

        Each input's gradient is the *other* input's value times the upstream
        gradient. Multiplication is therefore a "gradient swapper": the local
        derivative at one input depends entirely on the partner's data.

        Why it matters for neural networks: weights multiply activations.
        The gradient w.r.t. a weight is the corresponding input activation
        times the upstream gradient — this is exactly the local rule below
        and is the source of the classical "input × error" learning signal
        in the original backprop derivations.

        Implementation note: the data of ``other`` is captured by closure at
        forward time. If ``other.data`` were mutated between forward and
        backward (which we never do, but is worth understanding), the closure
        would see the latest value rather than the snapshot. Same caveat
        applies symmetrically to ``self.data``.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """Power operation with a constant exponent.

        Math:
            d/da (a ** n) = n * a**(n - 1)        (power rule)

        The exponent is required to be a plain int/float — not a Value. We do
        not support differentiating with respect to the exponent here. The
        general case (a ** b with b also a Value) requires the identity
        a**b = exp(b * log(a)), and the gradient w.r.t. b is a**b * log(a).
        That extra machinery is unnecessary for the operations needed to
        build an MLP, so we keep the interface tight and assert.

        This single primitive gives us reciprocals (a ** -1), square roots
        (a ** 0.5), and arbitrary fixed-exponent powers. Division then comes
        for free as ``self * (other ** -1)`` (see __truediv__ below).
        """
        assert isinstance(other, (int, float)), \
            "only int/float exponents supported (no gradient flows to the exponent)"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    # ------------------------------------------------------------------
    # Elementary functions: nonlinearity (tanh) and the analytic primitives
    # (exp, log) needed to build softmax + NLL.
    # ------------------------------------------------------------------

    def tanh(self):
        """Hyperbolic tangent — the classic neuron nonlinearity.

        Math:
            tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
            d/dx tanh(x) = sech^2(x) = 1 - tanh^2(x)

        The derivative identity ``1 - tanh^2(x)`` is what makes tanh cheap to
        backprop through: we already computed ``t = tanh(x)`` on the forward
        pass, so the local gradient is just ``(1 - t**2)`` — no new transcendental
        evaluation required.

        Why it matters for neural networks: tanh squashes any real number into
        the open interval (-1, 1). This bounded, zero-centered output is what
        gives a single neuron the ability to act as a soft binary decision and
        what lets stacks of neurons compose into nonlinear function approximators.
        Without a nonlinearity, an MLP collapses algebraically to a single
        linear transformation regardless of depth.

        Saturation caveat: for |x| >> 1, tanh(x) ≈ ±1 and the gradient
        ``1 - t^2`` collapses to ~0 — the vanishing-gradient regime. This is
        why feature normalization to roughly unit scale matters for tanh
        networks (and is why the moons_demo notebook normalizes inputs).
        """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        """Natural exponential.

        Math:
            d/dx e^x = e^x

        The exponential is the unique (up to scale) function that is its own
        derivative — which is why ``out.data`` (already equal to e^x) is the
        correct local multiplier and no fresh transcendental evaluation is
        needed in the backward pass.

        Why it matters for neural networks: exp is the building block of
        softmax. Combined with division and log (below), it lets us express
        a categorical probability distribution and its log-likelihood as a
        pure composition of differentiable primitives — meaning the engine
        recovers the analytical softmax+NLL gradient ``p_i - 1{i = label}``
        automatically, with no special-cased layer.
        """
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        """Natural logarithm.

        Math:
            d/dx ln(x) = 1/x          (defined for x > 0)

        Why it matters for neural networks: log is the second half of NLL
        (negative log-likelihood) loss. Given a softmax distribution ``p``,
        the loss for the true class label k is ``loss = -p[k].log()``. The
        log turns the product-of-probabilities likelihood into a sum, which
        is numerically stable and lets gradients propagate through every
        class score additively.

        Domain note: log is undefined at zero and negative for x < 1. In
        practice this is fine because log is composed with softmax outputs
        which are strictly positive. Numerical underflow (p[k] = 0 to float
        precision) is a known issue addressed in production systems by the
        log-sum-exp trick, which we do not implement here — the scalar
        engine is for understanding, not for training at scale.
        """
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward

        return out

    # ------------------------------------------------------------------
    # Derived operators — no new backward rules required.
    #
    # Subtraction is addition of a negation. Division is multiplication by
    # a power. Negation is multiplication by -1. Every gradient flows
    # correctly through the same three primitives (+, *, **) already defined
    # above. This is the central elegance of the design: the public surface
    # is rich, but the gradient bookkeeping lives in only three places.
    # ------------------------------------------------------------------

    def __neg__(self):
        """Unary negation: -a = a * -1."""
        return self * -1

    def __sub__(self, other):
        """Subtraction: a - b = a + (-b)."""
        return self + (-other)

    def __truediv__(self, other):
        """Division: a / b = a * b**(-1).

        Composes through ``__mul__`` and ``__pow__``: if ``other`` is a Value,
        ``other ** -1`` dispatches to our ``__pow__``; if it is a plain scalar,
        Python's native exponent handles it before ``__mul__`` sees the result.
        Either path produces correct gradients without new backward logic.
        Scalar-on-the-left (``2.0 / value``) is handled by ``__rtruediv__``.
        """
        return self * other ** -1

    def __radd__(self, other):
        """Reverse addition: handles ``scalar + Value``.

        Without this, Python's builtin ``sum()`` would fail. ``sum()`` starts
        from the integer 0 and calls ``0 + first_element``, which dispatches
        to ``int.__add__(Value)`` — that returns NotImplemented, and Python
        then falls back to ``Value.__radd__(0)``. We delegate to __add__,
        which is commutative for our purposes.
        """
        return self + other

    def __rmul__(self, other):
        """Reverse multiplication: handles ``scalar * Value``.

        Same fallback mechanism as __radd__: ``2 * value`` dispatches first
        to ``int.__mul__``, fails, and falls back here. We delegate to __mul__.
        """
        return self * other

    def __rsub__(self, other):
        """Reverse subtraction: handles ``scalar - Value``.

        Cannot simply delegate to __sub__ (which is non-commutative) — we
        explicitly construct ``Value(other) + (-self)`` to get the operand
        order right.
        """
        return Value(other) + (-self)

    def __rtruediv__(self, other):
        """Reverse division: handles ``scalar / Value``.

        Cannot delegate to __truediv__ (non-commutative). We construct the
        composition explicitly: ``other / self = other * self**(-1)``. The
        wrap of ``other`` into a Value is for symmetry with __rsub__ — it
        avoids relying on a second hop through Python's reflection chain
        (without it, ``other * self**-1`` would route through __rmul__ to
        get the same answer; explicit wrap makes the dependency clear).
        """
        return Value(other) * self ** -1

    # ------------------------------------------------------------------
    # Reverse-mode automatic differentiation.
    # ------------------------------------------------------------------

    def backward(self):
        """Compute gradients of self with respect to every node in its graph.

        Algorithm:
            1. Build a topological order of the graph rooted at self,
               with children appearing before parents.
            2. Seed self.grad = 1.0 (because dL/dL = 1 by definition,
               where L is the output node).
            3. Traverse the topological order in reverse, calling each
               node's _backward closure. Each closure pushes gradient
               from the node onto its operands using the chain rule with
               that node's local derivative.

        Why topological order is required: a node's _backward distributes
        out.grad to its children. For the distribution to be correct,
        out.grad must already contain *all* contributions from downstream
        consumers — otherwise the children receive a partial answer and
        their own _backward calls will propagate the wrong number further
        back. Topological sort guarantees every node is fully accumulated
        before being read. This is the same dependency resolution that a
        compiler uses to schedule expression evaluation.

        Why ``self.grad = 1.0`` is the correct seed: we are computing the
        gradient *of self* w.r.t. every ancestor. By definition the partial
        derivative of any quantity with respect to itself is 1, and that
        seed is what allows the chain rule to unroll correctly down the
        graph. If self represents a scalar loss L, then ``self.grad = 1.0``
        means we are asking "how does L change with respect to each upstream
        variable?" — which is exactly what gradient descent needs.

        Why ``+=`` in the closures, not ``=``: a Value may feed multiple
        downstream consumers (e.g. ``y = x * x`` reuses x; ``z = a + a*b``
        reuses a). The multivariate chain rule requires summing contributions
        from every downstream path. The leaf gradients are initialized to
        0.0 by __init__ precisely so that this unconditional accumulation
        starts from a clean zero.
        """
        # Build topological order: every node appears after all its children.
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                # Recurse into operands first — they must come before v
                # in the list because they need to receive gradient from v
                # *after* v has been fully accumulated from its consumers.
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Seed and propagate.
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
