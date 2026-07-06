"""Scalar reverse-mode autograd.

Every operation on a ``Value`` extends a computation graph; ``backward()`` walks
that graph in reverse topological order and applies the chain rule at each node.
Three primitives — add, multiply, power — plus a handful of elementary functions
are enough: subtraction, division, negation, and the reflected operators all
compose from the primitives and inherit correct gradients with no new backward
rules.

It's PyTorch's autograd idea restricted to scalars — one ``Value`` per number,
one backward closure per operation — built for understanding, not for speed.
"""

import math


class Value:
    """A scalar tracked in an autograd graph.

    Beyond its ``data``, each node holds the gradient ``dL/dself`` accumulated
    during ``backward()``, the set of parent nodes it was built from (``_prev``),
    a ``_backward`` closure encoding the local derivative rule, and an ``_op``
    label used only by the visualizer. The graph builds itself: writing
    ``a * b + c`` constructs the nodes and wires their closures automatically.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # Leaves have no children to push gradient to, so their closure is a
        # no-op; operator methods overwrite it. grad starts at 0.0 (not None)
        # so backward() can accumulate with += unconditionally.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # -- Primitives: the only three nodes that register backward rules. --------

    def __add__(self, other):
        """a + b. Addition distributes gradient unchanged: d/da = d/db = 1."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # += (not =) because a node reused across paths must sum its
            # contributions — the multivariate chain rule.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        """a * b. Each operand's local grad is the *other* operand: d/da = b, d/db = a."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        """a ** n for constant n. Power rule: d/da = n * a**(n-1).

        The exponent must be a plain scalar — I don't flow gradient into it,
        which keeps the interface tight and is all an MLP needs. This one
        primitive gives reciprocals (n=-1) and roots (n=0.5), so division
        falls out of it for free.
        """
        assert isinstance(other, (int, float)), \
            "only int/float exponents supported (no gradient flows to the exponent)"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    # -- Elementary functions. -------------------------------------------------

    def tanh(self):
        """tanh nonlinearity. Local grad reuses the forward value: 1 - t**2.

        I use ``math.tanh`` rather than the algebraic ``(e^2x-1)/(e^2x+1)`` form
        so large-magnitude inputs saturate cleanly instead of overflowing.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """ReLU. Gradient passes where the input is positive, and is zero elsewhere."""
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        """Logistic sigmoid. Local grad reuses the forward value: s * (1 - s).

        Computed in the stable branch-by-sign form so large negative inputs
        don't overflow ``exp``.
        """
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        """e^x. It's its own derivative, so the forward value is the local grad."""
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        """Natural log, x > 0. Local grad 1/x. Composed with softmax outputs,
        which are strictly positive, so the domain is safe in practice."""
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # -- Derived operators: no new backward rules; they compose from the above. -

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):    # scalar + Value  (lets sum() start from 0)
        return self + other

    def __rmul__(self, other):    # scalar * Value
        return self * other

    def __rsub__(self, other):    # scalar - Value  (non-commutative: build explicitly)
        return Value(other) + (-self)

    def __rtruediv__(self, other):  # scalar / Value
        return Value(other) * self ** -1

    # -- Reverse-mode autodiff. ------------------------------------------------

    def backward(self):
        """Populate ``.grad`` on every node in this graph, treating ``self`` as
        the output (dL/dself = 1).

        A node's closure distributes ``out.grad`` to its children, so every node
        must be fully accumulated from its downstream consumers before it fires.
        Topological order guarantees that: build children-before-parents, then
        walk it in reverse.
        """
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
