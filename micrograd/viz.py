"""Computation-graph visualization for Value objects.

Supplementary — not needed to use the engine or train networks. Given any Value,
it renders the DAG of operations that produced it, with data and gradient
annotated on every node, which makes the autograd graph concrete when explaining
or debugging. Requires graphviz; it is not imported by the package __init__, so
the rest of the library works without it.
"""

from graphviz import Digraph


def trace(root):
    """Collect every node and edge reachable from ``root`` by following ``_prev``.
    A visited set keeps it linear and handles diamonds (a Value feeding several
    consumers) without revisiting."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    """Render the graph rooted at ``root`` as a left-to-right graphviz Digraph.

    Each Value is a record node (label / data / grad); each operation is a small
    node between its operands and its result, so the graph is self-describing
    about what combined what. Returns a ``graphviz.Digraph`` — call ``.render()``
    or return it from a Jupyter cell to display inline.
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, shape='record',
                 label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad))
        if n._op:
            dot.node(name=uid + n._op, label=n._op)   # op node between operands and result
            dot.edge(uid + n._op, uid)

    for child, parent in edges:
        dot.edge(str(id(child)), str(id(parent)) + parent._op)

    return dot
