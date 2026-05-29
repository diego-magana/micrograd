"""
Computation graph visualization for Value objects.

This module is supplementary — it is not required to use the engine or to
train networks. It exists to make the autograd graph concrete during
debugging and explanation: given any Value, render the full DAG of operations
that produced it, with both data and accumulated gradient annotated on every
node.

Requires graphviz (the Python binding and the system binary). If graphviz
is not installed, the rest of the library is unaffected — this module is
not imported by ``micrograd/__init__.py``.
"""

from graphviz import Digraph


def trace(root):
    """Walk the graph backward from ``root`` and collect all nodes and edges.

    Returns two sets: ``nodes`` (every Value reachable from root by following
    ``_prev`` pointers) and ``edges`` (every (child, parent) relationship).
    A recursive DFS with a visited set keeps the work linear in graph size
    and handles diamond patterns (where a Value feeds multiple consumers)
    without revisiting.
    """
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
    """Render the computation graph rooted at ``root`` as a graphviz Digraph.

    Layout convention: left-to-right (``rankdir='LR'``), matching the direction
    of forward computation. Each Value is drawn as a rectangular "record" node
    showing its label, data, and gradient. Each operation (op) is drawn as a
    separate small node positioned between its operand(s) and its output —
    this makes the graph self-describing about *what* combined the inputs.

    Returns:
        A graphviz.Digraph object. Call ``.render()`` to write SVG, or display
        it inline in a Jupyter notebook by returning it as the cell value.
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # Value node: rectangular, shows label / data / grad.
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape='record',
        )
        if n._op:
            # Op node: small, positioned between operands and result.
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for child, parent in edges:
        # Wire each child into the op node of its parent.
        dot.edge(str(id(child)), str(id(parent)) + parent._op)

    return dot
