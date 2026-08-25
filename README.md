# micrograd

![CI](https://github.com/diego-magana/micrograd/actions/workflows/ci.yml/badge.svg)

A scalar autograd engine, small enough to read end to end. It builds a computation graph as you do arithmetic, then backpropagates through it — the same mechanism as PyTorch's loss.backward(), on scalars instead of tensors.

This is the first repo in a series — **micrograd → [makemore](https://github.com/diego-magana/makemore) → [gpt](https://github.com/diego-magana/gpt)** —
that builds up from a single differentiable scalar to a transformer with
interpretability tooling. micrograd is the foundation — the autograd mechanism everything else assumes.

If you want the short version, read [`notebooks/spirals_demo.ipynb`](https://github.com/diego-magana/micrograd/blob/main/notebooks/spirals_demo.ipynb) — it's the
whole thing end to end with outputs, from the engine to a trained classifier.

## What it does

A `Value` wraps one scalar and overloads Python's operators. Writing `a * b + c`
constructs the graph as a byproduct of Python's operator dispatch; `loss.backward()`
walks it in reverse topological order and applies the chain rule at each node. It's
the same algorithm PyTorch's `loss.backward()` runs, restricted to scalars so that
every step is visible.

The engine and the nn library are **pure standard library** (`math`, `random`) —
no NumPy, no framework. NumPy and PyTorch appear only in the demo and the tests.

## How it works

Each `Value` carries its data, its gradient, the parents it was built from, and
a backward closure encoding the local derivative rule. Three arithmetic
operators — addition, multiplication, and power — are the only *operators* that
register gradient rules; subtraction, division, negation, and the reflected
variants all compose from those three and inherit correct gradients for free.
Five elementary functions register their own (`tanh`, `exp`, `log`, plus `relu`
and `sigmoid`), and that's the whole surface needed to express an MLP and a
softmax classifier.

Two details carry more weight than they look:

- **Gradients accumulate with `+=`, not `=`.** A node reused across paths (`y = x*x`
  reuses `x`) must sum the contributions from every downstream path — the
  multivariate chain rule. Leaf gradients start at `0.0` so that accumulation is
  always safe.
- **`backward()` needs topological order.** A node's closure distributes its
  gradient to its parents, so it can only fire once it has received the full
  gradient from all of its own consumers. Building children-before-parents and
  walking that order in reverse guarantees it — the same dependency resolution a
  compiler uses to schedule an expression.

Softmax + negative-log-likelihood is built straight from the primitives; the
engine rediscovers the clean closed-form gradient (`p_i − 1` for the true class,
`p_i` otherwise) from the chain rule alone, with no special-cased layer. I verify
every gradient against PyTorch to 1e-5 — the primitives, `relu`/`sigmoid`, and the
full softmax+NLL composition (which runs every logit through a shared denominator
and stress-tests accumulation).

## The demo: three spirals

[`notebooks/spirals_demo.ipynb`](https://github.com/diego-magana/micrograd/blob/main/notebooks/spirals_demo.ipynb) trains a classifier on three interleaved spirals —
a 2D, three-class problem I generate directly (there's no `make_spirals` in
scikit-learn), chosen because a spiral forces a curved boundary that a linear
model provably cannot draw. The notebook's arc is deliberate: prove the task is
nonlinear, then solve it, then check it generalizes.

| Model | Params | Test accuracy |
|---|---|---|
| Linear softmax (`MLP(2, [3])`) | 9 | **0.27** — test NLL **1.14** vs $\ln 3 = 1.10$ |
| `MLP(2, [16, 16, 3])`, tanh hidden + softmax | 371 | **0.87** (train 0.99) |

The linear baseline converges to a held-out NLL of 1.14 against the 1.10 of
guessing uniformly — it learns a little on train (NLL 1.00) and none of it
transfers. It is not undertrained: the loss is unchanged from step 80 to step
3,000 and the gradient norm reaches 1e-16. 

[![Decision boundary](assets/decision_boundary.png)](https://github.com/diego-magana/micrograd/blob/main/notebooks/spirals_demo.ipynb)

## Run it

```bash
pip install -e .                              # the package itself needs nothing else
pip install -r requirements.txt               # deps for the tests + demo
pytest                                        # verify gradients against PyTorch
jupyter notebook notebooks/spirals_demo.ipynb # the full training demo
```

## Notes

**The whole engine is six gradient rules — plus two.** `+`, `*`, `**`, `tanh`,
`exp`, `log` are the six everything else composes from. `sigmoid` is a
convenience the six can already express as `1/(1+exp(-x))`, implemented directly
so large negative inputs don't overflow `exp`. `relu` is not — it needs a
comparison rather than an arithmetic operation, so it genuinely extends what the
engine can express. Eight closures in `engine.py`, six of them load-bearing. The
surface is rich, the bookkeeping is tiny — that concentration is the point of
the design.

**`tanh` and `sigmoid` are numerically stable.** `tanh` uses `math.tanh` rather
than the algebraic `(e^{2x}−1)/(e^{2x}+1)` form, which overflows for large inputs;
`sigmoid` branches on the sign so large-magnitude inputs don't overflow `exp`.
Small things, but an engine whose thesis is correctness shouldn't crash on a
large activation.

**It's built for understanding, not scale.** One `Value` per number and a Python
loop per forward pass make this O(samples × params) with no vectorization — fine
for a hundred 2D points, hopeless for a real dataset. That's the deliberate
tradeoff: you can read every gradient. Scale is what the next repos add.

## Attribution

The engine and MLP design follow Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd). What I added: `relu`/`sigmoid`
with stability fixes, gradient verification against PyTorch (including softmax+NLL),
the original three-spiral dataset and multi-class demo with a held-out split, and
production packaging (tests, CI, docs).
