"""Reproduce the loss floor: why a tanh output head caps confidence.

The current `MLP` hardcodes a linear output layer (`nonlin=False` on the last
layer), which is the fix — so the broken architecture can't be built through it.
This script hand-assembles the original V1 network from `Layer` to make the
finding checkable rather than just described.

The claim: with a tanh output feeding a sigmoid, the predicted probability is
bounded by sigma(1) ~ 0.731, so the best achievable per-sample BCE is
-ln(0.731) ~ 0.313. Over 100 samples that is a floor of ~31.33 that no amount
of training can cross, even at 100% accuracy.

    python scripts/loss_floor.py

No scikit-learn: the two-moons dataset is generated directly, the same way the
spiral demo generates its own data.
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from micrograd.engine import Value
from micrograd.nn import Layer

EPOCHS = 100
LR = 0.01
N = 100


def make_moons(n=N, noise=0.1, seed=0):
    """Two interleaving half-circles. Equivalent to sklearn's make_moons."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        t = math.pi * i / (n // 2 - 1) if i < n // 2 else math.pi * (i - n // 2) / (n // 2 - 1)
        if i < n // 2:
            px, py, label = math.cos(t), math.sin(t), 0
        else:
            px, py, label = 1 - math.cos(t), 0.5 - math.sin(t), 1
        X.append([px + rng.gauss(0, noise), py + rng.gauss(0, noise)])
        y.append(label)
    return X, y


def standardize(X):
    """Zero mean, unit variance per feature — keeps tanh out of saturation."""
    cols = list(zip(*X))
    stats = []
    for c in cols:
        mu = sum(c) / len(c)
        sd = math.sqrt(sum((v - mu) ** 2 for v in c) / len(c)) or 1.0
        stats.append((mu, sd))
    return [[(v - mu) / sd for v, (mu, sd) in zip(row, stats)] for row in X]


def build_v1_network(seed=1337):
    """MLP(2, [16, 16, 1]) with a *tanh* output — the architecture with the bug.

    `MLP` won't build this: it forces the head linear. So the layers are stacked
    by hand with nonlin=True all the way through, including the output.
    """
    random.seed(seed)
    return [Layer(2, 16, nonlin=True),
            Layer(16, 16, nonlin=True),
            Layer(16, 1, nonlin=True)]


def forward(layers, x):
    out = [Value(v) for v in x]
    for layer in layers:
        out = layer(out)
    return out


def bce(pred, target):
    """Binary cross-entropy on sigmoid(pred), built from engine primitives."""
    p = pred.sigmoid()
    return -(p.log() if target == 1 else (1 + (-p)).log())


def main():
    X, y = make_moons()
    X = standardize(X)
    layers = build_v1_network()
    params = [p for layer in layers for p in layer.parameters()]

    print(f"{len(params)} parameters, {len(X)} samples, tanh output head\n")
    print(f"{'epoch':>6}  {'summed BCE':>11}  {'accuracy':>9}")

    loss = None
    for epoch in range(EPOCHS):
        preds = [forward(layers, x) for x in X]
        loss = sum(bce(p, t) for p, t in zip(preds, y))

        for p in params:
            p.grad = 0.0
        loss.backward()
        for p in params:
            p.data -= LR * p.grad

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            correct = sum((p.data > 0) == (t == 1) for p, t in zip(preds, y))
            print(f"{epoch:>6}  {loss.data:>11.2f}  {correct:>6}/{len(X)}")

    floor = -math.log(1 / (1 + math.exp(-1))) * len(X)
    print()
    print(f"observed summed loss : {loss.data:.2f}")
    print(f"architectural floor  : {floor:.2f}   = -ln(sigma(1)) * {len(X)}")
    print(f"gap                  : {100 * (loss.data - floor) / floor:.1f}%")
    print()
    print("The tanh head bounds its output to (-1, 1), so p = sigma(z) < 0.731 and")
    print("the per-sample BCE cannot fall below -ln(0.731) ~ 0.313. The network")
    print("classifies correctly and is structurally forbidden from being confident.")


if __name__ == "__main__":
    main()
