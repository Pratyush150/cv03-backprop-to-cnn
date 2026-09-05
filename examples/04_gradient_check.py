"""Example 04 -- gradient checking, and the bugs it catches.

Run:  python3 examples/04_gradient_check.py

If you write a backward pass by hand you will get one wrong, and it will not
raise. The network still runs, the loss still goes down a bit -- a wrong
direction that partly correlates with the right one still helps for a while --
and you spend two days on the learning rate. Gradient checking is the only
thing that turns that into a five-second answer.

This script:
  1. checks every layer in netfs and prints the achieved relative error;
  2. sweeps the step size h across nine orders of magnitude and shows the
     U-curve -- truncation error on one side, floating-point cancellation on
     the other -- so the "use 1e-5, and stop shrinking if it gets worse" rule
     is something you have seen rather than been told;
  3. SABOTAGES three backward passes with the three most common real bugs and
     confirms the check catches all three, with the error each one produces.

Point 3 is the one that matters. A test that has never failed is a test whose
sensitivity is unknown.
"""

from __future__ import annotations

import numpy as np

from _common import figure, rule, save

from netfs import (Conv2D, Flatten, GlobalAvgPool2D, Linear, MaxPool2D, ReLU, Sequential,
                   Sigmoid, Tanh, check_layer, numerical_gradient, relative_error,
                   softmax_cross_entropy)


def rng(seed):
    return np.random.default_rng(seed)


class LinearMissingBatchSum(Linear):
    """BUG 1 -- the bias gradient without its sum over the batch.

    `db = dZ[0]` instead of `db = dZ.sum(axis=0)`. The same bias is added to
    every row of the batch, so it is a multi-path variable and the paths add;
    taking one row keeps the shape legal and throws away N-1 of the N
    contributions. Nothing raises.
    """

    def backward(self, dout):
        self._store("W", dout.T @ self.x)
        self._store("b", dout[0].copy())          # <-- the bug
        return dout @ self.params["W"]


class LinearTransposedWeightGrad(Linear):
    """BUG 2 -- dW transposed.

    On a non-square layer this raises a clean shape error, which is the
    friendly version of the failure. On a SQUARE layer the transpose has
    exactly the right shape, the assertion in Layer._store passes, and the only
    thing that can tell you the gradient is wrong is a number. That is the case
    reproduced here.
    """

    def backward(self, dout):
        self._store("W", (dout.T @ self.x).T)     # <-- the bug (square layer only)
        self._store("b", dout.sum(axis=0))
        return dout @ self.params["W"]


class ReLUIncludingZero(ReLU):
    """BUG 3 -- a ReLU whose backward pass uses `>= 0` instead of `> 0`.

    Almost invisible: it differs from the correct version only where the
    pre-activation is exactly 0, which for continuous data essentially never
    happens... until you feed it data that has been through another ReLU, or
    zero padding, or a masked region, at which point a large fraction of the
    entries are exactly 0 and the gradient leaks through units that were off.
    """

    def forward(self, x):
        self.z = x
        return np.maximum(x, 0.0)

    def backward(self, dout):
        return dout * (self.z >= 0)               # <-- the bug


def main() -> None:
    rule("1. every layer in the package, analytic vs central difference")
    checks = {
        "Linear(4->3)": (Linear(4, 3, rng=rng(1)), rng(2).standard_normal((5, 4))),
        "ReLU": (ReLU(), rng(3).standard_normal((4, 5)) + 1.5),
        "Sigmoid": (Sigmoid(), rng(4).standard_normal((4, 5))),
        "Tanh": (Tanh(), rng(5).standard_normal((4, 5))),
        "Flatten": (Flatten(), rng(6).standard_normal((2, 3, 4, 4))),
        "Conv2D(2->3, k3, p1)": (Conv2D(2, 3, 3, pad=1, rng=rng(7)),
                                 rng(8).standard_normal((2, 2, 6, 6))),
        "Conv2D(2->3, k3, s2)": (Conv2D(2, 3, 3, stride=2, rng=rng(9)),
                                 rng(10).standard_normal((2, 2, 7, 7))),
        "MaxPool2D(2, 2)": (MaxPool2D(2, 2), rng(11).standard_normal((2, 3, 6, 6))),
        "GlobalAvgPool2D": (GlobalAvgPool2D(), rng(12).standard_normal((2, 3, 5, 5))),
    }
    worst = {}
    print(f"  {'layer':<24}{'dW':>12}{'db':>12}{'dinput':>12}")
    for name, (layer, x) in checks.items():
        e = check_layer(layer, x)
        worst[name] = max(e.values())
        fmt = lambda k: f"{e[k]:.2e}" if k in e else "-"      # noqa: E731
        print(f"  {name:<24}{fmt('W'):>12}{fmt('b'):>12}{fmt('input'):>12}")
    print(f"  worst relative error anywhere: {max(worst.values()):.2e}")

    rule("2. the step size h -- why 1e-5 and not 1e-12")
    logits = rng(20).standard_normal((4, 5))
    y = np.array([0, 3, 1, 4])
    _, analytic = softmax_cross_entropy(logits, y)
    hs = np.logspace(-1, -13, 25)
    errs = []
    for h in hs:
        num = numerical_gradient(lambda: softmax_cross_entropy(logits, y)[0], logits, h=h)
        errs.append(relative_error(analytic, num))
    best = int(np.argmin(errs))
    print(f"  best h = {hs[best]:.1e}, relative error {errs[best]:.2e}")
    for h, e in zip(hs[::4], errs[::4]):
        print(f"    h = {h:8.1e}   rel err = {e:.2e}")
    print("  Left of the minimum the O(h^2) truncation error dominates; right of it the")
    print("  subtraction (L+ - L-) has cancelled away the mantissa. If your check gets")
    print("  WORSE as you shrink h, you are on the right-hand branch. Stop shrinking.")

    rule("3. three sabotaged backward passes, and what the check says")
    sabotage_results = []
    x = rng(30).standard_normal((6, 4))
    good = check_layer(Linear(4, 3, rng=rng(31)), x)
    sabotage_results.append(("correct Linear", max(good.values())))

    bad1 = check_layer(LinearMissingBatchSum(4, 3, rng=rng(31)), x)
    sabotage_results.append(("db without sum(axis=0)", bad1["b"]))

    xs = rng(32).standard_normal((6, 4))
    bad2 = check_layer(LinearTransposedWeightGrad(4, 4, rng=rng(33)), xs)
    sabotage_results.append(("dW transposed", bad2["W"]))

    # Feed the broken ReLU data with exact zeros in it -- the situation where
    # `>= 0` and `> 0` actually differ. Zero-padded feature maps and the output
    # of a previous ReLU are both full of exact zeros, so this is not contrived.
    z = rng(34).standard_normal((4, 6))
    z[z < 0] = 0.0
    bad3 = check_layer(ReLUIncludingZero(), z)
    sabotage_results.append(("ReLU'(0) = 1 instead of 0", bad3["input"]))

    for name, err in sabotage_results:
        verdict = "PASS" if err < 1e-7 else "CAUGHT"
        print(f"  {name:<32}{err:>10.2e}   {verdict}")
    print("  Note the size of the numbers. A broken gradient is not off by 1e-5; it is off")
    print("  by a factor of one, which is why a threshold anywhere in the range 1e-7 to")
    print("  1e-4 separates the two cases with a margin of several orders of magnitude.")

    # ------------------------------------------------------------------ figure
    fig, ax = figure(1, 3, figsize=(12.5, 3.8))
    names = list(worst)
    ax[0].barh(range(len(names)), list(worst.values()), color="#1565c0", height=0.6)
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=7)
    ax[0].set_xscale("log")
    ax[0].axvline(1e-7, color="#e57373", ls="--", lw=1.2, label="1e-7: 'be happy'")
    ax[0].set_xlabel("worst relative error (log)")
    ax[0].set_title("every layer, analytic vs numerical")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].invert_yaxis()

    ax[1].loglog(hs, errs, "o-", color="#1565c0", ms=3.5, lw=1.4)
    ax[1].axvline(hs[best], color="#43a047", ls=":", lw=1.2, label=f"best h = {hs[best]:.0e}")
    ax[1].axhline(1e-7, color="#e57373", ls="--", lw=1.2)
    ax[1].annotate("truncation\nerror O(h^2)", xy=(3e-2, 3e-4), fontsize=7, ha="center")
    ax[1].annotate("floating-point\ncancellation", xy=(3e-12, 3e-4), fontsize=7, ha="center")
    ax[1].set_xlabel("step size h")
    ax[1].set_ylabel("relative error")
    ax[1].set_title("h too small is worse than h too large")
    ax[1].legend(fontsize=7)

    labels = [n for n, _ in sabotage_results]
    values = [max(v, 1e-16) for _, v in sabotage_results]
    colours = ["#43a047"] + ["#e57373"] * (len(values) - 1)
    ax[2].barh(range(len(values)), values, color=colours, height=0.6)
    ax[2].set_yticks(range(len(values)))
    ax[2].set_yticklabels(labels, fontsize=7)
    ax[2].set_xscale("log")
    ax[2].axvline(1e-7, color="#37474f", ls="--", lw=1.2, label="check threshold")
    ax[2].set_xlabel("relative error (log)")
    ax[2].set_title("three real bugs, deliberately introduced")
    ax[2].legend(fontsize=7, loc="lower right")
    ax[2].invert_yaxis()
    save(fig, "04-gradient-check.png")


if __name__ == "__main__":
    main()
