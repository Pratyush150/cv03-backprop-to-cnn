"""Example 02 -- stacked linear layers collapse, and XOR is where you watch it.

Run:  python3 examples/02_why_nonlinearity.py

The claim in every textbook is that two linear layers with nothing between them
are algebraically one linear layer:

    W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)

This script does three things with that claim. It CONSTRUCTS the collapsed
single layer and checks it is numerically identical to the stack. It then
trains the stack on XOR and watches it fail -- not fail to converge, converge
to the best line, which gets two of the four corners right and floors the mean
squared error at exactly 0.25. Then it puts one ReLU between the same two
layers, changes nothing else, and watches the same optimiser drive the loss to
zero.

The failure is the lesson. A reader who has seen the loss curve flatten at 0.25
and knows why never has to take "we need nonlinearity" on trust again.
"""

from __future__ import annotations

import numpy as np

from _common import figure, rule, save

from netfs import Linear, ReLU, SGD, Sequential, mse_loss, xor_dataset

STEPS, LR = 6000, 0.05


def build(with_relu: bool, hidden: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    layers = [Linear(2, hidden, rng=rng)]
    if with_relu:
        layers.append(ReLU())
    layers.append(Linear(hidden, 1, rng=rng))
    return Sequential(*layers)


def fit(model, x, y, steps=STEPS, lr=LR):
    opt = SGD(model, lr=lr)
    history = []
    for _ in range(steps):
        history.append(model.backward_from_loss(mse_loss, x, y))
        opt.step()
    return history


def main() -> None:
    x, y = xor_dataset()

    rule("1. the collapse is an equality, not an approximation")
    rng = np.random.default_rng(1)
    a, b = Linear(2, 5, rng=rng), Linear(5, 3, rng=rng)
    stacked = Sequential(a, b)
    collapsed = Linear(2, 3, rng=rng)
    collapsed.params["W"][:] = b.params["W"] @ a.params["W"]        # W_eff = W2 W1
    collapsed.params["b"][:] = b.params["W"] @ a.params["b"] + b.params["b"]
    probe = rng.standard_normal((7, 2))
    print(f"  two layers (2->5->3) vs the single 2->3 layer they equal:")
    print(f"  max absolute difference over 7 random inputs: "
          f"{np.abs(stacked.forward(probe) - collapsed.forward(probe)).max():.2e}")
    n_collapsed = sum(v.size for v in collapsed.params.values())
    print(f"  the stack holds {stacked.n_params()} parameters; the single layer it is equal to "
          f"holds {n_collapsed}.")
    print("  Depth without a bend buys nothing at all.")

    rule("2. the hand-set XOR network -- no training at all")
    # From the paper drill: two ReLU units, weights chosen by hand.
    w1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    w2 = np.array([[1.0, -2.0]])
    b2 = np.array([0.0])
    h = np.maximum(x @ w1.T + b1, 0.0)
    print(f"  with ReLU   : {(h @ w2.T + b2).ravel()}   <- exactly XOR")
    w_eff, b_eff = w2 @ w1, w2 @ b1 + b2
    print(f"  ReLU deleted: {(x @ w_eff.T + b_eff).ravel()}   <- y = -x1 - x2 + 2")
    print(f"  collapsed layer: W_eff = {w_eff.ravel()}, b_eff = {b_eff}")

    rule("3. train both, same data, same optimiser, same learning rate")
    model_lin, model_relu = build(False), build(True)
    hist_lin = fit(model_lin, x, y)
    hist_relu = fit(model_relu, x, y)
    print(f"  linear stack final loss : {hist_lin[-1]:.6f}")
    print(f"  with one ReLU           : {hist_relu[-1]:.6f}")
    print(f"  linear predictions : {np.round(model_lin.forward(x).ravel(), 3)}")
    print(f"  ReLU predictions   : {np.round(model_relu.forward(x).ravel(), 3)}")
    # Why exactly 0.25, and not "some small number it happened to reach":
    # every linear f satisfies f(0,0) + f(1,1) = f(0,1) + f(1,0). XOR's targets
    # give 0 on the left and 2 on the right, so a gap of 2 must be split over
    # four points -- an error of 1/2 each, and mean(4 * 0.25) = 0.25.
    print("  0.25 is the algebraic floor for ANY linear model on XOR, not a tuning failure.")

    # ------------------------------------------------------------------ figure
    g = np.linspace(-0.4, 1.4, 220)
    gx, gy = np.meshgrid(g, g)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    fig, ax = figure(1, 3, figsize=(12, 3.8))
    for a_, model, title in ((ax[0], model_lin, "no activation: the best fit is a constant"),
                             (ax[1], model_relu, "with ReLU: a region, not a half-plane")):
        z = model.forward(grid).reshape(gx.shape)
        a_.grid(False)
        im = a_.pcolormesh(gx, gy, z, cmap="coolwarm", vmin=-0.6, vmax=1.6, shading="auto")
        # Guard the contour: the trained linear model outputs 0.5 EVERYWHERE, so
        # there is no 0.5 level set to draw and matplotlib would render the
        # floating-point speckle of a flat surface as if it were a boundary.
        if float(z.max() - z.min()) > 1e-6:
            a_.contour(gx, gy, z, levels=[0.5], colors="k", linewidths=1.4)
        else:
            a_.text(0.5, 0.62, f"output = {z.mean():.3f} everywhere\nMSE floors at 0.25",
                    ha="center", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="#90a4ae"))
        a_.scatter([0, 1], [1, 0], marker="o", s=90, facecolor="white", edgecolor="k",
                   zorder=4, label="target 1")
        a_.scatter([0, 1], [0, 1], marker="X", s=90, facecolor="k", edgecolor="k",
                   zorder=4, label="target 0")
        a_.set_title(title)
        a_.set_xlabel("x1")
        a_.set_ylabel("x2")
        a_.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=a_, fraction=0.046)
    ax[2].semilogy(hist_lin, color="#e57373", lw=1.6, label="linear stack")
    ax[2].semilogy(hist_relu, color="#1565c0", lw=1.6, label="one ReLU inserted")
    ax[2].axhline(0.25, color="#9e9e9e", ls="--", lw=1.0, label="0.25 = best any line can do")
    ax[2].set_title("same optimiser, same data")
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("mean squared error")
    ax[2].legend(fontsize=7)
    save(fig, "02-xor-collapse.png")


if __name__ == "__main__":
    main()
