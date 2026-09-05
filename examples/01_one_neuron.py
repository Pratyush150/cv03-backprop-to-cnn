"""Example 01 -- one neuron, one squared error, one gradient derived by hand.

Run:  python3 examples/01_one_neuron.py

The whole of deep learning is in this file, at a scale you can check with a
pencil. A single linear unit y_hat = w*x + b, a squared-error loss, two partial
derivatives worked out on paper (docs/DERIVATIONS.md section 2), and the update
rule w <- w - lr * dL/dw applied a few hundred times.

What it demonstrates:
  1. the hand-derived gradient agrees with a central-difference numerical
     gradient to about 1e-11, which is the first appearance of the check that
     the rest of this repository is built on;
  2. the loss falls to the noise floor of the data, and no further -- the model
     cannot beat the noise it was given, and the flat tail of the loss curve is
     that limit, not a bug;
  3. the general Linear layer in netfs, given one input and one output, is
     numerically identical to the two-line hand version. That is the "then a
     layer" step: the layer is not a new idea, it is the same arithmetic with a
     batch axis and a matrix.
"""

from __future__ import annotations

import numpy as np

from _common import figure, rule, save

from netfs import Linear, SGD, Sequential, mse_loss, numerical_gradient, relative_error

TRUE_W, TRUE_B, NOISE = 2.5, -1.3, 0.35


def make_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    y = TRUE_W * x + TRUE_B + NOISE * rng.standard_normal(n)
    return x, y


def loss_and_grad(w, b, x, y):
    """L = mean((w*x + b - y)^2), and the two derivatives, by hand.

        dL/dw = mean(2 * (w*x + b - y) * x)
        dL/db = mean(2 * (w*x + b - y))

    Read the chain rule in them: the residual r = w*x + b - y is the gradient
    of the loss with respect to the prediction (times 2), and each parameter
    then multiplies it by that parameter's own local derivative of the
    prediction -- x for the weight, 1 for the bias. Every backward pass in this
    package is that same two-step, with bigger arrays.
    """
    r = w * x + b - y
    return float((r ** 2).mean()), float((2 * r * x).mean()), float((2 * r).mean())


def main() -> None:
    x, y = make_data()

    rule("1. the hand-derived gradient, against a numerical one")
    w0, b0 = 0.0, 0.0
    _, dw, db = loss_and_grad(w0, b0, x, y)
    # Wiggle each parameter and measure. Two forward passes per parameter; with
    # nine parameters that is eighteen, with a hundred million it is impossible.
    # That impossibility is the entire reason backprop exists.
    p = np.array([w0, b0])
    num = numerical_gradient(lambda: loss_and_grad(p[0], p[1], x, y)[0], p)
    print(f"  analytic  dL/dw = {dw: .10f}   dL/db = {db: .10f}")
    print(f"  numerical dL/dw = {num[0]: .10f}   dL/db = {num[1]: .10f}")
    print(f"  relative error  = {relative_error(np.array([dw, db]), num):.2e}")

    rule("2. train it")
    lr, steps = 0.05, 400
    w, b = 0.0, 0.0
    history, path = [], []
    for step in range(steps):
        loss, dw, db = loss_and_grad(w, b, x, y)
        history.append(loss)
        path.append((w, b))
        w -= lr * dw          # THE update rule. Downhill, because the gradient
        b -= lr * db          # points uphill.
        if step in (0, 1, 2, 10, 100, steps - 1):
            print(f"  step {step:4d}  loss {loss:8.4f}   w {w:7.4f}  b {b:7.4f}")
    print(f"  recovered w = {w:.4f} (true {TRUE_W}),  b = {b:.4f} (true {TRUE_B})")
    # The loss cannot go below the variance of the noise we added: the model is
    # fitting a line to data that is not exactly on a line. Anyone who reports a
    # loss below this floor has leaked their targets.
    print(f"  final loss {history[-1]:.4f};  noise floor is about "
          f"NOISE^2 = {NOISE ** 2:.4f}")

    rule("3. the same thing as a netfs Linear layer")
    model = Sequential(Linear(1, 1, rng=np.random.default_rng(0)))
    model.layers[0].params["W"][:] = 0.0        # start from the same place
    model.layers[0].params["b"][:] = 0.0
    opt = SGD(model, lr=lr)
    xb, yb = x.reshape(-1, 1), y.reshape(-1, 1)
    layer_history = []
    for _ in range(steps):
        layer_history.append(model.backward_from_loss(mse_loss, xb, yb))
        opt.step()
    lw = float(model.layers[0].params["W"][0, 0])
    lb = float(model.layers[0].params["b"][0])
    print(f"  hand-written : w {w:.10f}  b {b:.10f}")
    print(f"  Linear layer : w {lw:.10f}  b {lb:.10f}")
    print(f"  max difference over the whole loss curve: "
          f"{np.abs(np.array(history) - np.array(layer_history)).max():.2e}")

    # ------------------------------------------------------------------ figure
    fig, ax = figure(1, 3, figsize=(12, 3.6))
    ax[0].scatter(x, y, s=14, color="#37474f", label="data", zorder=3)
    grid = np.array([-3.2, 3.2])
    for idx, colour in [(0, "#e57373"), (2, "#ffb74d"), (6, "#81c784"), (steps - 1, "#1565c0")]:
        pw, pb = path[idx]
        ax[0].plot(grid, pw * grid + pb, color=colour, lw=1.8, label=f"step {idx}")
    ax[0].set_title("one neuron fitting a line")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(fontsize=7)

    ax[1].semilogy(history, color="#1565c0", lw=1.6)
    ax[1].axhline(NOISE ** 2, color="#e57373", ls="--", lw=1.2,
                  label=f"noise floor {NOISE ** 2:.3f}")
    ax[1].set_title("loss (log scale)")
    ax[1].set_xlabel("gradient descent step")
    ax[1].set_ylabel("mean squared error")
    ax[1].legend(fontsize=7)

    # The loss surface of a linear model with squared error is a paraboloid --
    # one minimum, no local traps. That is why this problem is easy and a deep
    # network is not: depth plus a nonlinearity destroys the convexity.
    ws = np.linspace(-0.5, 3.5, 160)
    bs = np.linspace(-2.6, 0.8, 160)
    WW, BB = np.meshgrid(ws, bs)
    surface = ((WW[..., None] * x + BB[..., None] - y) ** 2).mean(axis=-1)
    cs = ax[2].contour(WW, BB, surface, levels=np.geomspace(0.13, 60, 14),
                       colors="#90a4ae", linewidths=0.8)
    ax[2].clabel(cs, fmt="%.1f", fontsize=6)
    pw, pb = np.array(path).T
    ax[2].plot(pw, pb, color="#1565c0", lw=1.6)
    ax[2].scatter([TRUE_W], [TRUE_B], marker="*", s=140, color="#e57373", zorder=4,
                  label="true (w, b)")
    ax[2].set_title("descent path on the loss surface")
    ax[2].set_xlabel("w")
    ax[2].set_ylabel("b")
    ax[2].legend(fontsize=7)
    save(fig, "01-one-neuron.png")


if __name__ == "__main__":
    main()
