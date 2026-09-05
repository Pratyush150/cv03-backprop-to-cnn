"""Example 03 -- backpropagation through a two-layer MLP, term by term.

Run:  python3 examples/03_backprop_mlp.py

Part 1 is a network small enough to check with a pencil: two inputs, two ReLU
hidden units, one linear output, mean squared error, one sample. Nine scalar
parameters, nine gradients. Every one of them is worked out symbolically in
docs/DERIVATIONS.md section 4 and printed here beside a central-difference
numerical gradient.

Two of those nine gradients are EXACTLY zero, and that is the most instructive
row in the table. The first hidden unit's pre-activation is negative, so its
ReLU is off, so it receives no gradient at all -- not a small one, zero. No
optimiser can move its incoming weights, so its pre-activation cannot change,
so it can never come back on. That is the dying-ReLU problem, visible in four
numbers.

Part 2 prints the shape of every intermediate array with the reason it has that
shape. Shape errors are the commonest bug in hand-written networks, and the
batched case is the dangerous one: a wrong shape in the single-sample case
raises immediately, while in the batched case NumPy broadcasts and hands you a
numerically wrong gradient with no exception at all.

Part 3 trains the thing on data no straight line can separate.
"""

from __future__ import annotations

import numpy as np

from _common import figure, rule, save

from netfs import (Adam, Linear, ReLU, Sequential, accuracy, check_model, numerical_gradient,
                   relative_error, softmax_cross_entropy, train, two_moons)

# The hand-worked example. Memorise it; it is small enough to reproduce cold.
X1 = np.array([[1.0, 2.0]])
Y1 = np.array([[1.0]])
W1 = np.array([[0.5, -1.0], [1.0, 1.0]])
B1 = np.array([0.0, -1.0])
W2 = np.array([[1.5, -0.5]])
B2 = np.array([0.5])


def main() -> None:
    rule("1. the forward pass, every number written out")
    z1 = X1 @ W1.T + B1
    h = np.maximum(z1, 0.0)
    z2 = h @ W2.T + B2
    loss = float(((z2 - Y1) ** 2).sum())
    print(f"  z1 = x @ W1.T + b1 = {z1[0]}      <- neuron 0's pre-activation is negative")
    print(f"  h  = relu(z1)      = {h[0]}      <- so neuron 0 is SWITCHED OFF")
    print(f"  z2 = h @ W2.T + b2 = {z2[0]}")
    print(f"  L  = (z2 - y)^2    = {loss}")
    print("  Everything on those four lines is cached. It is all needed again in a moment,")
    print("  and that is exactly why training uses more memory than inference.")

    rule("2. the backward pass, one node at a time")
    dz2 = 2.0 * (z2 - Y1)                    # dL/dz2, the seed
    dW2 = dz2.T @ h                          # multiply swaps: scaled by the OTHER input
    db2 = dz2.sum(axis=0)                    # add distributes, summed over the batch
    dh = dz2 @ W2                            # ...and back through the same multiply
    dz1 = dh * (z1 > 0)                      # max ROUTES: the dead unit gets exactly 0
    dW1 = dz1.T @ X1
    db1 = dz1.sum(axis=0)
    print(f"  dL/dz2 = 2*(z2 - y)        = {dz2[0]}    <- the seed; everything below is this")
    print(f"                                          number being routed and rescaled")
    print(f"  dL/dW2 = dz2.T @ h         = {dW2[0]}")
    print(f"  dL/db2 = dz2.sum(axis=0)   = {db2}")
    print(f"  dL/dh  = dz2 @ W2          = {dh[0]}")
    print(f"  dL/dz1 = dh * (z1 > 0)     = {dz1[0]}    <- entry 0 killed by the dead ReLU")
    print(f"  dL/dW1 = dz1.T @ x         = {dW1.tolist()}")
    print(f"  dL/db1 = dz1.sum(axis=0)   = {db1}")

    rule("3. all nine gradients, analytic against numerical")
    params = {"W1": W1.copy(), "b1": B1.copy(), "W2": W2.copy(), "b2": B2.copy()}
    analytic = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def forward_loss() -> float:
        zz1 = X1 @ params["W1"].T + params["b1"]
        hh = np.maximum(zz1, 0.0)
        zz2 = hh @ params["W2"].T + params["b2"]
        return float(((zz2 - Y1) ** 2).sum())

    print(f"  {'param':<6}{'index':<9}{'analytic':>11}{'numeric':>18}{'rel err':>13}")
    for name in ("W1", "b1", "W2", "b2"):
        assert analytic[name].shape == params[name].shape, name    # shape discipline, always
        num = numerical_gradient(forward_loss, params[name])
        for idx in np.ndindex(params[name].shape):
            a, n = analytic[name][idx] + 0.0, num[idx] + 0.0       # +0.0 turns -0.0 into 0.0
            den = max(abs(a), abs(n))
            tag = "exact-zero" if den == 0.0 else f"{abs(a - n) / den:.2e}"
            print(f"  {name:<6}{str(idx):<9}{a:>11.4f}{n:>18.10f}{tag:>13}")
    print("  The four exact zeros are not a failure of the check. Perturbing W1[0,0] by 1e-5")
    print("  moves z1[0] from -1.5 to -1.50001, which is still negative, so the ReLU stays")
    print("  off and the loss does not move. Both methods measure the same dead switch.")

    rule("4. the shape of every intermediate array, and why it has that shape")
    n, d_in, hidden, d_out = 8, 4, 6, 3
    rng = np.random.default_rng(0)
    model = Sequential(Linear(d_in, hidden, rng=rng), ReLU(), Linear(hidden, d_out, rng=rng))
    xb = rng.standard_normal((n, d_in))
    yb = rng.integers(0, d_out, size=n)
    rows = [
        ("X", (n, d_in), "one sample per ROW. Batch first, always, in this package."),
        ("W1", (hidden, d_in), "one NEURON per row. Forward is X @ W1.T, so the shared"),
        ("", None, "dimension d_in is what the transpose lines up."),
        ("Z1 = X @ W1.T + b1", (n, hidden), "b1 is (hidden,) and broadcasts down the batch."),
        ("H = relu(Z1)", (n, hidden), "elementwise: shape cannot change."),
        ("Z2 = H @ W2.T + b2", (n, d_out), "the logits."),
        ("dZ2", (n, d_out), "same shape as Z2: one derivative per entry."),
        ("dW2 = dZ2.T @ H", (d_out, hidden), "must equal W2's shape. There is exactly one way"),
        ("", None, "to multiply (n,d_out) and (n,hidden) into (d_out,hidden)."),
        ("db2 = dZ2.sum(0)", (d_out,), "SUM over the batch: one bias, added n times, so"),
        ("", None, "it is a multi-path variable and the paths add."),
        ("dH = dZ2 @ W2", (n, hidden), "back through the same multiply, other side."),
        ("dZ1 = dH * (Z1>0)", (n, hidden), "the ReLU switch, elementwise."),
    ]
    for name, shape, why in rows:
        shown = "" if shape is None else str(shape)
        print(f"  {name:<20}{shown:<12}{why}")
    # The table above is typed by hand, so check it against the arrays the
    # network actually produced. A shape table that has drifted out of date is
    # worse than no shape table.
    out = model.forward(xb)
    lin1, relu, lin2 = model.layers
    assert lin1.x.shape == (n, d_in)
    assert lin1.params["W"].shape == (hidden, d_in)
    assert relu.mask.shape == (n, hidden)
    assert out.shape == (n, d_out)
    errs = check_model(model, softmax_cross_entropy, xb, yb)
    assert lin2.grads["W"].shape == (d_out, hidden) and lin2.grads["b"].shape == (d_out,)
    print(f"  and the whole thing gradient-checks: worst relative error "
          f"{max(errs.values()):.2e}")

    rule("5. train it on data no straight line can separate")
    x, y = two_moons(600, noise=0.12, rng=np.random.default_rng(0))
    split = 450
    xtr, ytr, xte, yte = x[:split], y[:split], x[split:], y[split:]
    rng = np.random.default_rng(1)
    net = Sequential(Linear(2, 16, rng=rng), ReLU(), Linear(16, 2, rng=rng))
    print(f"  {net.n_params()} parameters, {len(xtr)} training points")
    # The number you write down BEFORE pressing go: an untrained two-class
    # classifier spreads its confidence evenly, so the loss must start near
    # -ln(1/2) = ln(2) = 0.6931. Far from it means the wiring is wrong, and it
    # is much cheaper to find that out now than after sixty epochs.
    init_loss, _ = softmax_cross_entropy(net.forward(xtr), ytr)
    print(f"  loss at init {init_loss:.4f}, expected about ln(2) = {np.log(2):.4f}")
    hist = train(net, softmax_cross_entropy, Adam(net, lr=0.02), xtr, ytr, xte, yte,
                 epochs=60, batch_size=32, verbose=False)
    print(f"  final train loss {hist['train_loss'][-1]:.4f}  "
          f"test accuracy {hist['val_acc'][-1]:.4f}")

    # A linear model on the same data, for scale.
    lin = Sequential(Linear(2, 2, rng=np.random.default_rng(2)))
    train(lin, softmax_cross_entropy, Adam(lin, lr=0.05), xtr, ytr, epochs=60, batch_size=32,
          verbose=False)
    lin_acc = accuracy(lin.forward(xte), yte)
    print(f"  a single linear layer on the same data: test accuracy {lin_acc:.4f}")

    # ------------------------------------------------------------------ figure
    g1 = np.linspace(x[:, 0].min() - 0.4, x[:, 0].max() + 0.4, 260)
    g2 = np.linspace(x[:, 1].min() - 0.4, x[:, 1].max() + 0.4, 260)
    gx, gy = np.meshgrid(g1, g2)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    fig, ax = figure(1, 3, figsize=(12, 3.8))
    for a_, m, title, acc in ((ax[0], lin, "one linear layer", lin_acc),
                              (ax[1], net, "2 -> 16 ReLU -> 2", hist["val_acc"][-1])):
        z = m.forward(grid)
        p = (z[:, 1] - z[:, 0]).reshape(gx.shape)
        a_.grid(False)
        a_.pcolormesh(gx, gy, p, cmap="coolwarm", shading="auto",
                      vmin=-np.abs(p).max(), vmax=np.abs(p).max())
        a_.contour(gx, gy, p, levels=[0.0], colors="k", linewidths=1.4)
        a_.scatter(*xte[yte == 0].T, s=12, c="#0d47a1", edgecolor="white", linewidth=0.3)
        a_.scatter(*xte[yte == 1].T, s=12, c="#b71c1c", edgecolor="white", linewidth=0.3)
        a_.set_title(f"{title}  --  test accuracy {acc:.3f}")
        a_.set_xlabel("x1")
        a_.set_ylabel("x2")
    ax[2].plot(hist["train_loss"], color="#1565c0", lw=1.6, label="train")
    ax[2].plot(hist["val_loss"], color="#e57373", lw=1.6, label="held out")
    ax[2].set_yscale("log")
    ax[2].set_title("MLP loss, 60 epochs")
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("cross-entropy")
    ax[2].legend(fontsize=7)
    save(fig, "03-backprop-mlp.png")


if __name__ == "__main__":
    main()
