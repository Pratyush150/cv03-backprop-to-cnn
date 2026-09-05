"""Gradient checking: the single most important test in this repository.

If you hand-write a backward pass, you WILL get it wrong. Not "might" -- a
transposed matmul, a missing sum over the batch axis, an `=` where `+=` was
needed. And the failure is silent: a network with a subtly wrong gradient still
runs, still prints a loss, and still goes down a bit, because a wrong direction
that happens to correlate with the right one still reduces the loss for a
while. You lose days blaming the learning rate.

The check is this. The derivative is defined as a limit of a difference
quotient, so approximate the limit numerically:

    numeric  = (L(theta + h) - L(theta - h)) / (2h)
    analytic = whatever your backward() returned
    agree?

Two things about that formula, both of which matter:

  * It is CENTRED. The one-sided (L(theta+h) - L(theta))/h has error O(h);
    the centred one has error O(h^2), because the h^1 terms of the two Taylor
    expansions cancel. At h = 1e-5 that is the difference between 5 correct
    digits and 10.
  * h must not be too small. Shrinking h shrinks the truncation error and
    grows the floating-point cancellation error -- L(theta+h) and L(theta-h)
    agree in their leading digits, and subtracting them throws those digits
    away. In float64 the sweet spot is h between 1e-6 and 1e-4; at 1e-10 there
    is nothing left. If your relative error GROWS as you shrink h, stop
    shrinking. examples/05_gradient_check.py plots that U-curve.

And do it in float64. In float32 the cancellation above eats so much of the
mantissa that a flawless implementation reports relative errors around 1e-3.
NumPy defaults to float64, which is one of the reasons this package is written
in NumPy and not in a framework whose default dtype is float32.
"""

from __future__ import annotations

import numpy as np


def relative_error(a: np.ndarray, b: np.ndarray, *, elementwise: bool = False) -> float:
    """Disagreement between two gradient arrays, as a single number.

    Default (array scale):   max|a - b| / max(max|a|, max|b|)
    With elementwise=True:   max over entries of  |a - b| / max(|a|, |b|)

    Relative and not absolute, because the same absolute error means completely
    different things next to a gradient of 1e-8 and a gradient of 1e4. The
    denominator is the max of the two magnitudes rather than one of them, so
    the measure is symmetric and does not blow up when the reference happens to
    be the small one.

    Why the default divides by the scale of the whole ARRAY and not by each
    entry, which is the more commonly quoted formula: a central difference has
    a noise floor of roughly 1e-11 in float64 no matter what it is measuring.
    In a softmax gradient whose largest entry is 0.16, an entry whose true
    value is 3e-06 is therefore known to about four digits and no more, and its
    per-entry relative error reads 3e-06 -- a number that describes the
    resolution of the measuring instrument, not the correctness of the code. A
    real backward-pass bug (a transpose, a missing sum, an `=` where `+=`
    belonged) is never confined below the noise floor of the largest entry; it
    moves entries that matter. Dividing by the array's own scale asks the
    question you actually mean: "is this gradient wrong by an amount that could
    affect training?"

    The honest cost: this summary can hide an error that lives only in entries
    far smaller than the biggest one. That is why it is not the only defence --
    every backward pass here also asserts its output shape, and
    tests/test_torch_oracle.py compares against an independent autograd engine.
    Pass elementwise=True when the entries are of comparable size and you want
    the stricter reading.

    Where both inputs are exactly zero the entry is agreement, not 0/0 = nan,
    and that case is real and routine: a ReLU that was off contributes exactly
    zero to both the analytic and the numeric gradient, because both are
    measuring the same dead switch.

    Rules of thumb worth memorising (they are CS231n's, and they are about
    right): below 1e-7 you should be happy; 1e-4 is only acceptable for an
    objective with kinks in it, like anything containing a ReLU; above 1e-2 is
    a bug.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    num = np.abs(a - b)
    if elementwise:
        den = np.maximum(np.abs(a), np.abs(b))
        nonzero = den > 0
        if not np.any(nonzero):
            return 0.0
        return float((num[nonzero] / den[nonzero]).max())
    scale = max(float(np.abs(a).max(initial=0.0)), float(np.abs(b).max(initial=0.0)))
    if scale == 0.0:
        return 0.0
    return float(num.max() / scale)


def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of scalar `f()` with respect to array `x`,
    which `f` reads through the array object itself.

    `x` is perturbed IN PLACE and restored, entry by entry. In place because
    the caller's model holds a reference to this exact array: rebinding a copy
    would compute the gradient of a network nobody is using. Restored because
    the entries after this one are perturbed from the same base point -- forget
    the restore and every row after the first is measured at the wrong place,
    which produces a plausible-looking table of subtly wrong numbers.

    Cost: two full forward passes per parameter. That is why this is a test and
    not a training method -- a hundred-million-parameter network would need two
    hundred million forward passes per step. Backprop gets all of them in one
    backward sweep costing about as much as one forward pass. Gradient checking
    is how you earn the right to trust that trade.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h
        lp = f()
        x[idx] = old - h
        lm = f()
        x[idx] = old                      # restore before moving on. Always.
        grad[idx] = (lp - lm) / (2.0 * h)
        it.iternext()
    return grad


def check_layer(layer, x: np.ndarray, *, h: float = 1e-5, seed: int = 0,
                include_input: bool = True) -> dict[str, float]:
    """Gradient-check one layer against a random scalar objective.

    A layer's forward pass returns an array, not a scalar, and a gradient is
    only defined for a scalar. So we invent one: L = sum(out * G) for a fixed
    random G. Then dL/dout is exactly G, which is what we hand to backward().
    Random rather than all-ones on purpose -- an all-ones seed cannot detect a
    backward pass that permutes or transposes its output, because summing is
    invariant to both. This is a real bug class in conv layers, where the
    forward pass ends in reshape+transpose and the backward pass has to undo
    them in the opposite order.

    Returns a dict of max relative errors: one entry per parameter, plus
    "input" for dL/dx.
    """
    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=np.float64)      # our own copy: we are about to perturb it
    out = layer.forward(x)
    g = rng.standard_normal(out.shape)

    def objective() -> float:
        return float((layer.forward(x) * g).sum())

    dx = layer.backward(g)
    errors: dict[str, float] = {}
    for name, p in layer.params.items():
        errors[name] = relative_error(layer.grads[name], numerical_gradient(objective, p, h))
    if include_input:
        errors["input"] = relative_error(dx, numerical_gradient(objective, x, h))
    # Re-run the forward pass so the layer's cache matches its parameters
    # again. Without this the layer is left holding the cache of the last
    # perturbed forward, which is a nasty surprise for anything that reuses it.
    layer.forward(x)
    return errors


def check_model(model, loss_fn, x: np.ndarray, y, *, h: float = 1e-5) -> dict[str, float]:
    """Gradient-check a whole network against its real loss.

    Keyed by "<layer index>.<param name>", so a failure names the exact array
    that is wrong. That specificity is worth having: the most common
    hand-backprop bug -- a multi-path gradient that was assigned instead of
    accumulated -- shows up as exactly ONE failing row while every other row
    passes perfectly. A single failing row is a signature, not noise.
    """
    x = np.array(x, dtype=np.float64)

    def objective() -> float:
        return loss_fn(model.forward(x), y)[0]

    model.backward_from_loss(loss_fn, x, y)
    errors: dict[str, float] = {}
    for i, layer in enumerate(model.layers):
        for name, p in layer.params.items():
            errors[f"{i}.{name}"] = relative_error(layer.grads[name],
                                                   numerical_gradient(objective, p, h))
    return errors
