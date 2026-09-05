"""Losses, and the numerical-stability story that fuses two of them.

Each loss returns `(loss, dpred)`: the scalar, and the gradient of that scalar
with respect to the thing the network produced. That pair is the seed of the
whole backward pass -- everything upstream is this one array being routed and
rescaled by the layers.

All derivations are in docs/DERIVATIONS.md sections 2, 6 and 7.
"""

from __future__ import annotations

import numpy as np


def mse_loss(pred: np.ndarray, target: np.ndarray, reduction: str = "mean"):
    """L = mean((pred - target)^2), and dL/dpred = 2*(pred - target)/N.

    The factor of 2 is real and people drop it. It does not change where the
    minimum is, so a model still trains with it missing -- it just trains at
    half the learning rate you think you set, which is precisely the kind of
    bug that is invisible until you compare against a reference.

    `reduction` changes the gradient by a factor of N, not just the printed
    loss. 'sum' and 'mean' are both correct and are not interchangeable: swap
    a mean loss for a sum loss at batch size 64 and every gradient is 64x
    bigger, which usually shows up as a run that diverges the moment you change
    the batch size.
    """
    diff = pred - target
    n = diff.shape[0] if reduction == "mean" else 1
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum' (got {reduction!r})")
    loss = float((diff ** 2).sum()) / n
    return loss, 2.0 * diff / n


def log_sum_exp(z: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """log(sum(exp(z))) computed without ever calling exp on a large number.

        log sum_j exp(z_j) = m + log sum_j exp(z_j - m),   m = max_j z_j

    This is an exact algebraic identity, not an approximation: factor exp(m)
    out of the sum and take its log. What it buys is that every exponent fed to
    exp() is now <= 0, so exp() lands in (0, 1] and cannot overflow. The
    largest term is exactly exp(0) = 1, so the sum is at least 1 and the log is
    finite.

    Without it, exp(1000) is inf in float64 (the limit is about 709.78) and
    inf/inf is nan. With it, logits of [1000, 1001, 1002] are handled with the
    same code path as [0, 1, 2].
    """
    m = np.max(z, axis=axis, keepdims=True)
    s = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return s if keepdims else np.squeeze(s, axis=axis)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax: subtract the row max before exponentiating.

    Subtracting a constant c from every logit multiplies numerator and
    denominator by exp(-c), so the result is unchanged for ANY c. Choosing
    c = max makes the largest exponent exactly 0. Underflow of the small terms
    to 0.0 is harmless here -- they were negligible -- while overflow of the
    large terms to inf is fatal.
    """
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_naive(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """The textbook formula, written out so its failure can be shown rather
    than described. exp(z)/sum(exp(z)) with no max subtraction.

    Kept in the library on purpose: examples/06_softmax_stability.py runs it on
    logits of 1000 and prints [nan nan nan], and tests/test_losses.py asserts
    both that it agrees with the stable version on safe inputs and that it
    fails on unsafe ones. A stability claim nobody has watched break is a
    slogan.
    """
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray, reduction: str = "mean"):
    """FUSED softmax + cross-entropy, the only form worth shipping.

    Forward, per sample:  L = -log p_y = logsumexp(z) - z_y

    That right-hand side never forms a probability at all. The unfused route --
    compute p = softmax(z), then -log(p[y]) -- has two failure modes that this
    one does not:

      1. Overflow. p = exp(z)/sum(exp(z)) is nan for z around 1000.
      2. Underflow. If the correct class is very wrong, p_y underflows to
         exactly 0.0 and -log(0.0) is +inf, which then poisons every parameter
         in the network on the next update. Nothing raises; the loss just
         prints `inf` and the run is dead.

    `logsumexp(z) - z_y` is finite for any finite input, because logsumexp is
    computed with the max shifted out and z_y is just a number.

    Backward:  dL/dz = (p - onehot(y)) / N

    That is the whole reason the fusion is universal. Differentiating the two
    stages separately gives you a CxC softmax Jacobian per sample, which then
    multiplies the cross-entropy gradient 1/p_y -- and the p_y in the
    denominator is exactly the quantity that just underflowed to zero. Composed
    algebraically first, the p_y cancels and what is left is a subtraction that
    cannot blow up: p is in [0, 1], the one-hot is in {0, 1}, so every entry of
    the gradient is in [-1, 1]. The derivation is in DERIVATIONS.md section 6.

    Free correctness check, valid for every input: each row of the gradient
    sums to exactly zero, because p sums to 1 and the one-hot sums to 1.
    Probability mass is conserved, so the gradient can only MOVE mass between
    classes, never create it. If your softmax gradient rows do not sum to zero,
    stop and look at the code rather than the learning rate.

    logits: (N, C) raw scores. y: (N,) integer class indices -- indices, not
    one-hot, and not probabilities.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum' (got {reduction!r})")
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)
    n = logits.shape[0]
    rows = np.arange(n)
    lse = log_sum_exp(logits, axis=1)                 # (N,)
    per_sample = lse - logits[rows, y]                # -log p_y, without forming p
    scale = 1.0 / n if reduction == "mean" else 1.0
    loss = float(per_sample.sum()) * scale

    p = softmax(logits, axis=1)
    dlogits = p.copy()
    dlogits[rows, y] -= 1.0                           # p - onehot
    return loss, dlogits * scale


def cross_entropy_unfused(logits: np.ndarray, y: np.ndarray, reduction: str = "mean"):
    """The two-stage version: form probabilities, then take -log of one of them.

    Mathematically identical to `softmax_cross_entropy` and numerically not.
    It exists so the failure can be demonstrated side by side (example 06 and
    tests/test_losses.py), and so the claim "they are fused for numerical
    stability" is backed by a number instead of being repeated.
    """
    p = softmax_naive(logits, axis=1)
    n = logits.shape[0]
    picked = p[np.arange(n), np.asarray(y)]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_sample = -np.log(picked)
    scale = 1.0 / n if reduction == "mean" else 1.0
    return float(per_sample.sum()) * scale


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    """Fraction correct. argmax over logits, no softmax needed -- softmax is
    monotonic, so it cannot change which class is largest. Applying it before
    argmax is a very common wasted exp() and a hint that the author is not sure
    what the layer does.
    """
    return float((np.argmax(logits, axis=1) == np.asarray(y)).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are truth, columns are the prediction. cm[i, j] counts samples of
    class i that the model called class j, so the diagonal is correct and every
    off-diagonal entry is one specific, nameable mistake.

    `np.add.at` and not `cm[i, j] += 1`: with repeated index pairs -- which is
    the entire point of a histogram -- buffered fancy-index assignment applies
    only ONE of the duplicate updates and silently drops the rest. The same
    trap appears in col2im in netfs.conv, for the same reason.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm
