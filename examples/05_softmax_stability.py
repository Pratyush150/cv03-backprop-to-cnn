"""Example 05 -- softmax, cross-entropy, and why they are fused.

Run:  python3 examples/05_softmax_stability.py

Every framework has ONE operation called cross-entropy-with-logits and none of
them expose a softmax followed by a log. This script shows the two reasons, by
running the naive version until it breaks.

  OVERFLOW.  exp(x) is inf in float64 for x above about 709.78. The naive
  softmax computes exp(z)/sum(exp(z)), so at logits around 1000 it is inf/inf,
  which is nan, and a single nan spreads through every parameter on the next
  update. Nothing raises.

  UNDERFLOW. If the correct class is assigned a probability that rounds to
  exactly 0.0, -log(0.0) is +inf. That is the more common of the two in
  practice, because it needs only a confidently wrong prediction rather than a
  huge logit.

The fix for the first is an exact algebraic identity: subtracting a constant
from every logit leaves the softmax unchanged, so subtract the largest one and
every exponent becomes <= 0. The fix for the second is to never form the
probability at all -- write the loss as logsumexp(z) - z_y, which is a
subtraction of finite numbers.
"""

from __future__ import annotations

import numpy as np

from _common import figure, rule, save

from netfs import (cross_entropy_unfused, log_sum_exp, softmax, softmax_cross_entropy,
                   softmax_naive)


def main() -> None:
    rule("1. the hand-worked example")
    z = np.array([[2.0, 1.0, 0.1]])
    y = np.array([0])
    print("  z = [2.0, 1.0, 0.1], true class 0")
    print("  e^2.0 = 7.3891,  e^1.0 = 2.7183,  e^0.1 = 1.1052,  sum = 11.2125")
    p = softmax(z)
    loss, grad = softmax_cross_entropy(z, y)
    print(f"  p        = {np.round(p[0], 4)}   (sums to {p.sum():.4f})")
    print(f"  L        = -ln(0.6590) = {loss:.4f}")
    print(f"  dL/dz    = p - onehot  = {np.round(grad[0], 4)}")
    print(f"  the gradient sums to {grad.sum():+.1e} -- exactly zero, for every input, always.")
    print("  Softmax outputs sum to 1 and the one-hot sums to 1, so the gradient can only")
    print("  MOVE probability mass between classes, never create it. Free correctness check.")

    rule("2. where the naive version dies")
    for scale in (1.0, 100.0, 700.0, 1000.0):
        zz = np.array([[0.0, 1.0, 2.0]]) * (scale / 2.0) if scale > 1 else z
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            naive_p = softmax_naive(zz)
            naive_l = cross_entropy_unfused(zz, y)
        stable_l, _ = softmax_cross_entropy(zz, y)
        print(f"  max logit {zz.max():8.1f} | naive p[0] {naive_p[0, 0]:>10.4g} | "
              f"naive loss {naive_l:>10.4g} | fused loss {stable_l:>10.4f}")
    print("  exp(709.79) is the last finite exponential in float64. Past it the naive")
    print("  denominator is inf, so the largest class reads inf/inf = nan and every other")
    print("  class reads 0/inf = 0.0 -- and -log(0.0) is +inf. Whichever entry you look at,")
    print("  the run is over. The fused version is not even slightly inconvenienced, because")
    print("  logsumexp(z) - z_y never exponentiates a positive number.")

    rule("3. the other failure: a confidently wrong prediction")
    zz = np.array([[0.0, 900.0]])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        print(f"  logits [0, 900], true class 0")
        print(f"  naive p = {softmax_naive(zz)[0]}  ->  -log(p[0]) = "
              f"{cross_entropy_unfused(zz, np.array([0])):.4g}")
    fused, fused_grad = softmax_cross_entropy(zz, np.array([0]))
    print(f"  fused loss = logsumexp(z) - z_0 = {fused:.4f}, gradient {fused_grad[0]}")
    print("  Note the gradient: every entry of p - onehot lies in [-1, 1] by construction,")
    print("  no matter how wrong the prediction was. An unfused implementation would have")
    print("  multiplied a softmax Jacobian by 1/p_y, and p_y is what just underflowed.")

    rule("4. the identity that makes it work")
    zz = np.array([[3.0, -1.0, 0.5]])
    print(f"  softmax(z)         = {np.round(softmax(zz)[0], 6)}")
    print(f"  softmax(z - 12345) = {np.round(softmax(zz - 12345.0)[0], 6)}   <- identical")
    print(f"  logsumexp(z)       = {log_sum_exp(zz, axis=1)[0]:.6f}")
    print(f"  logsumexp(z + 500) = {log_sum_exp(zz + 500.0, axis=1)[0]:.6f}  "
          f"(= the line above + 500)")

    # ------------------------------------------------------------------ figure
    scales = np.linspace(0, 1200, 160)
    naive_p0, stable_p0, naive_loss, fused_loss = [], [], [], []
    for s in scales:
        zz = np.array([[0.0, 0.5, 1.0]]) * s
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            naive_p0.append(softmax_naive(zz)[0, 2])
            naive_loss.append(cross_entropy_unfused(zz, np.array([2])))
        stable_p0.append(softmax(zz)[0, 2])
        fused_loss.append(softmax_cross_entropy(zz, np.array([2]))[0])
    naive_p0 = np.array(naive_p0)
    naive_loss = np.array(naive_loss)

    fig, ax = figure(1, 3, figsize=(12.5, 3.8))
    ax[0].plot(scales, stable_p0, color="#1565c0", lw=2.0, label="stable softmax")
    ax[0].plot(scales, naive_p0, color="#e57373", lw=1.6, ls="--", label="naive softmax")
    first_nan = scales[np.argmax(~np.isfinite(naive_p0))]
    ax[0].axvline(first_nan, color="#9e9e9e", ls=":", lw=1.2)
    ax[0].annotate(f"naive becomes nan\nat logit scale {first_nan:.0f}",
                   xy=(first_nan, 0.55), xytext=(first_nan * 0.32, 0.45), fontsize=7,
                   arrowprops=dict(arrowstyle="->", lw=0.8, color="#616161"))
    ax[0].set_xlabel("largest logit")
    ax[0].set_ylabel("p(correct class)")
    ax[0].set_title("p for logits [0, s/2, s]")
    ax[0].legend(fontsize=7, loc="lower right")

    # WHY it dies, in one line: the largest exponent each implementation hands
    # to exp(). The stable version subtracts the row max first, so its largest
    # exponent is exactly 0 forever. The naive version's grows with the logits
    # and walks off the float64 cliff at 709.78.
    ax[1].plot(scales, scales, color="#e57373", lw=1.6, ls="--",
               label="naive: max exponent = max logit")
    ax[1].plot(scales, np.zeros_like(scales), color="#1565c0", lw=2.0,
               label="stable: max exponent = 0, always")
    ax[1].axhline(709.78, color="#37474f", ls=":", lw=1.2)
    ax[1].annotate("709.78 -- the largest x with exp(x) finite in float64",
                   xy=(60, 730), fontsize=7)
    ax[1].set_xlabel("largest logit")
    ax[1].set_ylabel("largest exponent passed to exp()")
    ax[1].set_title("the cliff, and who walks off it")
    ax[1].legend(fontsize=7, loc="upper left")

    # The underflow branch: a confidently WRONG prediction, no huge logits
    # needed -- a gap of 750 between two logits is enough.
    gaps = np.linspace(0, 800, 200)
    naive_wrong, fused_wrong = [], []
    for g in gaps:
        zz = np.array([[0.0, g]])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            naive_wrong.append(cross_entropy_unfused(zz, np.array([0])))
        fused_wrong.append(softmax_cross_entropy(zz, np.array([0]))[0])
    naive_wrong = np.array(naive_wrong)
    ax[2].plot(gaps, fused_wrong, color="#1565c0", lw=2.0, label="fused")
    finite = np.isfinite(naive_wrong)
    ax[2].plot(gaps[finite], naive_wrong[finite], color="#e57373", lw=1.6, ls="--",
               label="softmax then -log")
    if (~finite).any():
        ax[2].axvline(gaps[np.argmax(~finite)], color="#9e9e9e", ls=":", lw=1.2)
        ax[2].annotate("-log(0.0) = +inf\nfrom here on",
                       xy=(gaps[np.argmax(~finite)], 620), xytext=(90, 700), fontsize=7,
                       arrowprops=dict(arrowstyle="->", lw=0.8, color="#616161"))
    ax[2].set_xlabel("logit gap, true class losing")
    ax[2].set_ylabel("cross-entropy loss")
    ax[2].set_title("a confidently wrong prediction")
    ax[2].legend(fontsize=7, loc="lower right")
    save(fig, "05-softmax-stability.png")


if __name__ == "__main__":
    main()
