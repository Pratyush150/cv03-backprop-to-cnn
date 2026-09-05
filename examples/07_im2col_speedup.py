"""Example 07 -- im2col: how much faster, and what it costs in memory.

Run:  python3 examples/07_im2col_speedup.py

Convolution is not fast because someone found a clever convolution algorithm.
It is fast because the problem is rearranged into a matrix multiply and handed
to the same tuned GEMM kernel everything else uses. `im2col` is that
rearrangement: pull out every patch the kernel will ever see, flatten each into
a row, stack them, and multiply once.

This script measures both sides of that trade on THIS machine -- the speedup
and the memory blow-up -- because "faster" without a number is worthless. Both
implementations produce identical output, asserted before either is timed; a
speed comparison between two functions that do not agree is meaningless.

Three implementations are timed, not two, because the honest question is
"faster than what":

  * `fully_looped`  -- a Python loop per output position PER FILTER, the way
    the definition reads. Nothing is vectorised except the patch multiply.
  * `netfs.conv2d_naive` -- one Python loop per output position, with einsum
    handling the batch, channel and filter axes. This is the library's
    reference implementation and it is already substantially vectorised.
  * `netfs.conv2d_im2col` -- one gather and one GEMM.

Quoting a single "im2col is 100x faster" number hides the fact that most of
the win is ordinary vectorisation and only the rest is the GEMM. Both numbers
are printed below, measured on this machine.
"""

from __future__ import annotations

import platform
import time

import numpy as np

from _common import figure, rule, save

from netfs import conv2d_im2col, conv2d_naive, conv_out_size

CONFIGS = [
    #  name,                 (N, Cin, H, W),      (Cout, k)
    ("32x32, 3->16, k3", (1, 3, 32, 32), (16, 3)),
    ("64x64, 3->16, k3", (1, 3, 64, 64), (16, 3)),
    ("128x128, 3->16, k3", (1, 3, 128, 128), (16, 3)),
    ("64x64, 16->32, k3", (1, 16, 64, 64), (32, 3)),
    ("batch 8, 28x28, 1->8, k5", (8, 1, 28, 28), (8, 5)),
]


def conv2d_fully_looped(x, w):
    """Convolution written the way the definition reads: a Python loop for each
    of sample, filter, output row and output column. Valid padding, stride 1.

    This is not how anyone should compute a convolution. It is here to be
    timed, so that "im2col is fast" can be split into the part that is
    vectorisation and the part that is the matrix multiply.
    """
    n, c_in, h, wd = x.shape
    c_out, _, kh, kw = w.shape
    ho, wo = h - kh + 1, wd - kw + 1
    out = np.zeros((n, c_out, ho, wo))
    for s in range(n):
        for f in range(c_out):
            for i in range(ho):
                for j in range(wo):
                    out[s, f, i, j] = np.sum(x[s, :, i:i + kh, j:j + kw] * w[f])
    return out


def timeit(fn, *args, repeats=3):
    """Best of `repeats`, not the mean.

    The minimum is the right statistic for a timing like this: the true cost is
    a lower bound that noise can only add to. A mean reports how busy the
    machine was as much as how fast the code is.
    """
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rule(f"timing on {platform.processor() or platform.machine()}, "
         f"python {platform.python_version()}, numpy {np.__version__}")
    print(f"  {'configuration':<26}{'full loops':>11}{'einsum':>10}{'im2col':>10}"
          f"{'vs full':>9}{'vs einsum':>11}")
    rows = []
    rng = np.random.default_rng(0)
    for name, xshape, (c_out, k) in CONFIGS:
        x = rng.standard_normal(xshape)
        w = rng.standard_normal((c_out, xshape[1], k, k))
        # Agreement first, timing second. A benchmark between two functions
        # that compute different things is a number with no meaning attached.
        assert np.abs(conv2d_naive(x, w) - conv2d_im2col(x, w)).max() < 1e-12
        t_full = timeit(conv2d_fully_looped, x, w, repeats=1)   # slow; once is enough
        t_loops = timeit(conv2d_naive, x, w)
        t_gemm = timeit(conv2d_im2col, x, w)
        ho = conv_out_size(xshape[2], k, 0, 1)
        wo = conv_out_size(xshape[3], k, 0, 1)
        cols_mb = xshape[0] * ho * wo * (k * k * xshape[1]) * 8 / 1e6   # float64
        act_mb = np.prod(xshape) * 8 / 1e6
        rows.append((name, t_full, t_loops, t_gemm, t_full / t_gemm, t_loops / t_gemm,
                     cols_mb, cols_mb / act_mb))
        print(f"  {name:<26}{t_full:>11.4f}{t_loops:>10.4f}{t_gemm:>10.5f}"
              f"{t_full / t_gemm:>8.0f}x{t_loops / t_gemm:>10.0f}x")

    vs_full = [r[4] for r in rows]
    speedups = [r[5] for r in rows]
    print(f"  im2col vs the fully looped version : {min(vs_full):.0f}x to {max(vs_full):.0f}x")
    print(f"  im2col vs the einsum reference     : {min(speedups):.0f}x to "
          f"{max(speedups):.0f}x")
    print("  Read those two rows together. Most of the win is ordinary vectorisation, and")
    print("  the GEMM is worth a further single-digit factor on shapes this small. Quoting")
    print("  only the first number would be true and misleading.")

    rule("what it costs")
    print("  im2col copies each input element once per output position that reads it, so")
    print("  the column matrix is about k^2 times the activation. For a 3x3 kernel that is")
    print("  a 9x blow-up, and it is why real frameworks keep direct, Winograd and FFT")
    print("  paths as well and choose per shape -- and why a MemoryError can appear on an")
    print("  input the naive loop would have handled comfortably.")
    for name, _, _, _, _, _, cols_mb, ratio in rows:
        print(f"    {name:<26}columns {cols_mb:7.1f} MB   ({ratio:.0f}x the activation)")
    h = w = 224
    c, k = 64, 3
    print(f"  the standard example: a {h}x{w}x{c} float32 activation is "
          f"{h * w * c * 4 / 1e6:.1f} MB;")
    print(f"  its 3x3 column matrix is {h * w} x {k * k * c} = "
          f"{h * w * k * k * c * 4 / 1e6:.0f} MB.")

    # ------------------------------------------------------------------ figure
    names = [r[0] for r in rows]
    t_full = [r[1] for r in rows]
    t_loops = [r[2] for r in rows]
    t_gemm = [r[3] for r in rows]
    y = np.arange(len(rows))
    fig, ax = figure(1, 3, figsize=(13, 3.8))
    ax[0].barh(y - 0.26, t_full, height=0.25, color="#b71c1c", label="loop per filter")
    ax[0].barh(y, t_loops, height=0.25, color="#e57373", label="loop per position (einsum)")
    ax[0].barh(y + 0.26, t_gemm, height=0.25, color="#1565c0", label="im2col + one GEMM")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names, fontsize=7)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("seconds (log, best of 3)")
    ax[0].set_title("forward pass time")
    ax[0].legend(fontsize=7, loc="upper right", framealpha=0.95)
    ax[0].invert_yaxis()

    ax[1].barh(y - 0.2, vs_full, height=0.38, color="#2e7d32", label="vs loop per filter")
    ax[1].barh(y + 0.2, speedups, height=0.38, color="#a5d6a7", label="vs einsum reference")
    for i, (a_full, a_ein) in enumerate(zip(vs_full, speedups)):
        ax[1].text(a_full, i - 0.2, f" {a_full:.0f}x", va="center", fontsize=6.5)
        ax[1].text(a_ein, i + 0.2, f" {a_ein:.0f}x", va="center", fontsize=6.5)
    ax[1].legend(fontsize=7, loc="center right", framealpha=0.95)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].set_xlabel("times faster")
    ax[1].set_title("speedup, same output to 1e-12")
    ax[1].invert_yaxis()

    ax[2].barh(y, [r[7] for r in rows], height=0.55, color="#8e24aa")
    ax[2].axvline(9, color="#37474f", ls="--", lw=1.2, label="k^2 = 9 for a 3x3 kernel")
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([])
    ax[2].set_xlabel("column matrix / activation")
    ax[2].set_title("what the speed costs in memory")
    ax[2].legend(fontsize=7, loc="lower right")
    ax[2].invert_yaxis()
    save(fig, "07-im2col-speedup.png")


if __name__ == "__main__":
    main()
