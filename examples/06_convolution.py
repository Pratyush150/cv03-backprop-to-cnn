"""Example 06 -- convolution as a layer: forward, backward, and the reference
implementations it has to agree with.

Run:  python3 examples/06_convolution.py

Four things happen here.

  1. A 4x4 input and a 3x3 vertical-edge kernel, computed by hand, then by the
     naive loop, then by im2col, then by scipy. All four agree exactly.
  2. The same kernel flipped 180 degrees, to show that what every framework
     calls "convolution" is really cross-correlation. scipy has both under
     different names, and the signs come out opposite.
  3. The backward pass: dW, db and dX, each checked against a central
     difference, on a layer with padding and on a layer with stride 2.
  4. A picture: a real edge map produced by our layer next to the same edge map
     from scipy, with their difference, which is zero to machine precision.

The figure is on a synthetic test card rather than a photograph so it needs no
data files and the expected answer is knowable in advance -- a vertical-edge
detector should respond on the vertical edges and nowhere else.
"""

from __future__ import annotations

import numpy as np

from _common import figure, rule, save

from netfs import (Conv2D, check_layer, conv2d_im2col, conv2d_naive, conv_macs, conv_out_size,
                   conv_params, im2col, im2col_indices)

X4 = np.array([[1, 2, 3, 0],
               [0, 1, 2, 3],
               [3, 0, 1, 2],
               [2, 3, 0, 1]], dtype=float)[None, None]
KV = np.array([[1, 0, -1],
               [1, 0, -1],
               [1, 0, -1]], dtype=float)[None, None]


def test_card(h=180, w=260):
    """A card with one hard vertical edge, one soft vertical ramp and one
    horizontal edge. The vertical-edge kernel must fire on the first two and
    ignore the third; if it fires on the third, the kernel is transposed.
    """
    img = np.zeros((h, w))
    img[:, 40:90] = 1.0                                   # hard vertical edges
    ramp = np.linspace(0.0, 1.0, 60)
    img[:, 120:180] = ramp[None, :]                       # soft vertical ramp
    img[110:, 200:] = 1.0                                 # a horizontal edge
    return img


def main() -> None:
    rule("1. the hand computation, four ways")
    print("  input                        kernel (vertical edge)")
    for r in range(4):
        k = "  ".join(f"{v:2.0f}" for v in KV[0, 0, r]) if r < 3 else ""
        print(f"    {'  '.join(f'{v:2.0f}' for v in X4[0, 0, r])}          {k}")
    print("  each kernel row is (left - right), the middle column being zero:")
    print("    out[0,0] = (1-3) + (0-2) + (3-1) = -2")
    loops = conv2d_naive(X4, KV)[0, 0]
    gemm = conv2d_im2col(X4, KV)[0, 0]
    print(f"  by hand    : [[-2, -2], [2, -2]]")
    print(f"  loops      : {loops.tolist()}")
    print(f"  im2col GEMM: {gemm.tolist()}")
    try:
        from scipy.signal import convolve2d, correlate2d
    except ImportError:
        print("  scipy not installed -- skipping the cross-check")
    else:
        print(f"  scipy correlate2d (no flip): {correlate2d(X4[0,0], KV[0,0], 'valid').tolist()}")
        print(f"  scipy convolve2d  (flips)  : {convolve2d(X4[0,0], KV[0,0], 'valid').tolist()}")
        print("  Every sign is opposite. A 'convolution layer' computes a CORRELATION.")
        print("  Irrelevant for learned filters -- backprop just learns the flipped kernel --")
        print("  and it matters the moment you port a hand-designed Sobel into a network.")

    rule("2. the index arithmetic im2col does")
    cols, ho, wo = im2col(X4, 3, 3, 1, 0)
    cc, ii, jj = im2col_indices(1, 3, 3, ho, wo, 1)
    print(f"  output is {ho}x{wo}, so the column matrix is {cols.shape} = (Ho*Wo, kh*kw*Cin)")
    print(f"  output pixel 0 gathers rows {ii[0]} cols {jj[0]}")
    print(f"  output pixel 1 gathers rows {ii[1]} cols {jj[1]}   <- same rows, columns +stride")
    print(f"  cols[0] = {cols[0]}")
    print(f"  cols[0] . kernel.ravel() = {cols[0] @ KV.reshape(-1)}   <- that is out[0,0]")

    rule("3. the backward pass, gradient-checked")
    rng = np.random.default_rng(0)
    for stride, pad in ((1, 1), (2, 0)):
        layer = Conv2D(3, 4, 3, stride=stride, pad=pad, rng=np.random.default_rng(1))
        errs = check_layer(layer, rng.standard_normal((2, 3, 7, 7)))
        print(f"  Conv2D(3->4, k3, stride={stride}, pad={pad}):  "
              f"dW {errs['W']:.2e}   db {errs['b']:.2e}   dX {errs['input']:.2e}")
    print("  dW = dOut_flat.T @ cols sums over every position the filter visited -- 4096")
    print("  contributions for one filter on a 64x64 map. That is weight sharing seen from")
    print("  the backward side, and it is why conv layers learn from less data than dense.")

    rule("4. the counting arguments")
    dense = 224 * 224 * 3 * 1000
    conv = conv_params(3, 3, 64)
    print(f"  dense layer, 224x224x3 -> 1000 units : {dense:,} weights")
    print(f"  conv 3x3, 3 -> 64 channels           : {conv:,} weights, "
          f"and independent of image size")
    print(f"  ratio                                : about {dense // conv:,}x")
    stem = conv_out_size(224, 7, 3, 2)
    print(f"  ResNet-18 stem: 224 -> {stem}, {conv_macs(stem, stem, 64, 7, 3):,} MACs "
          f"= {2 * conv_macs(stem, stem, 64, 7, 3) / 1e9:.3f} GFLOPs")
    two3, one5 = 2 * conv_params(3, 64, 64, bias=False), conv_params(5, 64, 64, bias=False)
    print(f"  two stacked 3x3 (C=64): {two3:,} weights, 5x5 receptive field")
    print(f"  one 5x5         (C=64): {one5:,} weights, the same receptive field")
    print(f"  the stack is {100 * (1 - two3 / one5):.0f}% cheaper AND has one extra "
          f"nonlinearity in it.")

    # ------------------------------------------------------------------ figure
    img = test_card()
    x = img[None, None]
    ours = conv2d_naive(x, KV)[0, 0]
    fig, ax = figure(1, 4, figsize=(13.5, 3.2))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("input test card")
    ax[1].imshow(ours, cmap="coolwarm", vmin=-3, vmax=3)
    ax[1].set_title("netfs conv2d_naive")
    try:
        from scipy.signal import convolve2d, correlate2d
        ref = correlate2d(img, KV[0, 0], mode="valid")
        flipped = convolve2d(img, KV[0, 0], mode="valid")
        diff = np.abs(ours - ref)
        # Panel 3 is scipy's CONVOLUTION -- the same kernel, flipped -- so the
        # colours are the exact reverse of panel 2. That reversal is the whole
        # correlation-versus-convolution point, in a picture.
        ax[2].imshow(flipped, cmap="coolwarm", vmin=-3, vmax=3)
        ax[2].set_title("scipy convolve2d (kernel flipped)")
        ax[3].imshow(diff, cmap="magma", vmin=0, vmax=1e-15)
        ax[3].set_title("netfs vs scipy correlate2d")
        ax[3].text(0.5, 0.5, f"identical everywhere\nmax |difference| = {diff.max():.1e}",
                   transform=ax[3].transAxes, ha="center", va="center", color="white",
                   fontsize=9)
        print(f"\n  max |netfs - scipy correlate2d| over the whole test card: {diff.max():.2e}")
    except ImportError:
        for a_ in ax[2:]:
            a_.text(0.5, 0.5, "scipy not installed", ha="center", va="center")
    for a_ in ax:
        a_.grid(False)
        a_.set_xticks([])
        a_.set_yticks([])
    save(fig, "06-conv-vs-reference.png")


if __name__ == "__main__":
    main()
