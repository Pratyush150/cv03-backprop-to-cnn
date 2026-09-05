"""Example 08 -- pooling, stride, padding, and the off-by-one everyone hits.

Run:  python3 examples/08_pooling_shapes.py

Two subjects that belong together because they are both about WHERE the window
lands.

Pooling: max pooling has no parameters, halves the spatial size, and its
backward pass routes the entire upstream gradient to the position that won and
exactly zero everywhere else. Twelve of the sixteen entries in the routing
matrix below are zero. If you did not store the winning positions on the
forward pass you cannot do the backward pass at all.

Shapes: out = floor((in + 2*pad - k)/stride) + 1. The floor silently discards
the remainder, which means input columns that no window ever covers. With
k = 3 and stride 2, an input of 7 and an input of 8 both give an output of 3 --
and on the input of 8 the last column is read by nothing. There is no warning.
The symptom is a spatial dimension that drifts by one against a reference
implementation somewhere deep in a network, and an exception three layers after
the mistake.
"""

from __future__ import annotations

import numpy as np

from _common import figure, rule, save

from netfs import (GlobalAvgPool2D, MaxPool2D, check_layer, conv_out_size, receptive_field,
                   same_padding)

X4 = np.array([[1, 3, 2, 4],
               [5, 6, 1, 2],
               [7, 2, 8, 3],
               [1, 4, 2, 9]], dtype=float)[None, None]
DOUT = np.array([[1.0, 2.0], [3.0, 4.0]])[None, None]


def main() -> None:
    rule("1. max pooling, forward and backward, by hand")
    layer = MaxPool2D(2, 2)
    out = layer.forward(X4)
    dx = layer.backward(DOUT)
    print("  input                 windows and winners")
    windows = [((0, 0), (1, 1)), ((0, 2), (0, 3)), ((2, 0), (2, 0)), ((2, 2), (3, 3))]
    for r in range(4):
        print("    " + "  ".join(f"{v:2.0f}" for v in X4[0, 0, r]))
    for (r0, c0), (wr, wc) in windows:
        print(f"    rows {r0}-{r0+1}, cols {c0}-{c0+1}: max "
              f"{X4[0, 0, wr, wc]:.0f} at absolute position ({wr}, {wc})")
    print(f"  forward  = {out[0, 0].tolist()}")
    print(f"  upstream = {DOUT[0, 0].tolist()}")
    print("  backward (max ROUTES: everything to the winner, nothing to the rest):")
    for r in range(4):
        print("    " + "  ".join(f"{v:2.0f}" for v in dx[0, 0, r]))
    print(f"  {(dx == 0).sum()} of {dx.size} entries are exactly zero.")
    print(f"  the routed total {dx.sum():.0f} equals the upstream total {DOUT.sum():.0f} -- "
          f"nothing is created or lost.")

    rule("2. gradient checks")
    rng = np.random.default_rng(0)
    for k, stride in ((2, 2), (3, 1), (2, 1)):
        e = check_layer(MaxPool2D(k, stride), rng.standard_normal((2, 3, 6, 6)))
        print(f"  MaxPool2D(k={k}, stride={stride}) dX relative error {e['input']:.2e}"
              + ("   <- windows overlap here, so gradients accumulate" if stride < k else ""))
    e = check_layer(GlobalAvgPool2D(), rng.standard_normal((2, 3, 5, 5)))
    print(f"  GlobalAvgPool2D          dX relative error {e['input']:.2e}")

    rule("3. the output-size formula, against cases computed on paper")
    cases = [(32, 3, 1, 1, 32, "'same' padding"),
             (32, 3, 0, 1, 30, "valid: you lose k-1"),
             (32, 3, 1, 2, 16, "stride 2 halves it"),
             (7, 3, 0, 2, 3, "windows at columns 0-2, 2-4, 4-6"),
             (8, 3, 0, 2, 3, "SAME answer -- column 7 is read by nothing"),
             (224, 7, 3, 2, 112, "the ResNet-18 stem")]
    print(f"  {'in':>5}{'k':>4}{'pad':>5}{'stride':>8}{'out':>6}   note")
    for n, k, pad, stride, expected, note in cases:
        got = conv_out_size(n, k, pad, stride)
        assert got == expected, (n, k, pad, stride, got, expected)
        print(f"  {n:>5}{k:>4}{pad:>5}{stride:>8}{got:>6}   {note}")

    rule("4. which input columns actually get read")
    for n in (7, 8, 9):
        ho = conv_out_size(n, 3, 0, 2)
        seen = sorted({i * 2 + a for i in range(ho) for a in range(3)})
        missed = [c for c in range(n) if c not in seen]
        print(f"  in={n}, k=3, s=2 -> out={ho};  columns read: {seen}"
              + (f";  NEVER READ: {missed}" if missed else ";  all read"))
    print("  A test suite that only ever feeds sizes 7 and 9 never sees this. Feed it 8 and")
    print("  a whole column of the image is invisible to the layer, silently. Compute the")
    print("  size at every layer on paper before you start debugging the code.")

    rule("5. receptive field, and the stacked-3x3 argument")
    for stack in ([3], [3, 3], [3, 3, 3]):
        print(f"  {len(stack)} x 3x3 conv, stride 1 -> receptive field "
              f"{receptive_field(stack)}x{receptive_field(stack)}")
    print(f"  a stride-2 layer doubles everything behind it: "
          f"[3 (s2), 3] -> {receptive_field([3, 3], strides=[2, 1])}")
    print(f"  'same' padding for k=3,5,7: "
          f"{[same_padding(k) for k in (3, 5, 7)]}   (only defined for odd k)")

    # ------------------------------------------------------------------ figure
    fig, ax = figure(1, 4, figsize=(14, 3.5))
    from matplotlib import patches            # after figure(): the backend is set by then
    ax[0].imshow(X4[0, 0], cmap="Blues", vmin=0, vmax=10)
    for r in range(4):
        for c in range(4):
            ax[0].text(c, r, f"{X4[0, 0, r, c]:.0f}", ha="center", va="center", fontsize=10)
    for (_, _), (wr, wc) in windows:
        ax[0].add_patch(patches.Circle((wc, wr), 0.38, fill=False, lw=2.0,
                                       edgecolor="#c62828"))
    for edge in (1.5,):
        ax[0].axhline(edge, color="k", lw=1.0)
        ax[0].axvline(edge, color="k", lw=1.0)
    ax[0].set_title("input, with each window's winner")

    ax[1].imshow(out[0, 0], cmap="Blues", vmin=0, vmax=10)
    for r in range(2):
        for c in range(2):
            ax[1].text(c, r, f"{out[0, 0, r, c]:.0f}", ha="center", va="center", fontsize=13)
    ax[1].set_title("max pool 2x2 stride 2")

    ax[2].imshow(dx[0, 0], cmap="Reds", vmin=0, vmax=5)
    for r in range(4):
        for c in range(4):
            v = dx[0, 0, r, c]
            ax[2].text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=10,
                       color="#9e9e9e" if v == 0 else "k")
    ax[2].set_title("backward: upstream [[1,2],[3,4]] routed")

    # The coverage picture: which input columns a stride-2 3x3 window ever
    # touches, for inputs of 7, 8 and 9.
    ax[3].grid(False)
    for row, n in enumerate((7, 8, 9)):
        ho = conv_out_size(n, 3, 0, 2)
        seen = {i * 2 + a for i in range(ho) for a in range(3)}
        for c in range(n):
            ax[3].add_patch(patches.Rectangle(
                (c - 0.45, -row), 0.9, 0.8, facecolor="#1565c0" if c in seen else "#e57373",
                edgecolor="white"))
        ax[3].text(-0.85, -row + 0.4, f"in={n}\nout={ho}", ha="right", va="center", fontsize=7)
    ax[3].set_xlim(-3.0, 8.8)
    ax[3].set_ylim(-2.4, 1.2)
    ax[3].set_yticks([])
    ax[3].set_xticks(range(9))
    ax[3].set_xlabel("input column index")
    ax[3].set_title("k=3, s=2: red = read by no window")
    for a_ in ax[:3]:
        a_.set_xticks([])
        a_.set_yticks([])
        a_.grid(False)
    save(fig, "08-pooling-and-shapes.png")


if __name__ == "__main__":
    main()
