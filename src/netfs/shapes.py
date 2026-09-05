"""Output-size arithmetic for convolution and pooling.

One formula, one off-by-one, and a whole chapter of confusing bugs downstream
if you get it wrong. It lives in its own module because every layer in this
package calls it and because it is worth testing in isolation against
hand-computed cases (tests/test_shapes.py).
"""

from __future__ import annotations


def conv_out_size(n: int, k: int, pad: int = 0, stride: int = 1, dilation: int = 1) -> int:
    """Spatial size of one axis after a conv/pool window.

        out = floor((n + 2*pad - dilation*(k - 1) - 1) / stride) + 1

    Read it as: `n + 2*pad` is how many positions exist after padding;
    `dilation*(k-1) + 1` is how far the window physically reaches; the
    subtraction leaves the number of *extra* positions the window can be
    slid into, and dividing by the stride counts how many slides that is.
    The `+ 1` is the window's own starting position, which needs no slide.

    The floor is not cosmetic. Python's `//` performs it, and it silently
    discards the remainder -- input columns that no window ever covers. With
    n=7, k=3, s=2 you get 3 outputs covering columns 0-2, 2-4, 4-6. Grow the
    input to n=8 and you still get 3: column 7 is read by nothing at all, no
    warning, no error. That is why this function exists as a named, tested
    thing instead of being inlined three times: the place a spatial size goes
    wrong is never the place the exception is raised.
    """
    if k <= 0 or stride <= 0 or dilation <= 0:
        raise ValueError(f"k, stride and dilation must be positive (got {k}, {stride}, {dilation})")
    if pad < 0:
        raise ValueError(f"pad must be non-negative (got {pad})")
    reach = dilation * (k - 1) + 1
    out = (n + 2 * pad - reach) // stride + 1
    if out <= 0:
        # Without this the caller gets an empty array and a confusing error
        # several layers later. A 3x3 kernel on a 2x2 input is a design bug,
        # not a runtime condition to be handled quietly.
        raise ValueError(
            f"window of reach {reach} does not fit in padded size {n + 2 * pad} "
            f"(n={n}, k={k}, pad={pad}, stride={stride}, dilation={dilation})"
        )
    return out


def same_padding(k: int, dilation: int = 1) -> int:
    """Padding that keeps the size unchanged at stride 1.

    Solve `n = (n + 2p - d*(k-1) - 1)//1 + 1` for p: `p = d*(k-1)/2`. It is an
    integer only for odd k, which is the real reason essentially every kernel
    you meet is 3x3, 5x5 or 7x7. An even kernel cannot be centred on a pixel,
    so "same" padding has to be asymmetric and frameworks disagree about which
    side gets the extra row.
    """
    if k % 2 == 0:
        raise ValueError(f"'same' padding is not symmetric for even k (got k={k})")
    return dilation * (k - 1) // 2


def conv_params(k: int, c_in: int, c_out: int, bias: bool = True) -> int:
    """Weight count of a conv layer: k*k*c_in*c_out (+ c_out).

    Note what is absent: the image size. This is the whole weight-sharing
    argument in one function. A dense layer from a 224x224x3 image to 1000
    units needs 150,528,000 weights; a 3x3 conv with 3 in and 64 out needs
    1,792 whether the image is 224x224 or 4000x3000.
    """
    return k * k * c_in * c_out + (c_out if bias else 0)


def conv_macs(h_out: int, w_out: int, c_out: int, k: int, c_in: int) -> int:
    """Multiply-accumulates for one conv forward pass: one MAC per output
    element per kernel weight. The usual convention is 2 FLOPs per MAC.
    """
    return h_out * w_out * c_out * (k * k * c_in)


def receptive_field(kernels, strides=None) -> int:
    """Receptive field of a stack of layers, back-to-front.

        r_{i-1} = (r_i - 1) * stride_i + k_i,  starting from r_L = 1

    Two stacked 3x3 convs at stride 1 give 5; three give 7. That is the VGG
    argument, and the parameter counts that make it decisive are in
    docs/DERIVATIONS.md.
    """
    kernels = list(kernels)
    strides = [1] * len(kernels) if strides is None else list(strides)
    if len(strides) != len(kernels):
        raise ValueError("kernels and strides must have the same length")
    r = 1
    for k, s in zip(reversed(kernels), reversed(strides)):
        r = (r - 1) * s + k
    return r
