"""Convolution: against a hand-computed answer, against scipy, and im2col
against the naive loops.

Three independent references for one operation, because "my convolution looks
right" is not a statement anyone should accept, least of all from themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from netfs.conv import Conv2D, conv2d_im2col, conv2d_naive, im2col, im2col_indices

# The 4x4 input and the vertical-edge kernel worked by hand in
# docs/DERIVATIONS.md section 7. Each row of the kernel contributes
# (left - right) because its middle column is zero, which is what makes the
# arithmetic checkable in your head.
X4 = np.array([[1, 2, 3, 0],
               [0, 1, 2, 3],
               [3, 0, 1, 2],
               [2, 3, 0, 1]], dtype=float)[None, None]
KV = np.array([[1, 0, -1],
               [1, 0, -1],
               [1, 0, -1]], dtype=float)[None, None]
HAND = np.array([[-2.0, -2.0], [2.0, -2.0]])


def test_naive_conv_matches_the_hand_computation():
    assert np.array_equal(conv2d_naive(X4, KV)[0, 0], HAND)


def test_im2col_conv_matches_the_hand_computation():
    assert np.array_equal(conv2d_im2col(X4, KV)[0, 0], HAND)


def test_what_we_compute_is_correlation_not_convolution():
    """The distinction that catches everyone porting a hand-designed kernel
    into a network.

    Flip the kernel 180 degrees and every sign of the output flips. scipy has
    both operations under different names: `correlate2d` does not flip (and
    agrees with us, and with cv2.filter2D, and with every deep learning
    framework), `convolve2d` does flip. For LEARNED filters this is irrelevant
    -- backprop simply learns the flipped kernel -- and for a Sobel operator
    carried over from a classical pipeline it means your gradient directions
    come out 180 degrees wrong.
    """
    scipy_signal = pytest.importorskip("scipy.signal")
    k = KV[0, 0]
    corr = scipy_signal.correlate2d(X4[0, 0], k, mode="valid")
    conv = scipy_signal.convolve2d(X4[0, 0], k, mode="valid")
    assert np.array_equal(corr, HAND)               # ours agrees with correlation
    assert np.array_equal(conv, -HAND)              # ...and every sign is flipped
    assert np.array_equal(conv2d_naive(X4, KV)[0, 0], corr)
    # ...and flipping our kernel by hand reproduces scipy's convolution exactly.
    flipped = KV[:, :, ::-1, ::-1].copy()
    assert np.array_equal(conv2d_naive(X4, flipped)[0, 0], conv)


@pytest.mark.parametrize("stride, pad", [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 2)])
def test_im2col_matches_naive_loops_exactly(stride, pad):
    """Not "approximately". The two routes perform the same multiplications in
    a different order, so they can differ in the last bit of a float; anything
    larger than 1e-12 is an indexing bug, not rounding.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 9, 9))
    w = rng.standard_normal((4, 3, 3, 3))
    b = rng.standard_normal(4)
    a = conv2d_naive(x, w, b, stride, pad)
    c = conv2d_im2col(x, w, b, stride, pad)
    assert a.shape == c.shape
    assert np.abs(a - c).max() < 1e-12


def test_multichannel_conv_against_scipy_channel_by_channel():
    """A multi-channel conv is a sum of single-channel correlations, one per
    input channel, and NOT a kernel sliding along the channel axis.

    Believing the sliding-in-depth story is the most common CNN misconception
    in interviews and it makes every parameter count wrong. Here it is settled
    by construction: build the reference by correlating each channel separately
    with scipy and adding the results.
    """
    scipy_signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 4, 8, 8))
    w = rng.standard_normal((2, 4, 3, 3))
    ours = conv2d_naive(x, w)[0]
    for f in range(2):
        ref = sum(scipy_signal.correlate2d(x[0, c], w[f, c], mode="valid") for c in range(4))
        assert np.abs(ours[f] - ref).max() < 1e-12


def test_im2col_index_trace():
    """The index arithmetic, spelled out for the first two output pixels.

    Row r of the column matrix is output pixel (r // wo, r % wo); the columns
    within a row run in (channel, kernel row, kernel column) order, which is
    exactly the order `w.reshape(c_out, -1)` flattens the weights in. Those two
    orders agreeing is the entire correctness argument for the GEMM.
    """
    cc, ii, jj = im2col_indices(1, 3, 3, 2, 2, 1)
    assert np.array_equal(ii[0], [0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert np.array_equal(jj[0], [0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert np.array_equal(ii[1], ii[0])            # same rows...
    assert np.array_equal(jj[1], jj[0] + 1)        # ...columns shifted by the stride
    cols, ho, wo = im2col(X4, 3, 3, 1, 0)
    assert cols.shape == (ho * wo, 9)
    # One row of the matrix, dotted with the flattened kernel, is one output.
    assert cols[0] @ KV.reshape(-1) == HAND[0, 0]


def test_im2col_channel_order_matches_the_weight_reshape():
    """If the (c, a, b) decomposition in im2col_indices ever drifts out of step
    with `w.reshape(c_out, -1)`, the convolution computes a permuted kernel and
    nothing raises. Pin it with a kernel that is different in every channel.
    """
    x = np.arange(2 * 3 * 3, dtype=float).reshape(1, 2, 3, 3)
    w = np.arange(2 * 3 * 3, dtype=float).reshape(1, 2, 3, 3)
    assert np.array_equal(conv2d_im2col(x, w), conv2d_naive(x, w))
    assert conv2d_naive(x, w)[0, 0, 0, 0] == float((x * w).sum())


def test_conv_layer_forward_matches_the_functions():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 2, 7, 7))
    layer = Conv2D(2, 5, 3, pad=1, rng=rng)
    out = layer.forward(x)
    ref = conv2d_naive(x, layer.params["W"], layer.params["b"], 1, 1)
    assert out.shape == (3, 5, 7, 7)
    assert np.abs(out - ref).max() < 1e-12


def test_zero_padding_asserts_the_world_outside_is_black():
    """Padding is a modelling assumption, not a formatting detail.

    On an all-ones image with an all-ones 3x3 kernel, the interior sums to 9
    and the corner to 4 -- the missing 5 is the border the padding invented.
    On raw 0-255 pixel data that invented border reads as a very strong edge
    all the way round the image, which is why you normalise before you pad.
    """
    x = np.ones((1, 1, 5, 5))
    w = np.ones((1, 1, 3, 3))
    out = conv2d_naive(x, w, pad=1)[0, 0]
    assert out[2, 2] == 9.0
    assert out[0, 0] == 4.0
    assert out.shape == (5, 5)


def test_im2col_memory_bill():
    """The k^2 blow-up, computed rather than asserted in prose: a 224x224x64
    float32 activation is 12.8 MB and its 3x3 column matrix is 116 MB, 9x.
    """
    h = w = 224
    c, k = 64, 3
    activation_mb = h * w * c * 4 / 1e6
    cols_mb = (h * w) * (k * k * c) * 4 / 1e6
    assert round(activation_mb, 1) == 12.8
    assert round(cols_mb) == 116
    assert round(cols_mb / activation_mb) == 9
