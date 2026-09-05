"""Pooling: forward against a hand-computed window, backward against the
routing matrix computed by hand, and the two off-by-ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from netfs.pool import GlobalAvgPool2D, MaxPool2D

# The 4x4 input worked by hand in docs/DERIVATIONS.md section 8, with the
# absolute coordinate of each window's winner written down -- that coordinate
# is the entire content of the backward pass.
X4 = np.array([[1, 3, 2, 4],
               [5, 6, 1, 2],
               [7, 2, 8, 3],
               [1, 4, 2, 9]], dtype=float)[None, None]


def test_maxpool_forward_matches_the_hand_computation():
    out = MaxPool2D(2, 2).forward(X4)
    assert np.array_equal(out[0, 0], [[6, 4], [7, 9]])


def test_maxpool_backward_routes_to_the_winners_only():
    """Max is a switch, not a splitter: the whole upstream value goes to the
    argmax and exactly zero everywhere else. Twelve of these sixteen entries
    are zero, and that sparsity is the point.
    """
    layer = MaxPool2D(2, 2)
    layer.forward(X4)
    dx = layer.backward(np.array([[1.0, 2.0], [3.0, 4.0]])[None, None])
    expected = np.array([[0, 0, 0, 2],
                         [0, 1, 0, 0],
                         [3, 0, 0, 0],
                         [0, 0, 0, 4]], dtype=float)
    assert np.array_equal(dx[0, 0], expected)
    assert (dx == 0).sum() == 12


def test_maxpool_backward_conserves_the_upstream_total():
    """Nothing is created or destroyed by the routing: the sum of the scattered
    gradient equals the sum of the upstream gradient. A cheap invariant that
    catches a scatter which dropped or double-counted a window.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 6, 6))
    layer = MaxPool2D(2, 2)
    out = layer.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = layer.backward(dout)
    # approx, not ==: the two sums add the same numbers in a different order,
    # and floating-point addition is not associative.
    assert dx.sum() == pytest.approx(dout.sum())


def test_overlapping_windows_accumulate():
    """With stride < k a single input can win two windows, and its gradient is
    then the SUM of both. This is the multi-path rule again, and it is what
    would break if the backward scatter used assignment instead of add.
    """
    x = np.array([[0.0, 9.0, 0.0],
                  [0.0, 0.0, 0.0]])[None, None]     # the 9 wins both windows
    layer = MaxPool2D(k=2, stride=1)
    out = layer.forward(x)
    assert np.array_equal(out[0, 0], [[9.0, 9.0]])
    dx = layer.backward(np.array([[1.0, 1.0]])[None, None])
    assert dx[0, 0, 0, 1] == 2.0
    assert dx.sum() == 2.0


def test_pooling_is_per_channel_and_does_not_mix_channels():
    """Unlike convolution, which sums the channel axis away, pooling treats
    each channel independently. Give two channels different winners and check
    that neither leaks into the other.
    """
    x = np.zeros((1, 2, 2, 2))
    x[0, 0, 0, 0] = 5.0
    x[0, 1, 1, 1] = 7.0
    out = MaxPool2D(2, 2).forward(x)
    assert out[0, 0, 0, 0] == 5.0 and out[0, 1, 0, 0] == 7.0


def test_a_window_that_does_not_divide_the_input_drops_the_remainder():
    """5x5 with a 2x2 stride-2 window gives 2x2, not 3x3: the last row and
    column are seen by no window. It passes every test at an even size and
    breaks at an odd one.
    """
    x = np.arange(25, dtype=float).reshape(1, 1, 5, 5)
    out = MaxPool2D(2, 2).forward(x)
    assert out.shape == (1, 1, 2, 2)
    assert out.max() < x.max()          # the largest value lives in the dropped corner


def test_ties_send_the_whole_gradient_to_the_first_winner():
    """Equal values in a window do not split the gradient. argmax takes the
    first in flat order. Rarely consequential; it is why two "identical" runs
    on integer data can diverge, and it is a fair interview probe.
    """
    x = np.array([[4.0, 4.0], [4.0, 4.0]])[None, None]
    layer = MaxPool2D(2, 2)
    layer.forward(x)
    dx = layer.backward(np.array([[1.0]])[None, None])
    assert dx[0, 0, 0, 0] == 1.0
    assert dx.sum() == 1.0


def test_global_average_pool_is_resolution_independent():
    """The reason every backbone since ResNet ends this way: the output length
    is the channel count, whatever the spatial size. Flatten would give 1024
    numbers for one input and 4096 for the other, and the classifier that
    follows it would only accept one of them.
    """
    rng = np.random.default_rng(1)
    for size in (8, 13):
        out = GlobalAvgPool2D().forward(rng.standard_normal((2, 16, size, size)))
        assert out.shape == (2, 16)


def test_global_average_pool_backward_shares_evenly():
    layer = GlobalAvgPool2D()
    layer.forward(np.zeros((1, 1, 4, 4)))
    dx = layer.backward(np.array([[16.0]]))
    assert np.allclose(dx, 1.0)         # 16 / (4*4)
