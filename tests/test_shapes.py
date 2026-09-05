"""Output-size arithmetic against cases computed by hand.

These look trivial. They are the tests that stop a whole class of bug where a
spatial dimension drifts by one somewhere deep in a network and the exception
surfaces three layers later.
"""

from __future__ import annotations

import pytest

from netfs.shapes import (conv_macs, conv_out_size, conv_params, receptive_field, same_padding)


@pytest.mark.parametrize(
    "n, k, pad, stride, expected",
    [
        (32, 3, 1, 1, 32),   # 'same' padding: (32 + 2 - 3)//1 + 1 = 32
        (32, 3, 0, 1, 30),   # valid: you lose k-1 = 2
        (32, 3, 1, 2, 16),   # stride 2 halves it
        (7, 3, 0, 2, 3),     # (7-3)//2 + 1 = 3, windows at columns 0-2, 2-4, 4-6
        (8, 3, 0, 2, 3),     # SAME answer on a larger input: column 7 is read by nothing
        (224, 7, 3, 2, 112),  # ResNet-18's stem
        (5, 5, 0, 1, 1),     # kernel exactly fills the input
    ],
)
def test_conv_out_size_hand_computed(n, k, pad, stride, expected):
    assert conv_out_size(n, k, pad, stride) == expected


def test_floor_silently_drops_a_column():
    """The off-by-one everyone hits, stated as an assertion.

    Inputs of 7 and 8 give the same output size at k=3, s=2. The eighth column
    of the larger input is covered by no window at all, and nothing warns.
    """
    assert conv_out_size(7, 3, 0, 2) == conv_out_size(8, 3, 0, 2) == 3
    last_window_end = (3 - 1) * 2 + 3      # start of last window + kernel reach
    assert last_window_end == 7           # covers indices 0..6 -- index 7 unseen


def test_dilation_collapses_to_the_plain_formula():
    for n, k, pad, stride in [(32, 3, 1, 1), (17, 5, 2, 3), (9, 3, 0, 2)]:
        assert conv_out_size(n, k, pad, stride, dilation=1) == conv_out_size(n, k, pad, stride)
    # A dilated 3x3 reaches as far as a 5x5, so it shrinks the input the same way.
    assert conv_out_size(32, 3, 0, 1, dilation=2) == conv_out_size(32, 5, 0, 1)


def test_window_that_does_not_fit_raises():
    with pytest.raises(ValueError, match="does not fit"):
        conv_out_size(2, 3, 0, 1)


def test_same_padding_only_exists_for_odd_kernels():
    assert same_padding(3) == 1 and same_padding(5) == 2 and same_padding(7) == 3
    for k in (3, 5, 7):
        assert conv_out_size(28, k, same_padding(k), 1) == 28
    with pytest.raises(ValueError, match="not symmetric"):
        same_padding(4)


def test_parameter_counts_are_image_size_independent():
    """1,792 weights for a 3x3, 3->64 conv, whatever the image size."""
    assert conv_params(3, 3, 64) == 1792
    assert conv_params(3, 3, 64, bias=False) == 1728
    dense = 224 * 224 * 3 * 1000
    assert dense == 150_528_000
    assert dense // conv_params(3, 3, 64) == 84_000     # the ratio, to the nearest whole number


def test_resnet_stem_macs():
    """112*112*64*(7*7*3) = 118,013,952 MACs ~ 0.236 GFLOPs at 2 FLOPs/MAC."""
    assert conv_macs(112, 112, 64, 7, 3) == 118_013_952


def test_receptive_field_stacking():
    assert receptive_field([3]) == 3
    assert receptive_field([3, 3]) == 5        # two 3x3 see 5x5
    assert receptive_field([3, 3, 3]) == 7     # three see 7x7 -- the VGG argument
    # ...and the parameter counts that make the argument decisive, at C=64:
    assert 2 * conv_params(3, 64, 64, bias=False) == 73_728
    assert conv_params(5, 64, 64, bias=False) == 102_400
    assert 3 * conv_params(3, 64, 64, bias=False) == 110_592
    assert conv_params(7, 64, 64, bias=False) == 200_704
    # A stride-2 layer doubles everything behind it.
    assert receptive_field([3, 3], strides=[2, 1]) == 7
