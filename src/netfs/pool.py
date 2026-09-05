"""Pooling layers: max pooling with a routing backward pass, and global
average pooling.

Max pooling has no parameters. It halves the spatial dimensions, throws away
precise position and keeps "the strongest response was somewhere around here".
Its backward pass is the third of the three flow patterns that cover most of a
network: max ROUTES. The entire upstream gradient goes to the winner and
exactly zero to everything else in the window -- no splitting, no averaging.
Twelve of the sixteen entries in a 4x4 example are zero.
"""

from __future__ import annotations

import numpy as np

from .conv import col2im, im2col
from .layers import Layer
from .shapes import conv_out_size


class MaxPool2D(Layer):
    """Max pooling, (N, C, H, W) -> (N, C, ho, wo).

    Implementation note that is also the lesson: pooling is applied to each
    channel INDEPENDENTLY -- unlike convolution, which sums the channel axis
    away. So the whole thing is done by folding the channels into the batch
    axis, `x.reshape(N*C, 1, H, W)`, and reusing the im2col machinery from
    netfs.conv. Every patch becomes a row of k*k numbers, the forward pass is
    one `argmax` along that row, and the backward pass is `col2im` of a matrix
    that is zero everywhere except at those argmax positions.

    Reusing im2col here is deliberate: it means the pooling backward pass
    inherits the `np.add.at` scatter that col2im already gets right, which
    matters the moment stride < k and windows overlap, because then a single
    input can win two windows and its gradient is the sum of both.

    Ties: `argmax` returns the FIRST maximum in flat order, so if two values in
    a window are equal the entire gradient goes to one of them rather than
    being split. Rarely consequential, and it is why two "identical" runs on
    integer-valued data can diverge.

    A window that does not divide the input drops the remainder: the floor in
    the size formula means the last row or column is seen by no window at all.
    That is an off-by-one that passes every test at 8x8 and breaks at 9x9.
    """

    def __init__(self, k: int = 2, stride: int | None = None) -> None:
        super().__init__()
        self.k = k
        self.stride = k if stride is None else stride  # non-overlapping by default
        self.argmax: np.ndarray | None = None
        self.folded_shape: tuple[int, ...] | None = None
        self.out_hw: tuple[int, int] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        ho = conv_out_size(h, self.k, 0, self.stride)
        wo = conv_out_size(w, self.k, 0, self.stride)
        folded = x.reshape(n * c, 1, h, w)          # channels into the batch axis
        cols, _, _ = im2col(folded, self.k, self.k, self.stride, 0)   # (N*C*ho*wo, k*k)
        self.argmax = np.argmax(cols, axis=1)
        self.folded_shape = folded.shape
        self.out_hw = (ho, wo)
        out = cols[np.arange(cols.shape[0]), self.argmax]
        return out.reshape(n, c, ho, wo)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n, c, ho, wo = dout.shape
        rows = np.arange(self.argmax.shape[0])
        dcols = np.zeros((self.argmax.shape[0], self.k * self.k), dtype=np.float64)
        # The routing, in one line: every upstream value lands on exactly the
        # position that won its window. Everything else stays zero.
        dcols[rows, self.argmax] = dout.reshape(-1)
        dfolded = col2im(dcols, self.folded_shape, self.k, self.k, self.stride, 0)
        return dfolded.reshape(n, c, *self.folded_shape[2:])


class GlobalAvgPool2D(Layer):
    """Mean over H and W: (N, C, H, W) -> (N, C).

    The backward pass is the average shared out again -- each of the H*W inputs
    contributed 1/(H*W) of the output, so each receives 1/(H*W) of the
    gradient. Contrast with max pooling, where one input gets everything.

    This is the layer that makes a backbone resolution-independent. Flatten
    hard-codes the training resolution into the classifier's weight shape;
    global average pooling produces a C-vector whatever H and W are, which is
    why every architecture since ResNet ends this way and why a modern detector
    accepts arbitrary input sizes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n, c, h, w = self.shape
        return np.broadcast_to(dout[:, :, None, None] / (h * w), self.shape).copy()
