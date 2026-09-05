"""Convolution as a layer: the honest loop version, the im2col version, and
the backward pass for both the input and the kernel.

LAYOUT -- NCHW, i.e. x is (N, C_in, H, W) and the weights are
(C_out, C_in, kh, kw). That is PyTorch's layout, chosen deliberately so the
cross-check in tests/test_torch_oracle.py is a direct comparison with no
transposes for either side to get wrong. The cost is that images arriving from
OpenCV or matplotlib are HWC and have to be transposed at the boundary; that
transpose is a named seam (netfs.data does it once) and not something sprinkled
through the code.

What every framework calls "convolution" is CROSS-CORRELATION: the kernel is
not flipped. Note the sign of the index below -- `i*stride + a`, both running
forward. Mathematical convolution uses `x[i - a, j - b]`. cv2.filter2D and
scipy.signal.correlate2d also correlate; scipy.signal.convolve2d genuinely
flips, and examples/07_convolution.py prints the two side by side so you can
watch every sign change. It is irrelevant for LEARNED filters -- backprop will
happily learn the flipped kernel -- and it matters enormously the moment you
port a hand-designed kernel (a Sobel, say) into a network.
"""

from __future__ import annotations

import numpy as np

from .layers import Layer
from .shapes import conv_out_size


def conv2d_naive(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None,
                 stride: int = 1, pad: int = 0) -> np.ndarray:
    """Convolution written as the four nested loops it is defined as.

        out[n, f, i, j] = sum_c sum_a sum_bb  xp[n, c, i*s + a, j*s + bb] * w[f, c, a, bb]

    Slow and unambiguous. It is the reference the fast version is asserted
    against (tests/test_conv.py), and it is the thing you read when you want to
    know what convolution *is* rather than how it is computed.

    Note the two sums that are NOT loops in the body: the patch multiply sums
    over the channel axis as well as over kh and kw. A "3x3 filter" applied to
    a 64-channel input is a 3x3x64 tensor and the channel dimension is consumed
    entirely, in one shot. The kernel slides over height and width only. Believe
    the sliding-in-depth story and every parameter count you produce is wrong.
    """
    n, c_in, h, wd = x.shape
    c_out, c_in_w, kh, kw = w.shape
    if c_in_w != c_in:
        raise ValueError(f"channel mismatch: input has {c_in}, kernel expects {c_in_w}")
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho = conv_out_size(h, kh, pad, stride)
    wo = conv_out_size(wd, kw, pad, stride)
    out = np.zeros((n, c_out, ho, wo), dtype=np.float64)
    for i in range(ho):
        for j in range(wo):
            r0, c0 = i * stride, j * stride
            patch = xp[:, :, r0:r0 + kh, c0:c0 + kw]          # (N, C_in, kh, kw)
            # einsum instead of a fifth and sixth loop over (n, f): still the
            # naive algorithm -- one pass per output position -- but without
            # paying Python interpreter overhead per sample.
            out[:, :, i, j] = np.einsum("nchw,fchw->nf", patch, w)
    return out if b is None else out + b.reshape(1, -1, 1, 1)


def im2col_indices(c_in: int, kh: int, kw: int, ho: int, wo: int, stride: int):
    """Address arithmetic for im2col, returned as three index arrays of shape
    (ho*wo, c_in*kh*kw) that address the PADDED input.

    Row r of the output matrix is output pixel (r // wo, r % wo). Column f
    within that row is one element of the patch, and the flat position f
    decomposes in (c, a, b) ravel order -- the SAME order `w.reshape(c_out, -1)`
    uses, which is the entire reason the GEMM below is correct:

        c = f // (kh*kw)        which input channel
        a = (f // kw) % kh      which kernel row
        b = f % kw              which kernel column

    Getting that decomposition out of step with the weight reshape is the
    classic im2col bug: no exception, no shape error, just a convolution that
    computes a permuted kernel. tests/test_conv.py catches it by asserting the
    im2col result equals the naive loop exactly.
    """
    f = np.arange(c_in * kh * kw)
    cc = f // (kh * kw)
    aa = (f // kw) % kh
    bb = f % kw
    r = np.arange(ho * wo)
    ii = (r // wo)[:, None] * stride + aa[None, :]
    jj = (r % wo)[:, None] * stride + bb[None, :]
    cc = np.broadcast_to(cc[None, :], ii.shape)
    return cc, ii, jj


def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0):
    """Pull out every patch the kernel will ever see and stack them as rows.

    Returns `cols` of shape (N*ho*wo, c_in*kh*kw), plus ho and wo.

    This is one fancy-index gather. It is also where the memory goes: each
    input element is copied once per output position that reads it, so the
    matrix is roughly k^2 times the size of the activation. A 224x224x64
    float32 activation is 12.8 MB; its 3x3 im2col matrix is 50176 x 576 = 116
    MB, a 9x blow-up. That is why real frameworks do not always use im2col --
    they keep direct, Winograd and FFT paths and choose per shape -- and why a
    MemoryError can appear on an input the naive loop handled comfortably.
    """
    n, c_in, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho = conv_out_size(h, kh, pad, stride)
    wo = conv_out_size(wd, kw, pad, stride)
    cc, ii, jj = im2col_indices(c_in, kh, kw, ho, wo, stride)
    cols = xp[:, cc, ii, jj]                 # (N, ho*wo, c_in*kh*kw)
    return cols.reshape(n * ho * wo, -1), ho, wo


def col2im(cols: np.ndarray, x_shape, kh: int, kw: int, stride: int = 1, pad: int = 0):
    """The transpose of im2col: scatter-ADD every patch element back to where
    it came from, then strip the padding.

    This is not a cosmetic inverse. It is exactly the backward pass of the
    gather, and the reason it is an *add* is the single most important rule in
    backprop: an input pixel that was read by nine different output positions
    is a multi-path variable, so its gradient is the SUM of the nine
    contributions.

    `np.add.at` and not `dxp[..., ii, jj] += cols`. Buffered fancy-index
    assignment with repeated indices -- which every overlapping window
    guarantees -- applies ONE of the duplicate updates and silently discards
    the others. There is no warning. The result is a gradient that is wrong
    only in the interior of the image, only where windows overlap, which is a
    magnificent bug to try to find by reading code. It is caught here by the
    gradient check, which is exactly what the gradient check is for.
    `np.add.at` is unbuffered and therefore correct, at a real speed cost.
    """
    n, c_in, h, wd = x_shape
    ho = conv_out_size(h, kh, pad, stride)
    wo = conv_out_size(wd, kw, pad, stride)
    cc, ii, jj = im2col_indices(c_in, kh, kw, ho, wo, stride)
    dxp = np.zeros((n, c_in, h + 2 * pad, wd + 2 * pad), dtype=np.float64)
    nn = np.arange(n)[:, None, None]
    np.add.at(dxp, (nn, cc[None], ii[None], jj[None]), cols.reshape(n, ho * wo, -1))
    if pad == 0:
        return dxp
    # The padded border received gradient too. It is discarded, correctly: those
    # entries are constants we invented, not inputs anyone can change.
    return dxp[:, :, pad:pad + h, pad:pad + wd]


def conv2d_im2col(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None,
                  stride: int = 1, pad: int = 0) -> np.ndarray:
    """Convolution as one matrix multiply.

        out = cols @ w.reshape(c_out, -1).T   +   b

    (N*ho*wo, K) @ (K, c_out) -> (N*ho*wo, c_out), then reshaped back to NCHW.

    Every output element is a dot product of a patch with a filter. im2col just
    arranges those dot products so they can all be done by one call into a
    tuned GEMM kernel, which is how convolution ends up being fast: it is not
    a special algorithm, it is matrix multiplication with the data rearranged.
    """
    c_out = w.shape[0]
    kh, kw = w.shape[2], w.shape[3]
    cols, ho, wo = im2col(x, kh, kw, stride, pad)
    out = cols @ w.reshape(c_out, -1).T
    if b is not None:
        out = out + b
    n = x.shape[0]
    return out.reshape(n, ho, wo, c_out).transpose(0, 3, 1, 2)


class Conv2D(Layer):
    """A convolution layer with a hand-derived backward pass.

    Forward:   out[n,f,i,j] = sum_{c,a,b} xp[n,c,i*s+a,j*s+b] * W[f,c,a,b] + b[f]

    Backward -- three gradients, all of them the same multiply-swap rule seen
    through im2col's rearrangement (full derivation in DERIVATIONS.md s7):

        dW    = dOut_flat.T @ cols     (c_out, K) -> reshape to W's shape
        dcols = dOut_flat @ Wr         (N*ho*wo, K)
        dX    = col2im(dcols)          scatter-add back, strip padding
        db    = dOut.sum over (n, i, j)

    dW sums over every position the filter visited, which is the backward
    statement of weight sharing: one filter, used at 4096 positions, collects
    4096 contributions to its gradient. That is also why conv layers train on
    far less data than their dense equivalents -- each weight gets many more
    gradient samples per image.

    db sums over the batch AND both spatial axes for the same reason: the same
    scalar bias was added at every position of every sample.
    """

    def __init__(self, c_in: int, c_out: int, k: int, *, stride: int = 1, pad: int = 0,
                 rng=None, bias: bool = True, weight_scale: float | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(0) if rng is None else rng
        fan_in = c_in * k * k          # NOT c_in: each output reads k*k*c_in numbers
        scale = np.sqrt(2.0 / fan_in) if weight_scale is None else weight_scale
        self.params["W"] = (rng.standard_normal((c_out, c_in, k, k)) * scale).astype(np.float64)
        if bias:
            self.params["b"] = np.zeros(c_out, dtype=np.float64)
        self.stride, self.pad, self.k = stride, pad, k
        self.cols: np.ndarray | None = None
        self.x_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        w = self.params["W"]
        c_out, _, kh, kw = w.shape
        # Cache the COLUMNS, not the input. dW needs cols and dX needs only
        # dcols, so keeping the (much larger) column matrix saves recomputing
        # the gather on the backward pass -- the standard time/memory trade
        # every framework makes, and the reason a conv layer's activation
        # memory during training is several times its output size.
        self.cols, ho, wo = im2col(x, kh, kw, self.stride, self.pad)
        self.x_shape = x.shape
        out = self.cols @ w.reshape(c_out, -1).T
        if "b" in self.params:
            out = out + self.params["b"]
        return out.reshape(x.shape[0], ho, wo, c_out).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        w = self.params["W"]
        c_out = w.shape[0]
        # (N, c_out, ho, wo) -> (N*ho*wo, c_out): the exact inverse of the
        # reshape+transpose the forward pass ended with. Do it in the wrong
        # order and the gradient is a permutation of the right answer, which
        # the gradient check catches instantly and reading the code does not.
        dflat = dout.transpose(0, 2, 3, 1).reshape(-1, c_out)
        self._store("W", (dflat.T @ self.cols).reshape(w.shape))
        if "b" in self.params:
            self._store("b", dflat.sum(axis=0))
        dcols = dflat @ w.reshape(c_out, -1)
        return col2im(dcols, self.x_shape, self.k, self.k, self.stride, self.pad)
