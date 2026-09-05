"""Dense layers and activations, forward and backward, in NumPy.

Every gradient in this file is derived by hand in docs/DERIVATIONS.md and
checked against a central-difference numerical gradient in
tests/test_gradients.py. If you change a line here and the derivation in the
docs no longer matches it, the docs are the bug report.

CONVENTION -- fixed once, here, and never mixed (this is the single most
expensive mistake in hand-written nets, because NumPy broadcasting turns it
into a silently wrong answer rather than an exception):

    batch first, row-major.  X has shape (N, D_in), one SAMPLE PER ROW.
    W has shape (D_out, D_in), one NEURON PER ROW.
    forward is  Z = X @ W.T + b,  which is exactly torch.nn.Linear's layout.

The single-sample textbook form `z = W x + b` is the same algebra with the
batch axis deleted. Both are correct; writing one line of a derivation in each
is how you get an (N, N) matrix where you wanted (N, D_out) and no error
message to tell you.
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Base class: a forward pass that caches, and a backward pass that uses
    the cache.

    The cache is not an optimisation, it is a requirement. The backward pass of
    every layer below needs a value the forward pass computed -- the input, the
    pre-activation, the argmax positions. That is the mechanical reason
    training uses far more memory than inference, and the reason gradient
    checkpointing (recompute instead of store) exists as a trade.
    """

    def __init__(self) -> None:
        # name -> array. Kept in a dict, not as attributes, so an optimiser and
        # the gradient checker can walk any layer generically without knowing
        # what kind of layer it is.
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def _store(self, name: str, grad: np.ndarray) -> None:
        """Record a parameter gradient, asserting the one invariant that
        catches most backward-pass bugs before they can do damage:

            dL/dW has exactly W's shape, always.

        There is one partial derivative per entry of W, so the gradient is the
        same shape as the thing. If your derivation handed you the transpose,
        you multiplied in the wrong order. In the single-sample case a wrong
        shape raises immediately; in the batched case NumPy broadcasts and
        produces a numerically wrong gradient with no exception at all, which
        is why this is an assert and not a comment.
        """
        expected = self.params[name].shape
        if grad.shape != expected:
            raise ValueError(f"grad for {name!r} has shape {grad.shape}, expected {expected}")
        self.grads[name] = grad


class Linear(Layer):
    """Fully connected layer:  Z = X @ W.T + b.

    Backward (derived term by term in docs/DERIVATIONS.md section 4):

        dX = dZ @ W          (N, D_out) @ (D_out, D_in) -> (N, D_in)   = X's shape
        dW = dZ.T @ X        (D_out, N) @ (N, D_in)     -> (D_out, D_in) = W's shape
        db = dZ.sum(axis=0)  collapse the batch axis    -> (D_out,)      = b's shape

    Read those three lines as one rule applied three times: the gradient
    flowing into an input of a multiply is the upstream gradient times *the
    other* input. The only work is arranging the transposes so the shapes come
    out right, and the shapes are forced -- there is exactly one way to
    multiply (N, D_out) and (N, D_in) into (D_out, D_in).

    `db` is a SUM and not a mean: the same bias vector is added to every row of
    the batch, so it is a multi-path variable and multi-path gradients add. Get
    this wrong by leaving off `.sum(axis=0)` and `db` keeps shape (N, D_out),
    which then broadcasts during the update and corrupts the bias into a matrix
    -- the parameter silently changes shape between iterations.
    """

    def __init__(self, d_in: int, d_out: int, *, weight_scale: float | None = None, rng=None,
                 bias: bool = True) -> None:
        super().__init__()
        rng = np.random.default_rng(0) if rng is None else rng
        # He/Kaiming scaling: std = sqrt(2 / fan_in). The 2 is there because
        # ReLU zeroes half the units, halving the variance of the signal at
        # every layer; without the compensation activations shrink geometrically
        # with depth and the gradient shrinks with them. Xavier's sqrt(1/fan_in)
        # is the same argument for a symmetric activation like tanh.
        scale = np.sqrt(2.0 / d_in) if weight_scale is None else weight_scale
        # float64 everywhere. Gradient checking in float32 reports relative
        # errors around 1e-3 on flawless code, because the subtraction
        # (L_plus - L_minus) cancels away most of the mantissa. An evening lost
        # hunting a bug that does not exist is the standard price of not
        # reading this comment.
        self.params["W"] = (rng.standard_normal((d_out, d_in)) * scale).astype(np.float64)
        if bias:
            # Zero, not random. The weights already break the symmetry between
            # neurons; a random bias only adds noise to the start.
            self.params["b"] = np.zeros(d_out, dtype=np.float64)
        self.x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x  # needed by dW; this is the whole memory cost of a dense layer
        z = x @ self.params["W"].T
        if "b" in self.params:
            z = z + self.params["b"]  # broadcasts over the batch axis
        return z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self._store("W", dout.T @ self.x)
        if "b" in self.params:
            self._store("b", dout.sum(axis=0))
        return dout @ self.params["W"]


class ReLU(Layer):
    """f(z) = max(0, z), elementwise.

    `np.maximum(0, z)` and not `np.max(z)`. The second one REDUCES to a scalar.
    Nothing raises: the activation silently becomes a single float, the next
    layer's matmul still works by broadcasting, and the network returns
    confident garbage. Look for an activation printing as a bare `1.0` where
    you expected `[1. 0.]`.

    Backward: max is a SWITCH, not a splitter. The entire upstream gradient
    goes to the input if it won (z > 0) and exactly zero if it lost. A unit
    whose pre-activation is negative for every sample therefore receives
    exactly zero gradient -- not small, zero -- so no optimiser can move its
    incoming weights, so its pre-activation cannot change, so it can never
    revive. That is the dying-ReLU problem, and it is a structural property of
    this one line rather than bad luck.

    At exactly z = 0 the derivative is mathematically undefined. `(z > 0)`
    picks 0, which is what PyTorch also picks. It never matters in practice
    (hitting exactly 0.0 has measure zero) and it is asked in interviews.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0  # a bool array; storing the mask is cheaper than storing x
        return np.where(self.mask, x, 0.0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


class Sigmoid(Layer):
    """f(z) = 1 / (1 + exp(-z)), with the overflow branch that the textbook
    form does not have.

    `1/(1+np.exp(-z))` at z = -800 evaluates exp(800), which overflows to inf,
    warns, and returns 0.0. It is not a theoretical concern: feed a network raw
    [0, 255] pixel values and the first pre-activations are in the hundreds
    immediately. The stable form branches on the sign so the exponent is never
    positive:

        z >= 0:  1 / (1 + exp(-z))          exp of a non-positive number
        z <  0:  exp(z) / (1 + exp(z))      exp of a negative number

    Both are the same function; only one of them is computable.

    Backward: f'(z) = f(z) * (1 - f(z)), and the maximum of that is 0.25 at
    z = 0. Chain ten sigmoids and the gradient is multiplied by at most
    0.25^10 = 1e-6. That single number is the seed of the vanishing-gradient
    story and the reason ReLU, whose derivative is exactly 1 on the live half,
    took over.
    """

    def __init__(self) -> None:
        super().__init__()
        self.out: np.ndarray | None = None

    @staticmethod
    def _stable(z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = self._stable(np.asarray(x, dtype=np.float64))
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Cache the OUTPUT, not the input: the derivative is expressible in
        # terms of the output alone, so recomputing exp() here would be pure
        # waste. Every framework does the same.
        return dout * self.out * (1.0 - self.out)


class Tanh(Layer):
    """f(z) = tanh(z). Derivative 1 - tanh(z)^2, peaking at 1.0 rather than
    sigmoid's 0.25, and zero-centred.

    The zero-centring is the part that is not cosmetic. With a non-zero-centred
    activation feeding a layer, every weight gradient in that layer shares the
    sign of the upstream gradient (because dW = dZ.T @ X and every entry of X
    is positive), so the whole weight vector can only move into one diagonal
    direction at a time. That is the classic zig-zag descent path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (1.0 - self.out ** 2)


class Flatten(Layer):
    """(N, C, H, W) -> (N, C*H*W), and the reverse on the way back.

    Nothing is computed, so the backward pass is a pure reshape: the gradient
    of a reshape is the reshape of the gradient.

    What this layer costs is worth naming. Flattening before a classifier
    hard-codes the input resolution into the next Linear layer's shape: train
    at 8x8, feed 10x10, and you get `matmul: shapes (N, 400) and (10, 256)`.
    Global average pooling (netfs.pool.GlobalAvgPool2D) takes the mean over H
    and W instead and produces a C-vector at any resolution, which is why every
    backbone since ResNet uses it and why a modern detector accepts arbitrary
    input sizes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.shape)
