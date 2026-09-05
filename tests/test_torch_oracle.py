"""PyTorch as an ORACLE, never as the thing that learns.

This is the only file in the repository that imports a deep learning framework,
and it imports it to disagree with us if we are wrong. Everything in
src/netfs/ is NumPy; autograd here computes a second opinion on gradients we
have already computed by hand, so a shared misconception between our forward
pass and our backward pass -- which a self-consistent finite-difference check
cannot detect, because it differentiates our own forward pass -- has somewhere
to show up.

That is a real gap and worth being precise about. If our conv forward pass is
wrong in some interesting way, the numerical gradient will faithfully report
the derivative of the wrong function and the gradient check will pass. Only an
independent implementation catches it. Hence three of them: scipy for
convolution (tests/test_conv.py), hand-computed answers for everything with
small enough arithmetic, and torch autograd here.

Everything is done in float64. PyTorch's default dtype is float32, and a
comparison at float32 will disagree in the sixth decimal for reasons that have
nothing to do with anyone's code.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional oracle, not a dependency")

from netfs.conv import Conv2D                      # noqa: E402
from netfs.gradcheck import relative_error         # noqa: E402
from netfs.layers import Flatten, Linear, ReLU     # noqa: E402
from netfs.losses import softmax_cross_entropy     # noqa: E402
from netfs.model import Sequential                 # noqa: E402
from netfs.pool import MaxPool2D                   # noqa: E402

TOL = 1e-10


def _t(a, requires_grad=False):
    return torch.tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64,
                        requires_grad=requires_grad)


def test_linear_against_autograd():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 4))
    layer = Linear(4, 3, rng=rng)
    g = rng.standard_normal((6, 3))

    out = layer.forward(x)
    dx = layer.backward(g)

    tx = _t(x, True)
    tw = _t(layer.params["W"], True)
    tb = _t(layer.params["b"], True)
    tout = torch.nn.functional.linear(tx, tw, tb)
    tout.backward(_t(g))

    assert relative_error(out, tout.detach().numpy()) < TOL
    assert relative_error(dx, tx.grad.numpy()) < TOL
    assert relative_error(layer.grads["W"], tw.grad.numpy()) < TOL
    assert relative_error(layer.grads["b"], tb.grad.numpy()) < TOL


@pytest.mark.parametrize("stride, pad", [(1, 0), (1, 1), (2, 1)])
def test_conv_against_autograd(stride, pad):
    """The layer this repository exists to justify, against the reference
    implementation of it.

    Our layout is deliberately PyTorch's -- NCHW input, (C_out, C_in, kh, kw)
    weights -- precisely so this comparison needs no transposes. A transpose
    here would be a place for the test to be wrong in the same way the code is.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 7, 7))
    layer = Conv2D(3, 4, 3, stride=stride, pad=pad, rng=rng)
    out = layer.forward(x)
    g = rng.standard_normal(out.shape)
    dx = layer.backward(g)

    tx = _t(x, True)
    tw = _t(layer.params["W"], True)
    tb = _t(layer.params["b"], True)
    tout = torch.nn.functional.conv2d(tx, tw, tb, stride=stride, padding=pad)
    tout.backward(_t(g))

    assert out.shape == tuple(tout.shape)
    assert relative_error(out, tout.detach().numpy()) < TOL
    assert relative_error(dx, tx.grad.numpy()) < TOL
    assert relative_error(layer.grads["W"], tw.grad.numpy()) < TOL
    assert relative_error(layer.grads["b"], tb.grad.numpy()) < TOL


def test_maxpool_against_autograd():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 6, 6))
    layer = MaxPool2D(2, 2)
    out = layer.forward(x)
    g = rng.standard_normal(out.shape)
    dx = layer.backward(g)

    tx = _t(x, True)
    tout = torch.nn.functional.max_pool2d(tx, 2, 2)
    tout.backward(_t(g))
    assert relative_error(out, tout.detach().numpy()) < TOL
    assert relative_error(dx, tx.grad.numpy()) < TOL


def test_softmax_cross_entropy_against_autograd():
    """torch's CrossEntropyLoss takes RAW logits and applies the fused
    log-softmax itself. Handing it softmax output instead is a common bug: it
    trains, badly, because a softmax of a softmax is still monotonic.
    """
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((8, 5)) * 2.0
    y = rng.integers(0, 5, size=8)
    loss, dlogits = softmax_cross_entropy(logits, y)

    tz = _t(logits, True)
    tloss = torch.nn.functional.cross_entropy(tz, torch.tensor(y, dtype=torch.long))
    tloss.backward()
    assert abs(loss - tloss.item()) < 1e-12
    assert relative_error(dlogits, tz.grad.numpy()) < TOL


def test_whole_cnn_against_autograd():
    """The full stack -- conv, relu, pool, conv, relu, pool, flatten, linear,
    cross-entropy -- one backward pass, every parameter compared.

    If this passes, the network trained in examples/09 is trained on gradients
    that an independent autograd engine agrees with to about 1e-13.
    """
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 1, 8, 8))
    y = np.array([0, 1, 2, 1])
    c1 = Conv2D(1, 3, 3, pad=1, rng=rng)
    c2 = Conv2D(3, 4, 3, pad=1, rng=rng)
    fc = Linear(16, 3, rng=rng)
    model = Sequential(c1, ReLU(), MaxPool2D(2), c2, ReLU(), MaxPool2D(2), Flatten(), fc)
    loss = model.backward_from_loss(softmax_cross_entropy, x, y)

    tx = _t(x)
    tw1, tb1 = _t(c1.params["W"], True), _t(c1.params["b"], True)
    tw2, tb2 = _t(c2.params["W"], True), _t(c2.params["b"], True)
    twf, tbf = _t(fc.params["W"], True), _t(fc.params["b"], True)
    h = torch.nn.functional.conv2d(tx, tw1, tb1, padding=1).relu()
    h = torch.nn.functional.max_pool2d(h, 2, 2)
    h = torch.nn.functional.conv2d(h, tw2, tb2, padding=1).relu()
    h = torch.nn.functional.max_pool2d(h, 2, 2)
    tout = torch.nn.functional.linear(h.reshape(h.shape[0], -1), twf, tbf)
    tloss = torch.nn.functional.cross_entropy(tout, torch.tensor(y, dtype=torch.long))
    tloss.backward()

    assert abs(loss - tloss.item()) < 1e-12
    pairs = [(c1, "W", tw1), (c1, "b", tb1), (c2, "W", tw2), (c2, "b", tb2),
             (fc, "W", twf), (fc, "b", tbf)]
    for layer, name, tensor in pairs:
        err = relative_error(layer.grads[name], tensor.grad.numpy())
        assert err < TOL, f"{type(layer).__name__}.{name}: {err:.2e}"


def test_torch_defaults_to_float32_and_that_matters():
    """Named here because it wastes an evening otherwise: PyTorch's default
    dtype is float32, NumPy's is float64. Compare a float64 hand-written
    gradient against a float32 autograd gradient and you get disagreement
    around 1e-7 on flawless code -- which is exactly the threshold people set
    for "is my gradient right".
    """
    assert torch.get_default_dtype() == torch.float32
    x32 = torch.tensor([0.1, 0.2, 0.3])
    assert x32.dtype == torch.float32
    x64 = np.array([0.1, 0.2, 0.3])
    assert x64.dtype == np.float64
    gap = float(np.abs(x32.numpy().astype(np.float64) - x64).max())
    # ~1.2e-08 for 0.3: float32 carries about 7 decimal digits, so the
    # rounding error is of the order of the threshold people set for "is my
    # gradient right", which is exactly why that comparison misleads.
    assert 0 < gap < 1e-7
