"""Analytic gradients against central-difference numerical gradients, for every
layer in the package. This is the most important test file here.

A hand-written backward pass that is subtly wrong does not raise. It trains --
badly, or slowly, or to a worse optimum -- and you blame the learning rate. The
only way to know a backward pass is right is to measure the derivative
independently and compare.

The tolerances below are the ones actually achieved on this machine, not
aspirations: every layer agrees to better than 1e-9 relative, and most to 1e-11.
The looser 1e-7 asserted here is deliberate headroom for a different BLAS or a
different summation order on someone else's CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from netfs.conv import Conv2D
from netfs.gradcheck import check_layer, check_model, numerical_gradient, relative_error
from netfs.layers import Flatten, Linear, ReLU, Sigmoid, Tanh
from netfs.losses import mse_loss, softmax_cross_entropy
from netfs.model import Sequential
from netfs.pool import GlobalAvgPool2D, MaxPool2D

TOL = 1e-7


def _rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------- dense layers
def test_linear_gradients():
    x = _rng(1).standard_normal((5, 4))
    errs = check_layer(Linear(4, 3, rng=_rng(2)), x)
    assert set(errs) == {"W", "b", "input"}
    for name, e in errs.items():
        assert e < TOL, f"{name}: {e:.2e}"


def test_linear_without_bias_has_no_bias_gradient():
    x = _rng(3).standard_normal((6, 5))
    layer = Linear(5, 2, rng=_rng(4), bias=False)
    errs = check_layer(layer, x)
    assert "b" not in errs
    assert errs["W"] < TOL and errs["input"] < TOL


@pytest.mark.parametrize("cls", [ReLU, Sigmoid, Tanh])
def test_activation_gradients(cls):
    # Offset away from zero for ReLU: a perturbation of size h that flips a
    # ReLU across its kink makes the numeric gradient measure a chord over the
    # kink, and it disagrees with the analytic gradient for a good reason
    # rather than a bug. Real gradient checks re-seed and look for rows that
    # MOVE (kink) versus rows that stay put (bug).
    x = _rng(5).standard_normal((4, 6)) + 0.5
    x[np.abs(x) < 1e-2] += 0.5
    errs = check_layer(cls(), x)
    assert errs["input"] < TOL, f"{cls.__name__}: {errs['input']:.2e}"


def test_relu_kink_is_a_kink_and_not_a_bug():
    """Deliberately place a pre-activation exactly on the kink and show the
    disagreement, so the failure mode is documented rather than mysterious.

    At z = 0 the analytic derivative is 0 (NumPy's `z > 0`, and PyTorch agrees).
    The centred difference measures (relu(h) - relu(-h))/(2h) = 0.5. That is
    not a bug in either one; the function has no derivative there.
    """
    layer = ReLU()
    x = np.array([[0.0]])
    layer.forward(x)
    analytic = layer.backward(np.ones((1, 1)))[0, 0]
    numeric = numerical_gradient(lambda: float(layer.forward(x).sum()), x)[0, 0]
    assert analytic == 0.0
    assert numeric == pytest.approx(0.5)


def test_flatten_gradients():
    x = _rng(6).standard_normal((3, 2, 4, 5))
    errs = check_layer(Flatten(), x)
    assert errs["input"] < TOL


# ---------------------------------------------------------------- convolution
@pytest.mark.parametrize("stride, pad", [(1, 0), (1, 1), (2, 1), (2, 0)])
def test_conv_gradients(stride, pad):
    x = _rng(7).standard_normal((2, 3, 7, 7))
    layer = Conv2D(3, 4, 3, stride=stride, pad=pad, rng=_rng(8))
    errs = check_layer(layer, x)
    for name, e in errs.items():
        assert e < TOL, f"conv(stride={stride}, pad={pad}) {name}: {e:.2e}"


def test_conv_input_gradient_accumulates_over_overlapping_windows():
    """The bug this test exists for: col2im scattering with `+=` on a fancy
    index instead of np.add.at.

    An interior pixel of a 3x3 stride-1 convolution is read by nine output
    positions, so its gradient is a sum of nine contributions. Buffered
    fancy-index assignment would keep one of them and drop eight, silently, and
    only in the interior -- the border, where fewer windows overlap, would look
    fine. Here the check is direct: with an all-ones upstream gradient and an
    all-ones 3x3 kernel, dx at an interior pixel must be exactly 9.
    """
    layer = Conv2D(1, 1, 3, pad=1, rng=_rng(9), bias=False)
    layer.params["W"][:] = 1.0
    x = np.zeros((1, 1, 5, 5))
    layer.forward(x)
    dx = layer.backward(np.ones((1, 1, 5, 5)))
    assert dx[0, 0, 2, 2] == 9.0     # interior: nine windows saw it
    assert dx[0, 0, 0, 0] == 4.0     # corner: only four, the rest fell in the padding


# -------------------------------------------------------------------- pooling
@pytest.mark.parametrize("k, stride", [(2, 2), (3, 1), (2, 1)])
def test_maxpool_gradients(k, stride):
    # No ties: exact ties would send the whole gradient to whichever entry
    # argmax picked first, and the numeric gradient would then measure a kink
    # for the same reason as ReLU at zero. Continuous random data has ties with
    # probability zero.
    x = _rng(10).standard_normal((2, 3, 6, 6))
    errs = check_layer(MaxPool2D(k, stride), x)
    assert errs["input"] < TOL, f"maxpool(k={k}, s={stride}): {errs['input']:.2e}"


def test_global_avg_pool_gradients():
    x = _rng(11).standard_normal((3, 4, 5, 5))
    errs = check_layer(GlobalAvgPool2D(), x)
    assert errs["input"] < TOL


# --------------------------------------------------------- losses, end to end
def test_mse_gradient():
    pred = _rng(12).standard_normal((4, 3))
    target = _rng(13).standard_normal((4, 3))
    _, dpred = mse_loss(pred, target)
    num = numerical_gradient(lambda: mse_loss(pred, target)[0], pred)
    assert relative_error(dpred, num) < TOL


def test_softmax_cross_entropy_gradient():
    logits = _rng(14).standard_normal((6, 5)) * 3.0
    y = _rng(15).integers(0, 5, size=6)
    _, dlogits = softmax_cross_entropy(logits, y)
    num = numerical_gradient(lambda: softmax_cross_entropy(logits, y)[0], logits)
    assert relative_error(dlogits, num) < TOL


def test_softmax_cross_entropy_gradient_rows_sum_to_zero():
    """A free correctness check that holds for every input, forever: softmax
    outputs sum to 1 and the one-hot sums to 1, so their difference sums to 0.
    The gradient can only move probability mass between classes, never create
    it.
    """
    logits = _rng(16).standard_normal((7, 4)) * 5.0
    y = _rng(17).integers(0, 4, size=7)
    _, dlogits = softmax_cross_entropy(logits, y, reduction="sum")
    assert np.abs(dlogits.sum(axis=1)).max() < 1e-14


def test_full_mlp_gradients():
    """Every parameter of a two-layer MLP, against its real loss."""
    model = Sequential(Linear(4, 6, rng=_rng(18)), ReLU(), Linear(6, 3, rng=_rng(19)))
    x = _rng(20).standard_normal((8, 4))
    y = _rng(21).integers(0, 3, size=8)
    errs = check_model(model, softmax_cross_entropy, x, y)
    assert len(errs) == 4                            # W and b for each Linear
    for name, e in errs.items():
        assert e < TOL, f"{name}: {e:.2e}"


def test_full_cnn_gradients():
    """The whole convolutional stack -- conv, relu, pool, conv, relu, pool,
    flatten, linear -- checked against its cross-entropy loss.

    This is the test that says the CNN in example 09 is training on real
    gradients and not on plausible-looking noise.
    """
    model = Sequential(
        Conv2D(1, 3, 3, pad=1, rng=_rng(22)), ReLU(), MaxPool2D(2),
        Conv2D(3, 4, 3, pad=1, rng=_rng(23)), ReLU(), MaxPool2D(2),
        Flatten(), Linear(4 * 2 * 2, 3, rng=_rng(24)),
    )
    x = _rng(25).standard_normal((3, 1, 8, 8))
    y = np.array([0, 2, 1])
    errs = check_model(model, softmax_cross_entropy, x, y)
    assert len(errs) == 6
    for name, e in errs.items():
        assert e < TOL, f"{name}: {e:.2e}"


def test_relative_error_treats_matching_zeros_as_agreement():
    """A dead ReLU produces exact zeros on both sides. That is agreement, and
    the naive relative formula would produce 0/0 = nan and fail the run.
    """
    a = np.array([0.0, 1.0])
    assert relative_error(a, a) == 0.0
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0


def test_step_size_too_small_is_worse_than_too_large():
    """Catastrophic cancellation, demonstrated rather than asserted in prose.

    Shrinking h shrinks the O(h^2) truncation error and grows the
    floating-point cancellation error. Somewhere around 1e-6 the second term
    takes over, and by 1e-10 the difference (L+ - L-) has lost essentially
    every significant digit. If your gradient check gets WORSE as you shrink h,
    that is what you are looking at -- stop shrinking.
    """
    logits = _rng(26).standard_normal((3, 4))
    y = np.array([0, 1, 2])
    _, analytic = softmax_cross_entropy(logits, y)
    err = {}
    for h in (1e-5, 1e-10):
        num = numerical_gradient(lambda: softmax_cross_entropy(logits, y)[0], logits, h=h)
        err[h] = relative_error(analytic, num)
    assert err[1e-5] < 1e-9
    assert err[1e-10] > err[1e-5] * 100


def test_elementwise_relative_error_is_stricter_on_a_wide_dynamic_range():
    """The two readings of "relative error", and why the default is the one it
    is.

    Take a gradient array whose largest entry is 1.0 and whose smallest is
    1e-9, and perturb the small entry by the noise floor of a float64 central
    difference. Measured against the array's scale the disagreement is
    negligible; measured entry by entry it reads 1e-2 and looks like a
    catastrophe. Neither number is wrong -- they answer different questions,
    and only the first one is about whether training will work.
    """
    a = np.array([1.0, 1e-9])
    b = np.array([1.0, 1e-9 + 1e-11])
    assert relative_error(a, b) == pytest.approx(1e-11, rel=1e-6)
    assert relative_error(a, b, elementwise=True) == pytest.approx(1e-2, rel=1e-2)


def test_every_layer_agrees_to_1e_9():
    """The headline number, asserted rather than claimed.

    One table, every layer in the package, worst relative error over
    parameters and input. The threshold here is 1e-9; the errors actually
    achieved on the machine this was developed on are all between 1e-12 and
    1e-10, and they are printed by examples/05_gradient_check.py.
    """
    checks = {
        "Linear": (Linear(4, 3, rng=_rng(30)), _rng(31).standard_normal((5, 4))),
        "ReLU": (ReLU(), _rng(32).standard_normal((4, 5)) + 1.5),
        "Sigmoid": (Sigmoid(), _rng(33).standard_normal((4, 5))),
        "Tanh": (Tanh(), _rng(34).standard_normal((4, 5))),
        "Flatten": (Flatten(), _rng(35).standard_normal((2, 3, 4, 4))),
        "Conv2D": (Conv2D(2, 3, 3, pad=1, rng=_rng(36)), _rng(37).standard_normal((2, 2, 6, 6))),
        "MaxPool2D": (MaxPool2D(2), _rng(38).standard_normal((2, 3, 6, 6))),
        "GlobalAvgPool2D": (GlobalAvgPool2D(), _rng(39).standard_normal((2, 3, 5, 5))),
    }
    worst = {}
    for name, (layer, x) in checks.items():
        worst[name] = max(check_layer(layer, x).values())
    for name, e in worst.items():
        assert e < 1e-9, f"{name}: {e:.2e}"
