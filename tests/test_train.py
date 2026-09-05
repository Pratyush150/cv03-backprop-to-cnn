"""End-to-end behaviour: things that must be true of a network that learns.

These are slower than the unit tests and they are the ones that would catch a
gradient that is correct in isolation and wired up backwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from netfs.conv import Conv2D
from netfs.data import load_image_dataset, two_moons, xor_dataset
from netfs.layers import Flatten, Linear, ReLU
from netfs.losses import accuracy, mse_loss, softmax_cross_entropy
from netfs.model import Sequential
from netfs.optim import SGD, Adam
from netfs.pool import MaxPool2D
from netfs.train import evaluate, train


def _fit_xor(with_relu: bool, steps: int = 4000, lr: float = 0.05, seed: int = 0):
    x, y = xor_dataset()
    rng = np.random.default_rng(seed)
    layers = [Linear(2, 8, rng=rng)]
    if with_relu:
        layers.append(ReLU())
    layers.append(Linear(8, 1, rng=rng))
    model = Sequential(*layers)
    opt = SGD(model, lr=lr)
    for _ in range(steps):
        model.backward_from_loss(mse_loss, x, y)
        opt.step()
    return mse_loss(model.forward(x), y)[0]


def test_xor_fails_without_a_nonlinearity_and_succeeds_with_one():
    """The failure IS the lesson, so it is asserted, not described.

    Two linear layers with nothing between them are algebraically one linear
    layer, and a single linear function of two binary inputs cannot produce
    0, 1, 1, 0. Gradient descent converges perfectly well -- to the best line,
    which gets exactly two of the four corners right and floors the mean
    squared error at 0.25. Insert one ReLU between the same two layers and the
    same optimiser on the same data drives it to essentially zero.

    0.25 is not an empirical observation. Any linear f on the four corners
    satisfies f(0,0) + f(1,1) = f(0,1) + f(1,0), while the XOR targets give
    0 + 0 = 0 on the left and 1 + 1 = 2 on the right. The best it can do is
    split that gap of 2 evenly across the four points: an error of 1/2 each,
    so an MSE of 4*(1/2)^2/4 = 0.25.
    """
    linear_loss = _fit_xor(with_relu=False)
    relu_loss = _fit_xor(with_relu=True)
    assert linear_loss == pytest.approx(0.25, abs=0.01)
    assert relu_loss < 1e-3
    assert linear_loss > 100 * relu_loss


def test_extra_linear_capacity_does_not_help():
    """A wider or deeper linear stack is still one linear layer, so it floors
    at exactly the same 0.25. If depth were buying anything, this number would
    move.
    """
    x, y = xor_dataset()
    rng = np.random.default_rng(1)
    model = Sequential(Linear(2, 32, rng=rng), Linear(32, 32, rng=rng), Linear(32, 1, rng=rng))
    opt = SGD(model, lr=0.02)
    for _ in range(4000):
        model.backward_from_loss(mse_loss, x, y)
        opt.step()
    assert mse_loss(model.forward(x), y)[0] == pytest.approx(0.25, abs=0.01)


def test_learning_rate_regimes_on_a_quadratic():
    """L(w) = (2w - 6)^2, whose minimum is at w = 3 and whose curvature is 8.

    Descent multiplies the distance to the optimum by (1 - lr*8) every step, so:
    below 0.125 it approaches monotonically, at 0.125 it lands exactly in one
    step, between 0.125 and 0.25 it overshoots and alternates, at 0.25 it
    orbits forever, and above 0.25 it diverges geometrically.

    The 0.25 row is the trap worth knowing: L(0) = L(6) = 36, so the printed
    loss reads 36.0000 on every single step and never moves while w swings by
    6 each iteration. "My loss is flat" does not mean "nothing is happening".
    """
    def run(lr, steps=6):
        w, hist = 0.0, []
        for _ in range(steps):
            w = w - lr * (8 * w - 24)
            hist.append(w)
        return hist

    assert run(0.125)[0] == pytest.approx(3.0)
    assert all(abs(v - 3.0) == pytest.approx(0.0) for v in run(0.125))
    approach = run(0.05)
    assert approach == pytest.approx([1.2, 1.92, 2.352, 2.6112, 2.7667, 2.86], abs=1e-3)
    overshoot = run(0.2)
    assert overshoot == pytest.approx([4.8, 1.92, 3.648, 2.6112, 3.2333, 2.86], abs=1e-3)
    # lr = 0.05 and lr = 0.2 have contraction factors +0.6 and -0.6, so their
    # LOSS sequences are identical to machine precision while their weight
    # trajectories are completely different -- one approaches from below, the
    # other alternates sides. The loss curve does not tell you what the weights
    # are doing.
    loss_slow = [(2 * w - 6) ** 2 for w in approach]
    loss_fast = [(2 * w - 6) ** 2 for w in overshoot]
    assert loss_slow == pytest.approx(loss_fast, rel=1e-12)
    orbit = run(0.25)
    assert orbit == pytest.approx([6, 0, 6, 0, 6, 0])
    losses = [(2 * w - 6) ** 2 for w in [0.0] + orbit[:-1]]
    assert losses == pytest.approx([36.0] * 6)          # identical every step
    diverge = run(0.3)
    assert abs(diverge[-1]) > abs(diverge[0]) * 2


def test_mlp_separates_two_moons():
    """A curved boundary, which one linear layer cannot draw at all."""
    x, y = two_moons(400, noise=0.12, rng=np.random.default_rng(0))
    rng = np.random.default_rng(2)
    model = Sequential(Linear(2, 16, rng=rng), ReLU(), Linear(16, 2, rng=rng))
    opt = Adam(model, lr=0.02)
    hist = train(model, softmax_cross_entropy, opt, x, y, epochs=40, batch_size=32,
                 verbose=False)
    assert hist["train_loss"][-1] < hist["train_loss"][0] / 4
    assert accuracy(model.forward(x), y) > 0.98


def test_cnn_can_overfit_a_tiny_batch():
    """The first thing to do with any new architecture: check it can drive the
    loss to zero on a handful of samples.

    If a network cannot memorise twelve images it will certainly not
    generalise on twelve hundred, and the bug is in the model or the gradients
    rather than in the data or the schedule. It is the cheapest diagnostic
    there is and almost nobody runs it before spending an afternoon on a
    learning rate.
    """
    data = load_image_dataset(seed=0)
    x, y = data.x_train[:12], data.y_train[:12]
    rng = np.random.default_rng(3)
    model = Sequential(
        Conv2D(1, 8, 3, pad=1, rng=rng), ReLU(), MaxPool2D(2),
        Flatten(), Linear(8 * 4 * 4, len(data.class_names), rng=rng),
    )
    opt = Adam(model, lr=0.01)
    for _ in range(120):
        model.backward_from_loss(softmax_cross_entropy, x, y)
        opt.step()
    assert accuracy(model.forward(x), y) == 1.0
    assert softmax_cross_entropy(model.forward(x), y)[0] < 0.01


def test_initial_loss_is_log_c_on_real_data():
    """The prediction you make before pressing go. An untrained ten-class
    classifier must start at about ln(10) = 2.303; a first loss far from it
    means the labels or the normalisation are wrong, not the architecture.
    """
    data = load_image_dataset(seed=0)
    rng = np.random.default_rng(4)
    c = len(data.class_names)
    model = Sequential(Flatten(), Linear(data.x_train[0].size, c, rng=rng, weight_scale=1e-3))
    loss, _ = softmax_cross_entropy(model.forward(data.x_train[:64]), data.y_train[:64])
    assert loss == pytest.approx(np.log(c), abs=0.05)


def test_momentum_and_adam_both_reduce_the_loss():
    x, y = two_moons(200, rng=np.random.default_rng(5))
    for opt_cls, kwargs in [(SGD, {"lr": 0.1}), (SGD, {"lr": 0.1, "momentum": 0.9}),
                            (Adam, {"lr": 0.01})]:
        rng = np.random.default_rng(6)
        model = Sequential(Linear(2, 8, rng=rng), ReLU(), Linear(8, 2, rng=rng))
        opt = opt_cls(model, **kwargs)
        first = None
        for _ in range(200):
            loss = model.backward_from_loss(softmax_cross_entropy, x, y)
            first = loss if first is None else first
            opt.step()
        assert loss < first / 2, f"{opt_cls.__name__} {kwargs}"


def test_optimiser_updates_parameters_in_place():
    """`p -= lr * g` and not `p = p - lr * g`. The second rebinds a local name,
    leaves the layer holding the old array, trains nothing, and raises nothing.
    """
    rng = np.random.default_rng(7)
    model = Sequential(Linear(3, 2, rng=rng))
    w = model.layers[0].params["W"]
    before = w.copy()
    model.backward_from_loss(mse_loss, rng.standard_normal((4, 3)), rng.standard_normal((4, 2)))
    SGD(model, lr=0.1).step()
    assert w is model.layers[0].params["W"]       # same object
    assert not np.array_equal(w, before)          # and it moved


def test_evaluate_matches_a_direct_computation():
    data = load_image_dataset(seed=0)
    rng = np.random.default_rng(8)
    model = Sequential(Flatten(), Linear(data.x_train[0].size, len(data.class_names), rng=rng))
    loss, acc = evaluate(model, softmax_cross_entropy, data.x_test, data.y_test)
    out = model.forward(data.x_test)
    assert acc == pytest.approx(accuracy(out, data.y_test))
    assert loss == pytest.approx(softmax_cross_entropy(out, data.y_test)[0], rel=1e-9)


def test_synthetic_fallback_dataset_is_usable():
    """The no-scikit-learn path has to work, because CI machines without it
    exist and a fallback nobody exercises is not a fallback.
    """
    data = load_image_dataset(force_synthetic=True, seed=0)
    assert data.x_train.shape[1:] == (1, 12, 12)
    assert set(np.unique(data.y_train)) == {0, 1, 2}
    assert "synthetic" in data.source.lower()
