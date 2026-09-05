"""Losses, and the overflow that fusing softmax with cross-entropy prevents.

The headline test here is `test_fused_survives_where_naive_dies`: the two
implementations agree to machine precision on safe inputs, and on unsafe ones
the naive route returns nan while the fused route returns the right answer.
That pair of facts is what turns "they are fused for numerical stability" from
a slogan into a claim you have watched hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from netfs.losses import (accuracy, confusion_matrix, cross_entropy_unfused, log_sum_exp,
                          mse_loss, softmax, softmax_cross_entropy, softmax_naive)


def test_softmax_hand_computation():
    """Logits [2.0, 1.0, 0.1], worked with a calculator:
    e^2 = 7.3891, e^1 = 2.7183, e^0.1 = 1.1052, sum = 11.2125.
    """
    p = softmax(np.array([[2.0, 1.0, 0.1]]))[0]
    assert np.allclose(p, [0.6590, 0.2424, 0.0986], atol=1e-4)
    assert p.sum() == pytest.approx(1.0)


def test_cross_entropy_hand_computation():
    """-ln(0.6590) = 0.4170, and the gradient p - onehot = [-0.3410, 0.2424,
    0.0986], whose entries sum to exactly zero.
    """
    loss, grad = softmax_cross_entropy(np.array([[2.0, 1.0, 0.1]]), np.array([0]))
    assert loss == pytest.approx(0.4170, abs=1e-4)
    assert np.allclose(grad[0], [-0.3410, 0.2424, 0.0986], atol=1e-4)
    assert abs(grad.sum()) < 1e-15


def test_loss_at_initialisation_is_log_c():
    """The number you write down BEFORE a run starts.

    An untrained classifier spreads its confidence evenly, so every class gets
    1/C and the loss is -ln(1/C) = ln(C): 2.3026 for ten classes, 1.0986 for
    three. If the first loss of a real run is far from this, the bug is in the
    wiring -- labels, normalisation, or the loss being handed the wrong
    argument -- and it is worth finding before you spend eight epochs.
    """
    for c in (3, 10):
        logits = np.zeros((4, c))
        loss, _ = softmax_cross_entropy(logits, np.zeros(4, dtype=int))
        assert loss == pytest.approx(np.log(c))
    assert np.log(3) == pytest.approx(1.0986, abs=1e-4)
    assert np.log(10) == pytest.approx(2.3026, abs=1e-4)


def test_fused_and_naive_agree_on_safe_inputs():
    """On inputs the naive version can handle, the two are the same function
    to machine precision. Without this half of the pair, "the fused one is
    better" would just mean "the fused one is different".
    """
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((20, 6)) * 2.0
    y = rng.integers(0, 6, size=20)
    fused, _ = softmax_cross_entropy(logits, y)
    naive = cross_entropy_unfused(logits, y)
    assert abs(fused - naive) < 1e-12
    assert np.abs(softmax(logits) - softmax_naive(logits)).max() < 1e-15


def test_fused_survives_where_naive_dies():
    """Logits of 1000 -- not exotic; a badly initialised layer on unnormalised
    input reaches them in one step.

    exp(1000) overflows float64 (the limit is about 709.78), so the naive
    softmax is inf/inf = nan and the loss is nan. The fused loss is
    logsumexp(z) - z_y, which never exponentiates a large number and returns
    a correct, finite answer.
    """
    logits = np.array([[1000.0, 1001.0, 1002.0]])
    y = np.array([0])
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(softmax_naive(logits)).all()
        assert np.isnan(cross_entropy_unfused(logits, y))
    loss, grad = softmax_cross_entropy(logits, y)
    assert np.isfinite(loss)
    # Only the differences between logits matter, so this must equal the loss
    # for [0, 1, 2] exactly.
    ref, ref_grad = softmax_cross_entropy(np.array([[0.0, 1.0, 2.0]]), y)
    assert loss == pytest.approx(ref)
    assert np.allclose(grad, ref_grad)
    assert np.allclose(softmax(logits), [[0.0900, 0.2447, 0.6652]], atol=1e-4)


def test_fused_survives_a_confidently_wrong_prediction():
    """The other failure mode, and the one that actually kills training runs.

    If the correct class gets a probability that underflows to exactly 0.0,
    -log(0.0) is +inf, and one inf poisons every parameter on the next update.
    Nothing raises; the loss simply prints inf. The fused form is
    logsumexp(z) - z_y, which is a subtraction of finite numbers.
    """
    logits = np.array([[0.0, 900.0]])
    y = np.array([0])                      # the model is as wrong as it is possible to be
    with np.errstate(over="ignore", invalid="ignore"):
        assert not np.isfinite(cross_entropy_unfused(logits, y))
    loss, grad = softmax_cross_entropy(logits, y)
    assert loss == pytest.approx(900.0)    # logsumexp(z) - z_0 = 900 - 0
    assert np.isfinite(grad).all()
    assert np.allclose(grad, [[-1.0, 1.0]])   # every entry stays inside [-1, 1]


def test_log_sum_exp_is_shift_invariant():
    z = np.array([[3.0, -1.0, 0.5]])
    assert log_sum_exp(z + 500.0, axis=1)[0] == pytest.approx(log_sum_exp(z, axis=1)[0] + 500.0)


def test_softmax_is_shift_invariant():
    """Subtracting a constant from every logit multiplies numerator and
    denominator by the same factor. That identity is the whole trick, and it
    is exact rather than approximate.
    """
    z = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(softmax(z), softmax(z - 12345.0))


def test_mse_and_its_factor_of_two():
    pred = np.array([[3.0], [1.0]])
    target = np.array([[1.0], [1.0]])
    loss, grad = mse_loss(pred, target)
    assert loss == pytest.approx(2.0)                 # (4 + 0) / 2
    assert np.allclose(grad, [[2.0], [0.0]])          # 2*(pred-target)/N


def test_reduction_changes_the_gradient_by_a_factor_of_n():
    """'sum' and 'mean' are both correct and are not interchangeable. Swapping
    one for the other at batch size N multiplies every gradient by N, which
    usually shows up as a run that was stable and diverges the moment the batch
    size changes.
    """
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((5, 3))
    y = rng.integers(0, 3, size=5)
    lm, gm = softmax_cross_entropy(logits, y, reduction="mean")
    ls, gs = softmax_cross_entropy(logits, y, reduction="sum")
    assert ls == pytest.approx(lm * 5)
    assert np.allclose(gs, gm * 5)


def test_accuracy_and_confusion_matrix():
    logits = np.array([[2.0, 1.0], [0.0, 5.0], [1.0, 0.5]])
    y = np.array([0, 1, 1])
    assert accuracy(logits, y) == pytest.approx(2 / 3)
    cm = confusion_matrix(y, np.argmax(logits, axis=1), 2)
    assert np.array_equal(cm, [[1, 0], [1, 1]])       # rows truth, columns guess
    assert cm.sum() == 3
    assert np.trace(cm) / cm.sum() == pytest.approx(accuracy(logits, y))


def test_confusion_matrix_counts_duplicates():
    """`np.add.at` and not `cm[i, j] += 1`: buffered fancy-index assignment
    applies one of a set of duplicate updates and silently drops the rest,
    which for a histogram is every update but one.
    """
    cm = confusion_matrix(np.zeros(7, dtype=int), np.zeros(7, dtype=int), 2)
    assert cm[0, 0] == 7
