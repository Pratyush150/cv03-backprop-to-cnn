"""Dense layers and activations: shapes, the collapse of stacked linear layers,
and the hand-set XOR network.
"""

from __future__ import annotations

import numpy as np
import pytest

from netfs.data import xor_dataset
from netfs.layers import Flatten, Linear, ReLU, Sigmoid, Tanh
from netfs.model import Sequential


def test_linear_forward_matches_the_single_sample_formula():
    """Batched `X @ W.T + b` and single-sample `W @ x + b` are the same
    algebra with the batch axis added. Mixing the two conventions inside one
    derivation is the error that produces an (N, N) matrix where an (N, D_out)
    belonged -- with no exception, because NumPy is happy to broadcast it.
    """
    rng = np.random.default_rng(0)
    layer = Linear(4, 3, rng=rng)
    x = rng.standard_normal((5, 4))
    batched = layer.forward(x)
    for i in range(5):
        single = layer.params["W"] @ x[i] + layer.params["b"]
        assert np.allclose(batched[i], single)


def test_gradient_shapes_match_parameter_shapes():
    """dL/dW has exactly W's shape, always -- one partial derivative per entry.
    The layer asserts this itself; here we confirm the assertion fires.
    """
    layer = Linear(4, 3, rng=np.random.default_rng(1))
    layer.forward(np.zeros((6, 4)))
    layer.backward(np.ones((6, 3)))
    assert layer.grads["W"].shape == layer.params["W"].shape == (3, 4)
    assert layer.grads["b"].shape == layer.params["b"].shape == (3,)
    with pytest.raises(ValueError, match="expected"):
        layer._store("W", np.zeros((4, 3)))       # the transpose, rejected


def test_bias_gradient_is_a_sum_over_the_batch():
    """The same bias vector is added to every row of the batch, so it is a
    multi-path variable and multi-path gradients ADD. With an all-ones upstream
    gradient and a batch of 7, db must be exactly 7 -- not 1, and not 1/7.
    """
    layer = Linear(3, 2, rng=np.random.default_rng(2))
    layer.forward(np.zeros((7, 3)))
    layer.backward(np.ones((7, 2)))
    assert np.array_equal(layer.grads["b"], [7.0, 7.0])


def test_two_linear_layers_collapse_to_one():
    """The claim is algebraic, not approximate:

        W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)

    Stack a hundred linear layers and you still have a single matrix multiply.
    You have burned a hundred layers' worth of parameters to buy exactly zero
    extra expressive power. Here the collapsed single layer is constructed and
    asserted to be identical to the stack, to machine precision.
    """
    rng = np.random.default_rng(3)
    a, b = Linear(3, 5, rng=rng), Linear(5, 2, rng=rng)
    stacked = Sequential(a, b)
    collapsed = Linear(3, 2, rng=rng)
    collapsed.params["W"][:] = b.params["W"] @ a.params["W"]
    collapsed.params["b"][:] = b.params["W"] @ a.params["b"] + b.params["b"]
    x = rng.standard_normal((10, 3))
    assert np.abs(stacked.forward(x) - collapsed.forward(x)).max() < 1e-12


def test_the_hand_set_xor_network():
    """Two ReLU hidden units and one linear output, weights set by hand, no
    training at all -- and the outputs are exactly XOR.

        W1 = [[1, 1], [1, 1]]   b1 = [0, -1]
        W2 = [1, -2]            b2 = 0

    Delete the ReLU and the same weights collapse to y = -x1 - x2 + 2, which
    outputs 2, 1, 1, 0 instead of 0, 1, 1, 0 -- and no choice of weights could
    fix it, because a single linear function of two binary inputs cannot
    produce 0, 1, 1, 0. The nonlinearity is not a tuning detail.
    """
    x, y = xor_dataset()
    w1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    w2 = np.array([[1.0, -2.0]])
    b2 = np.array([0.0])

    h = np.maximum(x @ w1.T + b1, 0.0)
    out = h @ w2.T + b2
    assert np.array_equal(out.ravel(), [0.0, 1.0, 1.0, 0.0])
    assert np.array_equal(out, y)

    w_eff = w2 @ w1
    b_eff = w2 @ b1 + b2
    assert np.allclose(w_eff, [[-1.0, -1.0]])
    assert np.allclose(b_eff, [2.0])
    linear_out = x @ w_eff.T + b_eff
    assert np.array_equal(linear_out.ravel(), [2.0, 1.0, 1.0, 0.0])


def test_relu_is_elementwise_not_a_reduction():
    """np.maximum(0, z) keeps the shape; np.max(z) collapses to a scalar and
    nothing raises. The symptom is an activation printing as a bare float where
    a vector belonged, and a network that returns confident garbage.
    """
    z = np.array([[-1.0, 2.0], [3.0, -4.0]])
    out = ReLU().forward(z)
    assert out.shape == z.shape
    assert np.array_equal(out, [[0.0, 2.0], [3.0, 0.0]])
    assert np.max(z).shape == ()          # what the wrong call would have given


def test_dead_relu_receives_exactly_zero_gradient():
    """Not small -- zero. So no optimiser can move the weights feeding it, so
    its pre-activation cannot change, so it can never revive on its own. The
    dying-ReLU problem is a structural property of this one line.
    """
    layer = ReLU()
    layer.forward(np.array([[-1.5, 2.0]]))
    dz = layer.backward(np.array([[-4.5, 1.5]]))
    assert dz[0, 0] == 0.0
    assert dz[0, 1] == 1.5


def test_sigmoid_survives_inputs_that_overflow_the_textbook_form():
    """1/(1+exp(-z)) at z = -800 evaluates exp(800): inf, a warning, and 0.0.
    Not a theoretical concern -- feed a net raw 0-255 pixels and the first
    pre-activations are in the hundreds immediately.
    """
    z = np.array([[-800.0, 800.0, 0.0]])
    out = Sigmoid().forward(z)
    assert np.isfinite(out).all()
    assert out[0, 0] == pytest.approx(0.0, abs=1e-300)
    assert out[0, 1] == pytest.approx(1.0)
    assert out[0, 2] == pytest.approx(0.5)
    with np.errstate(over="ignore"):
        naive = 1.0 / (1.0 + np.exp(-z))
    assert naive[0, 0] == 0.0             # the same answer here, by luck
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.exp(800.0))   # ...and this is why it warned


def test_activation_derivative_maxima():
    """The two numbers behind the vanishing-gradient story: sigmoid's
    derivative peaks at 0.25 and tanh's at 1.0, both at z = 0. Chain ten
    sigmoids and the gradient is multiplied by at most 0.25^10 = 1e-6.
    """
    z = np.linspace(-6, 6, 2001).reshape(1, -1)
    s = Sigmoid()
    s.forward(z)
    assert s.backward(np.ones_like(z)).max() == pytest.approx(0.25, abs=1e-6)
    t = Tanh()
    t.forward(z)
    assert t.backward(np.ones_like(z)).max() == pytest.approx(1.0, abs=1e-6)
    assert 0.25 ** 10 == pytest.approx(9.5367e-7, rel=1e-3)


def test_flatten_round_trips():
    x = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    layer = Flatten()
    out = layer.forward(x)
    assert out.shape == (2, 12)
    assert np.array_equal(layer.backward(out), x)


def test_he_initialisation_preserves_activation_scale_through_depth():
    """Why the 2 in sqrt(2/fan_in): ReLU zeroes half the units, halving the
    variance at every layer. With the compensation the activation standard
    deviation stays near 1 through ten layers; with plain sqrt(1/fan_in) it
    decays geometrically, and by layer ten the signal -- and the gradient with
    it -- is a fraction of what it started as.
    """
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal((256, 128))
    for scale_factor, expect_stable in ((np.sqrt(2.0), True), (np.sqrt(1.0), False)):
        x = x0
        for _ in range(10):
            layer = Linear(128, 128, weight_scale=scale_factor / np.sqrt(128), rng=rng, bias=False)
            x = ReLU().forward(layer.forward(x))
        ratio = x.std() / x0.std()
        assert (ratio > 0.5) == expect_stable, f"{scale_factor}: ratio {ratio:.3f}"
