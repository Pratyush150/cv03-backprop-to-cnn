"""Optimisers: the rule that turns a gradient into a parameter update.

The gradient points UPHILL -- it is the direction of steepest increase -- so
every rule here moves against it. Everything else is a question of how much
history to keep.
"""

from __future__ import annotations

import numpy as np


class SGD:
    """w <- w - lr * g, with optional momentum.

    The learning rate is the hyperparameter that decides whether anything works
    at all, and the reason has a closed form on a quadratic: the distance to the
    optimum is multiplied by (1 - lr*curvature) every step, so descent converges
    iff |1 - lr*curvature| < 1, i.e. 0 < lr < 2/curvature, and lands exactly on
    the optimum in one step at lr = 1/curvature. Above 2/curvature it diverges
    geometrically. Exactly AT 2/curvature it orbits forever -- and, because the
    two points it alternates between are equidistant from the minimum, the
    printed loss is IDENTICAL on every step while the weights swing wildly. A
    flat loss curve does not mean nothing is happening. examples/02 shows all
    five regimes.

    Momentum keeps a running average of past gradients and steps along that:

        v <- beta*v + g ;  w <- w - lr*v

    Consistent directions accumulate, oscillating ones cancel. The number to
    know is that a constant gradient g drives v towards g/(1-beta), so beta=0.9
    multiplies the eventual step by TEN. Turning momentum on without lowering
    the learning rate is a standard way to blow up a run that was fine.
    """

    def __init__(self, model, lr: float = 0.1, momentum: float = 0.0) -> None:
        self.model, self.lr, self.momentum = model, lr, momentum
        self.velocity: dict[tuple[int, str], np.ndarray] = {}

    def step(self) -> None:
        for i, (layer, name) in enumerate(self.model.parameters()):
            p, g = layer.params[name], layer.grads[name]
            if self.momentum:
                key = (i, name)
                v = self.velocity.get(key)
                v = g.copy() if v is None else self.momentum * v + g
                self.velocity[key] = v
                g = v
            # In place. `p = p - lr*g` would rebind the local name and leave
            # the layer holding the old array, so the model would never train
            # and nothing would raise.
            p -= self.lr * g


class Adam:
    """Adam: per-parameter step sizes from a running estimate of each
    parameter's own gradient magnitude.

        m <- b1*m + (1-b1)*g            running mean of the gradient
        v <- b2*v + (1-b2)*g^2          running mean of the SQUARED gradient
        w <- w - lr * m_hat / (sqrt(v_hat) + eps)

    "Adaptive" is not a hand-wave, it is that division. A parameter whose
    gradient is consistently about 0.01 gets m ~ 0.01 and sqrt(v) ~ 0.01, so
    the ratio is about 1 and it moves a full lr. A parameter whose gradient is
    consistently about 10 gets the same ratio and the same step. The raw
    magnitudes cancel, which is the entire point and the reason Adam trains a
    badly scaled network that plain SGD cannot.

    The hats are BIAS CORRECTION. m and v both start at exactly zero, so for
    the first handful of steps they read too small -- after one step with
    b1=0.9, m is 0.1*g rather than g. Dividing by (1 - b1^t) removes exactly
    that startup shortfall. Leave it out and the first steps are ~10x too
    small, which looks like a warm-up you did not ask for.

    eps is inside the sqrt's parentheses, not outside: it exists to stop a
    division by zero for a parameter whose gradient has been zero throughout
    (a dead ReLU's incoming weights, for instance).
    """

    def __init__(self, model, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8) -> None:
        self.model, self.lr = model, lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m: dict[tuple[int, str], np.ndarray] = {}
        self.v: dict[tuple[int, str], np.ndarray] = {}
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, (layer, name) in enumerate(self.model.parameters()):
            key = (i, name)
            p, g = layer.params[name], layer.grads[name]
            m = self.m.get(key, np.zeros_like(p))
            v = self.v.get(key, np.zeros_like(p))
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g * g
            self.m[key], self.v[key] = m, v
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
