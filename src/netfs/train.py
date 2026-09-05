"""The training loop, written out once.

Five lines do the work -- forward, loss, backward, step, repeat -- and
everything else in this file is measurement. That ratio is honest: the loop is
not the hard part, knowing whether it is working is.
"""

from __future__ import annotations

import time

import numpy as np

from .data import iterate_minibatches
from .losses import accuracy


def evaluate(model, loss_fn, x, y, batch_size: int = 256):
    """Loss and accuracy over a set, in batches.

    Batched rather than done in one shot because the im2col column matrix for a
    whole set is k^2 times the size of the set itself, and the evaluation pass
    is the easiest place to run out of memory on a machine that trained fine.
    """
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in iterate_minibatches(x, y, batch_size, shuffle=False):
        out = model.forward(xb)
        loss, _ = loss_fn(out, yb)
        total_loss += loss * len(xb)
        correct += int(round(accuracy(out, yb) * len(xb)))
        n += len(xb)
    return total_loss / n, correct / n


def train(model, loss_fn, optimizer, x_train, y_train, x_val=None, y_val=None, *,
          epochs: int = 10, batch_size: int = 32, seed: int = 0, verbose: bool = True,
          expected_init_loss: float | None = None):
    """Fit `model`, returning a history dict.

    `expected_init_loss` is not decoration. Before a run starts you can predict
    the loss of an untrained classifier exactly: it spreads its confidence
    evenly, so every class gets probability 1/C and the cross-entropy is
    -ln(1/C) = ln(C). That is 2.303 for ten classes and 1.099 for three. If the
    first printed loss is far from it, the bug is in the wiring -- labels,
    normalisation, or a loss being handed the wrong argument -- and it is worth
    finding now rather than after eight epochs. A first loss near zero cannot
    mean the model is already right; it means you are not computing the loss
    you think you are.
    """
    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_seconds": []}
    checked_init = expected_init_loss is None
    for epoch in range(epochs):
        t0 = time.perf_counter()
        running, n = 0.0, 0
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, rng):
            loss = model.backward_from_loss(loss_fn, xb, yb)
            if not checked_init:
                # Checked on the FIRST batch of the FIRST epoch, before any
                # parameter has moved. One epoch later the number means nothing.
                if verbose:
                    print(f"  loss at init: {loss:.4f}  (expected about "
                          f"{expected_init_loss:.4f} for an untrained classifier)")
                checked_init = True
            optimizer.step()
            running += loss * len(xb)
            n += len(xb)
        dt = time.perf_counter() - t0
        history["train_loss"].append(running / n)
        history["epoch_seconds"].append(dt)
        if x_val is not None:
            vl, va = evaluate(model, loss_fn, x_val, y_val)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
            if verbose:
                print(f"  epoch {epoch:2d}  train_loss {running / n:.4f}  "
                      f"val_loss {vl:.4f}  val_acc {va:.4f}  ({dt:.2f}s)")
        elif verbose:
            print(f"  epoch {epoch:2d}  train_loss {running / n:.4f}  ({dt:.2f}s)")
    return history
