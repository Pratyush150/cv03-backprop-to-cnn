"""Datasets. Small, offline, deterministic, and honest about what they are.

Nothing here downloads anything at import time or at call time. The image
dataset is either scikit-learn's bundled 8x8 digits (which ships inside the
scikit-learn wheel -- no network) or, if scikit-learn is not installed, a
synthetic shapes set generated from a seed. `load_image_dataset` reports which
one you got in its return value, and every figure and number in the README says
which one produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def xor_dataset() -> tuple[np.ndarray, np.ndarray]:
    """The four corners of XOR: output 1 when exactly one input is 1.

    Four points, and no straight line separates them -- the two 1s are
    diagonally opposite. A single linear layer draws exactly one straight line,
    so a linear model cannot do XOR. That is a geometry fact, not a training
    problem, and no amount of tuning fixes it. It is the smallest honest
    demonstration that a nonlinearity is structural rather than decorative.
    """
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return x, y


def two_moons(n: int = 400, noise: float = 0.15, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles: the standard "needs a curved boundary"
    toy, generated here rather than imported so the package keeps its
    numpy-only promise.

    Written out because the shape of the data is the whole point: the two
    classes are not linearly separable, so a network without a hidden
    nonlinearity is capped at roughly 85% here no matter how long it trains,
    while a two-layer MLP with ReLU gets past 99%.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_a = n // 2
    t_a = np.pi * rng.random(n_a)
    t_b = np.pi * rng.random(n - n_a)
    a = np.stack([np.cos(t_a), np.sin(t_a)], axis=1)
    b = np.stack([1.0 - np.cos(t_b), 0.5 - np.sin(t_b)], axis=1)
    x = np.concatenate([a, b]) + noise * rng.standard_normal((n, 2))
    y = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n - n_a, dtype=int)])
    order = rng.permutation(n)
    return x[order], y[order]


@dataclass
class ImageDataset:
    """Train/test split of an image classification set, in NCHW float64."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    source: str          # printed in every figure caption; do not let it drift
    note: str            # the honest caveat that belongs next to the accuracy


def _stratified_split(y: np.ndarray, test_fraction: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Split indices class by class, so both halves have every class in roughly
    the original proportion.

    A plain random split is fine on a balanced 1797-sample set and stops being
    fine the moment a class is rare: the test set can end up with three
    examples of it, and the reported per-class accuracy is then quantised to
    thirds and means nothing.
    """
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(idx.size)]
        cut = int(round(idx.size * test_fraction))
        test_idx.append(idx[:cut])
        train_idx.append(idx[cut:])
    return (np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(test_idx)))


def synthetic_shapes(n_per_class: int = 300, size: int = 12, rng=None) -> tuple:
    """Three classes -- square, disc, cross -- drawn on a `size`x`size` grid
    with random position, scale and a little noise.

    The fallback for a machine without scikit-learn. It is clean by
    construction, which is the point: a synthetic set proves the training loop
    works, and it cannot tell you anything about how the model behaves on real
    variation. Anything reported from this set says so.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    imgs, labels = [], []
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    for cls in range(3):
        for _ in range(n_per_class):
            img = np.zeros((size, size))
            r = rng.uniform(size * 0.20, size * 0.34)
            cy = rng.uniform(r + 1, size - r - 1)
            cx = rng.uniform(r + 1, size - r - 1)
            dy, dx = yy - cy, xx - cx
            if cls == 0:                                        # filled square
                img[(np.abs(dy) <= r) & (np.abs(dx) <= r)] = 1.0
            elif cls == 1:                                      # filled disc
                img[dy ** 2 + dx ** 2 <= r ** 2] = 1.0
            else:                                               # cross / plus
                arm = max(1.0, r * 0.35)
                img[((np.abs(dy) <= arm) & (np.abs(dx) <= r))
                    | ((np.abs(dx) <= arm) & (np.abs(dy) <= r))] = 1.0
            img += 0.08 * rng.standard_normal((size, size))
            imgs.append(img)
            labels.append(cls)
    return np.array(imgs), np.array(labels)


def load_image_dataset(test_fraction: float = 0.3, seed: int = 0,
                       force_synthetic: bool = False) -> ImageDataset:
    """The dataset the CNN example trains on.

    Preference order, and the reason for it: a REAL dataset if one is available
    without a download, otherwise a synthetic one. scikit-learn bundles the
    UCI optical-digits test set -- 1797 images, 8x8, ten classes, values 0-16 --
    inside the wheel, so it costs no network access and is genuinely
    handwritten data with genuine ambiguity in it (4/9 and 3/8 really do get
    confused). That is worth far more as a demonstration than any synthetic set,
    because a synthetic set can only tell you that your training loop runs.

    Normalisation: divide by 16 to land in [0, 1]. Do it BEFORE anything pads,
    because zero padding asserts that the world outside the image is black --
    harmless for data centred near zero, and a hard bright/dark edge injected
    into the first layer for raw 0-255 (or 0-16) data.
    """
    if not force_synthetic:
        try:
            from sklearn.datasets import load_digits          # noqa: PLC0415
        except ImportError:
            pass
        else:
            d = load_digits()
            images = d.images / 16.0
            labels = d.target
            rng = np.random.default_rng(seed)
            tr, te = _stratified_split(labels, test_fraction, rng)
            return ImageDataset(
                x_train=images[tr][:, None, :, :],            # HW -> NCHW, one channel
                y_train=labels[tr],
                x_test=images[te][:, None, :, :],
                y_test=labels[te],
                class_names=[str(i) for i in range(10)],
                source="scikit-learn load_digits (UCI optical digits, 1797 images, 8x8, 10 classes)",
                note=("Split is stratified-random with seed 0. The dataset does not expose which "
                      "of its 43 writers produced each sample, so a writer-disjoint split is not "
                      "possible; the test accuracy is therefore an optimistic estimate of "
                      "performance on a new writer's handwriting."),
            )
    images, labels = synthetic_shapes(rng=np.random.default_rng(seed))
    rng = np.random.default_rng(seed)
    tr, te = _stratified_split(labels, test_fraction, rng)
    return ImageDataset(
        x_train=images[tr][:, None, :, :], y_train=labels[tr],
        x_test=images[te][:, None, :, :], y_test=labels[te],
        class_names=["square", "disc", "cross"],
        source="netfs.data.synthetic_shapes (900 images, 12x12, 3 classes, generated from seed 0)",
        note=("Synthetic fallback: scikit-learn was not available. The classes are separable by "
              "construction, so a high accuracy here demonstrates that the training loop works "
              "and nothing about real-world performance."),
    )


def iterate_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, rng=None,
                        shuffle: bool = True):
    """Yield (x_batch, y_batch).

    Shuffle every epoch, and shuffle the TRAINING set only. Without shuffling,
    a dataset sorted by class -- which optical digits very nearly is -- hands
    the optimiser a hundred consecutive 0s, then a hundred consecutive 1s, and
    the network chases whichever class it saw most recently. The loss curve
    develops a sawtooth with the period of one epoch, which people usually
    blame on the learning rate.

    The last batch is smaller when the set does not divide evenly. It is kept
    rather than dropped: with a mean-reduction loss its gradient is on the same
    scale as every other batch's, so there is nothing to fix.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = x.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]
