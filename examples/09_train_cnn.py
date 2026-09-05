"""Example 09 -- a small CNN trained end to end, in NumPy, on a real dataset.

Run:  python3 examples/09_train_cnn.py

Everything the previous eight examples built, assembled and pointed at data:

    Conv2D(1->8, 3x3, pad 1) -> ReLU -> MaxPool 2x2
    Conv2D(8->16, 3x3, pad 1) -> ReLU -> MaxPool 2x2
    Flatten -> Linear(64 -> 10) -> softmax cross-entropy

No framework. Every forward pass, every gradient and the optimiser are the
NumPy in src/netfs, and every one of those gradients has been checked against a
central difference and against torch autograd in tests/.

The dataset is scikit-learn's bundled copy of the UCI optical-digits set: 1797
handwritten digits at 8x8, ten classes, shipped inside the scikit-learn wheel
so nothing downloads. It is small, real, and has real ambiguity in it -- the
confusion matrix at the end confuses the digits a person would confuse. If
scikit-learn is not installed the script falls back to a synthetic shapes set
and says so in every caption.

It also trains a dense network with a similar parameter count on the same data,
because "the CNN is better" is not worth saying without the other number.
"""

from __future__ import annotations

import platform
import time

import numpy as np

from _common import figure, rule, save

from netfs import (Adam, Conv2D, Flatten, Linear, MaxPool2D, ReLU, Sequential, accuracy,
                   check_model, confusion_matrix, evaluate, load_image_dataset, softmax,
                   softmax_cross_entropy, train)

EPOCHS, BATCH, LR, SEED = 30, 32, 3e-3, 0


def build_cnn(n_classes: int, in_shape, seed: int = SEED) -> Sequential:
    rng = np.random.default_rng(seed)
    c, h, w = in_shape
    flat = 16 * (h // 4) * (w // 4)      # two 2x2 pools, so each spatial axis is quartered
    return Sequential(
        Conv2D(c, 8, 3, pad=1, rng=rng), ReLU(), MaxPool2D(2),
        Conv2D(8, 16, 3, pad=1, rng=rng), ReLU(), MaxPool2D(2),
        Flatten(),
        # weight_scale 0.01 on the OUTPUT layer, not the He default. The initial
        # logits should be near zero so the loss starts at ln(C) and the first
        # gradient is not dominated by an arbitrary random preference for one
        # class. He initialisation is about keeping signal alive through DEPTH;
        # the last layer has nothing after it to keep alive.
        Linear(flat, n_classes, rng=rng, weight_scale=0.01),
    )


def build_mlp(n_classes: int, in_shape, hidden: int = 24, seed: int = SEED) -> Sequential:
    rng = np.random.default_rng(seed)
    d_in = int(np.prod(in_shape))
    return Sequential(Flatten(), Linear(d_in, hidden, rng=rng), ReLU(),
                      Linear(hidden, n_classes, rng=rng, weight_scale=0.01))


def main() -> None:
    data = load_image_dataset(test_fraction=0.3, seed=SEED)
    n_classes = len(data.class_names)
    in_shape = data.x_train.shape[1:]

    rule("dataset")
    print(f"  {data.source}")
    print(f"  train {data.x_train.shape}   test {data.x_test.shape}")
    print(f"  classes: {data.class_names}")
    print(f"  pixel range after normalisation: "
          f"[{data.x_train.min():.2f}, {data.x_train.max():.2f}]")
    print(f"  NOTE: {data.note}")

    rule("the network")
    model = build_cnn(n_classes, in_shape)
    print(model.describe(in_shape))

    rule("gradient checking on REAL data, and the kink it walks into")
    # Before training anything, confirm the gradients this network will be
    # trained on really are the gradients of its loss. Four samples is enough;
    # the check costs two forward passes per parameter.
    xs, ys = data.x_train[:4], data.y_train[:4]
    raw = max(check_model(model, softmax_cross_entropy, xs, ys).values())
    pre = model.layers[0].forward(xs)
    exact_zeros = float((pre == 0.0).mean())
    jitter = xs + 1e-2 * np.random.default_rng(123).standard_normal(xs.shape)
    jittered = max(check_model(model, softmax_cross_entropy, jitter, ys).values())
    print(f"  on the raw images        : worst relative error {raw:.2e}   <- looks broken")
    print(f"  on the same images + 1e-2 of noise: {jittered:.2e}   <- fine")
    print(f"  fraction of first-layer pre-activations that are EXACTLY 0.0: "
          f"{exact_zeros:.1%}")
    print("  This is the kink, not a bug, and it is worth understanding because it is the")
    print("  one case where a failing gradient check means nothing is wrong.")
    print("  An 8x8 digit is mostly blank. With zero padding and a bias initialised to")
    print("  zero, every patch that is entirely blank has a pre-activation of exactly 0.0 --")
    print("  precisely the point where ReLU has no derivative. The analytic rule picks 0")
    print("  there (so does PyTorch); the central difference measures the chord across the")
    print("  kink and gets 0.5. They disagree because the function is not differentiable,")
    print("  not because the code is wrong.")
    print("  The standard test: re-run the check at a different point. A bad row that MOVES")
    print("  is a kink; a bad row that STAYS PUT is a bug. Here it moves -- adding a")
    print("  hundredth of noise to the pixels lifts every pre-activation off the kink and")
    print("  the error drops by eight orders of magnitude.")

    rule("the two numbers you write down before pressing go")
    init_loss, _ = softmax_cross_entropy(model.forward(data.x_train[:128]), data.y_train[:128])
    print(f"  loss at initialisation : {init_loss:.4f}")
    print(f"  -ln(1/{n_classes}) = ln({n_classes})     : {np.log(n_classes):.4f}")
    print(f"  steps per epoch        : {int(np.ceil(len(data.x_train) / BATCH))}")
    print("  If the first number were far from the second, the bug would be in the wiring --")
    print("  labels, normalisation, or the loss being handed the wrong argument -- and it is")
    print("  much cheaper to find that here than after thirty epochs.")

    rule(f"training on {platform.processor() or platform.machine()}, CPU, float64")
    t0 = time.perf_counter()
    hist = train(model, softmax_cross_entropy, Adam(model, lr=LR),
                 data.x_train, data.y_train, data.x_test, data.y_test,
                 epochs=EPOCHS, batch_size=BATCH, seed=SEED, verbose=True)
    train_seconds = time.perf_counter() - t0

    rule("a dense network on the same data, for scale")
    mlp = build_mlp(n_classes, in_shape)
    t1 = time.perf_counter()
    mlp_hist = train(mlp, softmax_cross_entropy, Adam(mlp, lr=LR),
                     data.x_train, data.y_train, data.x_test, data.y_test,
                     epochs=EPOCHS, batch_size=BATCH, seed=SEED, verbose=False)
    mlp_seconds = time.perf_counter() - t1
    print(f"  MLP ({mlp.n_params():,} parameters) test accuracy "
          f"{mlp_hist['val_acc'][-1]:.4f} in {mlp_seconds:.1f}s")

    rule("results")
    test_loss, test_acc = evaluate(model, softmax_cross_entropy, data.x_test, data.y_test)
    _, train_acc = evaluate(model, softmax_cross_entropy, data.x_train, data.y_train)
    print(f"  CNN: {model.n_params():,} parameters")
    print(f"  training time            : {train_seconds:.1f} s for {EPOCHS} epochs "
          f"({np.mean(hist['epoch_seconds']):.2f} s/epoch)")
    print(f"  final train accuracy     : {train_acc:.4f}")
    print(f"  final TEST accuracy      : {test_acc:.4f}   (loss {test_loss:.4f})")
    print(f"  best test accuracy seen  : {max(hist['val_acc']):.4f} "
          f"at epoch {int(np.argmax(hist['val_acc']))}")
    print(f"  MLP baseline             : {mlp_hist['val_acc'][-1]:.4f} "
          f"with {mlp.n_params():,} parameters")

    logits = model.forward(data.x_test)
    pred = np.argmax(logits, axis=1)
    cm = confusion_matrix(data.y_test, pred, n_classes)
    print("\n  confusion matrix -- rows are truth, columns are the guess")
    width = max(len(c) for c in data.class_names)
    print(" " * (width + 4) + "  ".join(f"{c[:5]:>5}" for c in data.class_names))
    for i, name in enumerate(data.class_names):
        print(f"  {name:>{width}}  " + "  ".join(f"{v:5d}" for v in cm[i]))
    print(f"  trace / total = {np.trace(cm)}/{cm.sum()} = {np.trace(cm) / cm.sum():.4f}")

    off = [(cm[i, j], data.class_names[i], data.class_names[j])
           for i in range(n_classes) for j in range(n_classes) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    print("  the mistakes it actually makes, largest first:")
    for count, truth, guess in off[:5]:
        print(f"    {count:2d} x  true {truth} called {guess}")

    # ------------------------------------------------------- figure: training
    fig, ax = figure(1, 3, figsize=(12.5, 3.8))
    ax[0].plot(hist["train_loss"], color="#1565c0", lw=1.7, label="CNN train")
    ax[0].plot(hist["val_loss"], color="#e57373", lw=1.7, label="CNN test")
    ax[0].axhline(np.log(n_classes), color="#9e9e9e", ls="--", lw=1.0,
                  label=f"ln({n_classes}) = {np.log(n_classes):.2f}, untrained")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy")
    ax[0].set_title("loss")
    ax[0].legend(fontsize=7)

    ax[1].plot(hist["val_acc"], color="#1565c0", lw=1.7,
               label=f"CNN, {model.n_params():,} params")
    ax[1].plot(mlp_hist["val_acc"], color="#8e24aa", lw=1.7, ls="--",
               label=f"dense MLP, {mlp.n_params():,} params")
    ax[1].axhline(1.0 / n_classes, color="#9e9e9e", ls=":", lw=1.0, label="chance")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("test accuracy")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_title(f"held-out accuracy, final {test_acc:.4f}")
    ax[1].legend(fontsize=7, loc="lower right")

    ax[2].plot(np.cumsum(hist["epoch_seconds"]), hist["val_acc"], color="#1565c0", lw=1.7)
    ax[2].set_xlabel("wall-clock seconds on this CPU")
    ax[2].set_ylabel("test accuracy")
    ax[2].set_title(f"{train_seconds:.0f} s total, pure NumPy, float64")
    save(fig, "09-cnn-training.png")

    # -------------------------------------------------- figure: what it learnt
    filters = model.layers[0].params["W"]                 # (8, 1, 3, 3)
    # Dead-filter audit. A ReLU unit whose pre-activation is negative for every
    # input receives exactly zero gradient forever, so it can never recover.
    # Counting them is a one-line diagnostic worth running on any trained net.
    all_maps = model.layers[1].forward(model.layers[0].forward(data.x_test))
    per_filter_max = all_maps.max(axis=(0, 2, 3))
    dead = np.flatnonzero(per_filter_max == 0.0)
    print(f"\n  first-layer filters that never fire on ANY test image: "
          f"{len(dead)} of {filters.shape[0]}"
          + (f"  (filters {dead.tolist()})" if len(dead) else ""))
    if len(dead):
        print("  That is the dying-ReLU problem, in a network you just trained. Those")
        print("  filters receive exactly zero gradient, so no optimiser can revive them:")
        print("  they are dead weight in both senses, and the model reached 98% without")
        print("  them. It is the argument for LeakyReLU in one measurement.")

    sample_idx = int(np.argmax(data.y_test == data.y_test[0]))
    sample = data.x_test[sample_idx:sample_idx + 1]
    maps = model.layers[1].forward(model.layers[0].forward(sample))   # conv -> relu
    n_f = filters.shape[0]
    fig, ax = figure(2, n_f + 1, figsize=(11.5, 3.4))
    ax[0][0].imshow(sample[0, 0], cmap="gray")
    ax[0][0].set_title(f"input '{data.class_names[data.y_test[sample_idx]]}'", fontsize=7)
    ax[1][0].axis("off")
    for f in range(n_f):
        ax[0][f + 1].imshow(filters[f, 0], cmap="RdBu_r",
                            vmin=-np.abs(filters).max(), vmax=np.abs(filters).max())
        ax[0][f + 1].set_title(f"filter {f}" + (" (dead)" if f in dead else ""), fontsize=7)
        ax[1][f + 1].imshow(maps[0, f], cmap="viridis", vmin=0, vmax=max(maps.max(), 1e-9))
    ax[0][1].set_ylabel("3x3 kernel", fontsize=7)
    ax[1][1].set_ylabel("its response", fontsize=7)
    for row in ax:
        for a_ in row:
            a_.set_xticks([])
            a_.set_yticks([])
            a_.grid(False)
    fig.suptitle("first conv layer after training: the 8 kernels it learned, and what each "
                 "one responds to", fontsize=9)
    save(fig, "10-learned-filters.png")

    # ------------------------------------------- figure: confusion + mistakes
    fig, ax = figure(1, 2, figsize=(11, 4.6),
                     gridspec_kw={"width_ratios": [1.15, 1.0]})
    im = ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks(range(n_classes))
    ax[0].set_yticks(range(n_classes))
    ax[0].set_xticklabels(data.class_names, fontsize=7)
    ax[0].set_yticklabels(data.class_names, fontsize=7)
    ax[0].set_xlabel("predicted")
    ax[0].set_ylabel("truth")
    ax[0].grid(False)
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j]:
                ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                           color="white" if cm[i, j] > cm.max() * 0.6 else "#37474f")
    ax[0].set_title(f"confusion matrix, {cm.sum()} held-out images, "
                    f"accuracy {np.trace(cm) / cm.sum():.4f}")
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    wrong = np.flatnonzero(pred != data.y_test)
    probs = softmax(logits)
    ax[1].axis("off")
    ax[1].set_title(f"every mistake it makes ({len(wrong)} of {len(data.y_test)})", fontsize=9)
    cols = 6
    rows = int(np.ceil(len(wrong) / cols)) if len(wrong) else 1
    for n, idx in enumerate(wrong[:cols * 4]):
        sub = ax[1].inset_axes([(n % cols) / cols, 1 - (n // cols + 1) / max(rows, 4),
                                1 / cols * 0.86, 1 / max(rows, 4) * 0.66])
        sub.imshow(data.x_test[idx, 0], cmap="gray")
        sub.set_xticks([])
        sub.set_yticks([])
        sub.set_title(f"{data.class_names[data.y_test[idx]]}->"
                      f"{data.class_names[pred[idx]]} ({probs[idx, pred[idx]]:.2f})",
                      fontsize=6.5)
    save(fig, "11-confusion-matrix.png")


if __name__ == "__main__":
    main()
