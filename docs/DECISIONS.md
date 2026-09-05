# DECISIONS

Architectural decision records. One per real choice: what was decided, what
else was on the table, why this one, and what it costs. Where a decision has a
measurable cost, the measurement is here.

---

## ADR-001 — NumPy does the learning; PyTorch and scipy only ever check it

**Decision.** Every forward pass, every backward pass, every loss and every
optimiser in `src/netfs/` is written against NumPy alone. `import torch` appears
in exactly one file, `tests/test_torch_oracle.py`, and `import scipy` in exactly
one other, `tests/test_conv.py`. Neither is ever imported by the library, and
neither ever computes a gradient that is then used to train anything.

**Alternatives.**
- *Use PyTorch throughout.* Faster to write, faster to run, and it would defeat
  the entire purpose: the point of this repository is that nothing is hidden.
- *Use no external reference at all.* Self-contained, and much weaker. A
  central-difference check differentiates **our own forward pass**; if the
  forward pass is wrong in an interesting way, the numerical gradient
  faithfully reports the derivative of the wrong function and the check passes.
  Only an independent implementation catches that class of error.

**Why this split.** It gives two genuinely independent lines of defence and
keeps them clearly labelled. The finite-difference check answers "is my
backward pass the derivative of my forward pass?". The scipy and torch
cross-checks answer "is my forward pass the operation I think it is?". Those
are different questions and a repository that teaches gradient checking should
be explicit that the first one alone is not enough.

**Cost.** Slower than a framework by a wide margin, and float64 throughout
(ADR-003) makes it slower again. The CNN in `examples/09_train_cnn.py` trains
in about 10 s for 30 epochs on 1,258 images -- fine for this dataset, useless
for anything real. The repository says so rather than implying otherwise.

**Consequence for CI.** `torch` is installed with `|| true` in the workflow and
`tests/test_torch_oracle.py` skips itself if the import fails, so the suite
still runs on a machine without it. Skipping an oracle is acceptable; skipping
the gradient check is not, and the gradient check needs nothing but NumPy.

---

## ADR-002 — NCHW layout, matching PyTorch exactly

**Decision.** Batched images are `(N, C, H, W)` and conv weights are
`(C_out, C_in, kh, kw)`.

**Alternatives.**
- *NHWC / HWC*, which is what OpenCV, matplotlib and most image files use, and
  what a from-scratch tutorial usually picks because a single image is then
  `(H, W, C)` and prints nicely.

**Why NCHW.** The torch cross-check in `tests/test_torch_oracle.py` becomes a
direct comparison with no transposes on either side. A transpose inside a test
is a place for the test to be wrong in the same way the code is wrong, and the
one thing an oracle must not share with the code under test is a bug.

**Cost.** Every image arriving from a file or leaving for matplotlib needs a
transpose. That conversion is confined to `netfs.data` and to the example
scripts' plotting code, and it is named where it happens. It is a real seam and
pretending otherwise would be worse: the mismatch between OpenCV's HWC and
PyTorch's NCHW is one of the most common sources of a silently transposed
image in practice.

---

## ADR-003 — float64 everywhere, deliberately

**Decision.** Every array in the package is float64. There is no float32 path
and no dtype argument.

**Alternatives.** float32, which halves memory and is roughly what real
training uses.

**Why.** Gradient checking is the central claim of this repository, and in
float32 it does not work. The central difference subtracts two nearly equal
numbers, and float32's 24-bit mantissa leaves so little behind that a
completely correct implementation reports relative errors around 1e-3 -- which
is above every threshold anyone sets for "is my gradient right". An evening
lost hunting a bug that does not exist is the standard price. NumPy's default
is float64; PyTorch's default is float32, which is exactly why
`tests/test_torch_oracle.py` casts everything to float64 before comparing, and
why one of its tests asserts that the defaults differ.

**Cost.** Roughly 2x the memory and, on this CPU, a similar factor of time
against a float32 implementation. Irrelevant at this scale and decisive at any
real scale, which is why real training is mixed-precision and real gradient
checks are not.

---

## ADR-004 — batch-first row-major, `Z = X @ W.T + b`

**Decision.** One sample per row, `X` is `(N, D_in)`, `W` is `(D_out, D_in)`
with one neuron per row, and the forward pass is `X @ W.T + b`. The
single-sample column form `z = W x + b` appears in the derivations for
intuition and never in the code.

**Alternatives.** Column-major `(D, N)`, which makes the single-sample maths
prettier and is what many textbooks use.

**Why.** It matches `torch.nn.Linear`, so the oracle comparison is literal, and
it makes the batch axis the first axis everywhere, so `sum(axis=0)` always means
"over the batch".

**Cost.** The transposes in the backward pass (`dW = dZ.T @ X`) look less
symmetric than the column-major version. That is a presentation cost paid once
in the docs, against a correctness benefit paid on every layer -- mixing the
two conventions inside a single derivation produces an `(N, N)` matrix with no
exception at all, because NumPy is perfectly happy to broadcast it.

---

## ADR-005 — im2col is the implementation; the naive loop is kept as a tested reference

**Decision.** `Conv2D` computes its forward and backward passes through im2col
and a matrix multiply. `conv2d_naive`, the four-nested-loop version, stays in
the package as public API and is asserted equal to the fast path in
`tests/test_conv.py` for six combinations of stride and padding.

**Alternatives.**
- *Ship only the loops.* Honest and unusably slow: `examples/09` would take
  minutes instead of seconds.
- *Ship only im2col.* Faster to maintain, and it deletes the thing a reader
  needs in order to believe the fast version. im2col is index bookkeeping; the
  loops are the definition.

**Why both.** The comparison is the lesson, and asserting the agreement is what
makes the lesson credible. It also means an im2col indexing bug -- which
produces no exception and no shape error, just a permuted kernel -- is caught
by a test rather than by a confusing training curve.

**Cost, measured** (`examples/07_im2col_speedup.py`, this CPU, best of 3):

| configuration | loop per filter | loop per position (einsum) | im2col + GEMM |
|---|---|---|---|
| 32x32, 3->16, k3 | 0.0626 s | 0.0039 s | 0.00046 s |
| 128x128, 3->16, k3 | 1.0927 s | 0.0636 s | 0.00785 s |
| batch 8, 28x28, 1->8, k5 | 0.2041 s | 0.0048 s | 0.00052 s |

im2col is **74x to 389x** faster than the fully looped version and **5x to 9x**
faster than the einsum reference the library actually ships. Reporting only the
first number would be true and misleading: most of the win is ordinary
vectorisation and only the remainder is the GEMM.

The memory cost is real: the column matrix is about `k^2` times the activation.
A 224x224x64 float32 activation is 12.8 MB and its 3x3 column matrix is 116 MB.
That 9x is why production frameworks keep direct, Winograd and FFT paths as
well and choose per shape.

---

## ADR-006 — `np.add.at` in `col2im`, despite being slow

**Decision.** The input gradient of a convolution is scattered back with
`np.add.at`, not with `dxp[..., ii, jj] += dcols`.

**Why.** Fancy-index assignment in NumPy is *buffered*: with repeated indices
it applies one of the duplicate updates and silently discards the others. Every
overlapping convolution window guarantees repeated indices, so the fast
spelling produces a gradient that is wrong only in the interior of the image
and only where windows overlap. There is no warning. `np.add.at` is unbuffered
and therefore correct.

**Cost.** `np.add.at` is substantially slower than buffered assignment and is
the dominant cost of `Conv2D.backward`. The alternative -- reshaping the
scatter into a series of non-overlapping slice adds, one per kernel offset --
is faster and considerably harder to read, and this repository is optimised for
being read. The same trap appears in `confusion_matrix`, for the same reason,
and is asserted there too.

---

## ADR-007 — one fused softmax-cross-entropy, with the broken version kept for demonstration

**Decision.** `softmax_cross_entropy` computes the loss as
`logsumexp(z) - z_y` and the gradient as `p - onehot`. It is the only loss the
models use. `softmax_naive` and `cross_entropy_unfused` are also shipped, are
never used for training, and exist to be watched failing.

**Why fused.** Two independent failure modes disappear at once. `exp(1000)` is
inf in float64, so a naive softmax is `inf/inf = nan`; and a confidently wrong
prediction underflows `p_y` to exactly 0.0, so `-log(p_y)` is `+inf`. Neither
raises. The fused form never exponentiates a positive number and never forms a
probability, so a logit gap of 900 gives a loss of 900 and a gradient whose
entries are all in [-1, 1].

The gradient derivation is where the argument really lands: differentiating the
two stages separately produces a per-sample CxC softmax Jacobian multiplied by
`1/p_y` -- and `p_y` is exactly the quantity that just underflowed. Composed
algebraically first, the `p_y` cancels (DERIVATIONS.md section 6) and what
remains is a subtraction that cannot blow up.

**Why keep the broken one.** A stability claim nobody has watched break is a
slogan. `tests/test_losses.py` asserts both halves of the pair: the two agree
to 1e-12 on safe inputs, and on unsafe ones the naive route returns nan or inf
while the fused route returns the right answer.

**Cost.** Two extra public functions that must never be used by accident. They
are named `_naive` and `_unfused` and their docstrings say what they are for.

---

## ADR-008 — relative error is measured against the array's scale, not entry by entry

**Decision.** `relative_error(a, b)` returns `max|a-b| / max(max|a|, max|b|)` by
default. The more commonly quoted per-entry form is available as
`relative_error(a, b, elementwise=True)`.

**Why.** A central difference has a noise floor of roughly 1e-11 in float64 no
matter what it is measuring. In a softmax gradient whose largest entry is 0.16,
an entry whose true value is 3e-06 is therefore known to about four digits and
no more, and its per-entry relative error reads 3e-06 -- a number that
describes the resolution of the measuring instrument, not the correctness of
the code. Using the per-entry form, several correct layers in this package
"fail" a 1e-7 threshold purely because they contain some very small gradients.

**Cost, stated plainly.** The array-scale measure can hide an error that lives
only in entries far smaller than the largest one. That is why it is not the
only defence: every backward pass also asserts its output shape, every layer is
cross-checked against torch autograd, and the per-entry form is one keyword
argument away. `tests/test_gradients.py` contains a test that demonstrates the
difference between the two readings on a constructed example.

---

## ADR-009 — scikit-learn's bundled 8x8 digits, with a synthetic fallback

**Decision.** `examples/09_train_cnn.py` trains on `sklearn.datasets.load_digits`
-- the UCI optical-digits set, 1797 images at 8x8, ten classes -- which ships
inside the scikit-learn wheel and needs no network. If scikit-learn is not
installed, `netfs.data.load_image_dataset` generates a three-class synthetic
shapes dataset instead and says so in the `source` field that every caption and
printout quotes.

**Alternatives.**
- *MNIST.* Real, standard, 70,000 images -- and a download, which the spec for
  this repository rules out, and 28x28 rather than 8x8, which would make the
  pure-NumPy training run minutes rather than seconds.
- *Synthetic only.* No dependency at all, and it can only demonstrate that the
  training loop runs. A synthetic set is separable by construction, so a high
  accuracy on it means nothing.

**Why the digits.** It is real handwriting with real ambiguity, so the
confusion matrix confuses the digits a person would confuse (three 8s called 1,
a 9 called 5), and it is small enough that 30 epochs of pure-NumPy float64
training finish in 12 seconds.

**Cost, and the honest caveat.** The dataset does not expose which of its 43
writers produced each sample, so a writer-disjoint split is impossible and the
split here is stratified-random with a fixed seed. Consecutive samples from the
same writer are therefore split across train and test, which makes the reported
98.14% an **optimistic** estimate of accuracy on a new writer's handwriting.
That caveat is printed by the script, stored on the dataset object, and repeated
in the README. A number without its split protocol is not a result.

---

## ADR-010 — explicit layer objects with `params`/`grads` dicts, and no autograd tape

**Decision.** Each layer is an object with `forward`, `backward`, a `params`
dict and a `grads` dict. `Sequential.backward` walks the list in reverse. There
is no tape, no graph object, and no operator overloading.

**Alternatives.**
- *A tiny autograd engine* (a `Tensor` class recording operations). It is a
  well-known and enjoyable exercise, and it would hide precisely the thing this
  repository exists to show. Once `loss.backward()` works, the reader is back
  to trusting a mechanism instead of reading one.
- *Pure functions returning `(out, backward_closure)`.* Elegant, and it makes
  the cache invisible -- which is a shame, because the cache is a lesson: it is
  the mechanical reason training uses more memory than inference and the reason
  gradient checkpointing exists as a trade.

**Why.** Every intermediate is a named attribute you can print. `check_layer`
can walk any layer generically because the parameters live in a dict rather than
in ad-hoc attributes, and the optimisers can update them in place for the same
reason.

**Cost.** Only straight chains are supported. A ResNet skip connection would
need a real graph, and the one extra rule it requires -- a value consumed twice
receives the sum of both gradients -- is stated in DERIVATIONS.md section 1 but
not implemented. That is a deliberate stopping point, named in the README's
limitations.

---

## ADR-011 — He initialisation by default, and a deliberately small output layer

**Decision.** `Linear` and `Conv2D` initialise weights with standard deviation
`sqrt(2 / fan_in)`, where `fan_in` is `C_in * k * k` for a convolution. Biases
start at zero. The final classification layer in `examples/09` overrides this
with `weight_scale=0.01`.

**Why the 2.** ReLU zeroes half its inputs, which halves the variance of the
signal at every layer. Without the compensating factor the activations shrink
geometrically with depth and the gradient shrinks with them.
`tests/test_layers.py` asserts this over a ten-layer stack: with
`sqrt(2/fan_in)` the activation standard deviation is still within a factor of
two of the input's, and with `sqrt(1/fan_in)` it is not.

**Why the small output layer.** The initial logits should be near zero so the
loss starts at `ln(C)` and the first gradient is not dominated by a random
preference for some class. He initialisation exists to keep signal alive
through depth; the last layer has nothing after it to keep alive. With the He
default the ten-class run starts at 2.84 instead of 2.30 -- not fatal, and it
breaks the single most useful pre-flight check there is.

**Cost.** One more knob, and a default that is right for ReLU and wrong for
tanh (which wants Xavier's `sqrt(1/fan_in)`). The argument is exposed rather
than hidden.

---

## ADR-012 — figures are committed to `docs/figures/`

**Decision.** The eleven PNGs produced by the examples are checked in, not
gitignored, and `examples/run_all.py` regenerates all of them.

**Why.** These are teaching materials. A reader who is browsing the repository
on the web, or who does not want to install scikit-learn, still has to be able
to see what the code produced -- and every number in the README is a number
they can point at in a figure.

**Cost.** About 1 MB of binary in the repository and figures that can drift out
of date if someone changes the code without re-running the examples.
`run_all.py` exists to make that a single command, and every figure caption
names the script that produced it.
