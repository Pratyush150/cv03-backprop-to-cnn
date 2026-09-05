# WALKTHROUGH

The long version. Nine stages, in the order the examples run, each quoting the
code that does the work and the output it produces. The maths behind every
gradient is in [DERIVATIONS.md](DERIVATIONS.md); the reasons behind every
structural choice are in [DECISIONS.md](DECISIONS.md). This file is the thread
that connects them.

Everything printed below is real output from the scripts in `examples/`, on the
machine described in the README. Re-running `python3 examples/run_all.py`
reproduces all of it and regenerates every figure.

**Stages**

1. [One neuron](#stage-1--one-neuron)
2. [Why a nonlinearity is not optional](#stage-2--why-a-nonlinearity-is-not-optional)
3. [Backpropagation through a two-layer MLP](#stage-3--backpropagation-through-a-two-layer-mlp)
4. [Gradient checking](#stage-4--gradient-checking)
5. [Softmax, cross-entropy, and the fusion](#stage-5--softmax-cross-entropy-and-the-fusion)
6. [Convolution as a layer](#stage-6--convolution-as-a-layer)
7. [im2col](#stage-7--im2col)
8. [Pooling, stride, padding](#stage-8--pooling-stride-padding)
9. [A CNN, trained](#stage-9--a-cnn-trained)
10. [Bridge: from classifier to detector](#stage-10--bridge-from-classifier-to-detector)

---

## Stage 1 — one neuron

`examples/01_one_neuron.py`

A neuron is two operations glued together: a weighted sum, then a bend. Stage 1
uses only the first half -- one weight, one bias, no activation -- because the
goal is to isolate the training machinery from everything else.

```python
def loss_and_grad(w, b, x, y):
    r = w * x + b - y
    return float((r ** 2).mean()), float((2 * r * x).mean()), float((2 * r).mean())
```

Three lines, and the middle one is the entire chain rule. `r` is the residual;
`dL/dp = 2r/N` is the loss's derivative with respect to the prediction; then
each parameter multiplies that by its own local derivative of the prediction --
`x` for the weight, `1` for the bias. Every backward pass in this package is
that same two-step with bigger arrays.

The first thing the script does is not train. It checks:

```
  analytic  dL/dw = -15.7432084277   dL/db =  2.3808934547
  numerical dL/dw = -15.7432084274   dL/db =  2.3808934550
  relative error  = 1.81e-11
```

The numerical gradient wiggles each parameter by 1e-5 in both directions and
measures how the loss moved. Two forward passes per parameter. With two
parameters that is four forward passes; with a hundred million it is two
hundred million *per training step*, which is not a slow method but an
impossible one. That impossibility is the whole reason backpropagation exists,
and it is worth feeling it here where the alternative is still cheap.

Then it trains, with the one line that is gradient descent:

```python
w -= lr * dw          # THE update rule. Downhill, because the gradient
b -= lr * db          # points uphill.
```

```
  step    0  loss  21.2853   w  0.7872  b -0.1190
  step    2  loss   5.4775   w  1.6930  b -0.3291
  step   10  loss   0.3419   w  2.4527  b -0.8604
  step  399  loss   0.1247   w  2.4961  b -1.2701
  recovered w = 2.4961 (true 2.5),  b = -1.2701 (true -1.3)
  final loss 0.1247;  noise floor is about NOISE^2 = 0.1225
```

The loss stops at 0.1247 and not at 0.0, and the reason is in the data, not the
optimiser: the targets had 0.35 of Gaussian noise added, so a line cannot do
better than the variance of that noise. The flat tail of the loss curve is that
floor. Anyone reporting a loss below it has leaked their targets into their
model.

Finally, the "then a layer" step. The same problem is run through
`netfs.Linear(1, 1)` and `netfs.SGD`, starting from the same zeros:

```
  hand-written : w 2.4961389785  b -1.2700801219
  Linear layer : w 2.4961389785  b -1.2700801219
  max difference over the whole loss curve: 5.33e-15
```

A layer is not a new idea. It is this arithmetic with a batch axis and a matrix.

![One neuron fitting a line: data with the fitted line at four stages, the loss curve on a log axis with the noise floor marked, and the descent path on the loss surface](figures/01-one-neuron.png)

---

## Stage 2 — why a nonlinearity is not optional

`examples/02_why_nonlinearity.py`

The claim is that two linear layers with nothing between them are *algebraically*
one linear layer. The script does not assert that in prose; it builds the
collapsed layer and compares:

```python
collapsed.params["W"][:] = b.params["W"] @ a.params["W"]        # W_eff = W2 W1
collapsed.params["b"][:] = b.params["W"] @ a.params["b"] + b.params["b"]
```

```
  two layers (2->5->3) vs the single 2->3 layer they equal:
  max absolute difference over 7 random inputs: 2.78e-16
  the stack holds 33 parameters; the single layer it is equal to holds 9.
  Depth without a bend buys nothing at all.
```

Then XOR, which is where the consequence bites. XOR's four corners cannot be
separated by a straight line -- the two 1s are diagonally opposite -- and a
single linear layer draws exactly one straight line. The script trains the
linear stack and a version with one ReLU inserted, same data, same optimiser,
same learning rate:

```
  linear stack final loss : 0.250000
  with one ReLU           : 0.000000
  linear predictions : [0.5 0.5 0.5 0.5]
  ReLU predictions   : [0. 1. 1. 0.]
```

**0.25 is not a number the run happened to reach.** Any affine f satisfies
`f(0,0) + f(1,1) = f(0,1) + f(1,0)`; XOR's targets give 0 on the left and 2 on
the right; the best possible compromise is an error of 1/2 at each of four
corners, so the mean squared error floors at exactly 0.25. The derivation is in
DERIVATIONS.md section 3 and the assertion is in `tests/test_train.py`, which
also confirms that a wider and deeper linear stack floors at the same 0.25 --
if depth were buying anything, that number would move.

The figure shows what each model can express. The left panel is flat: the best
line through XOR is the constant 0.5. The middle panel shows the ReLU network's
piecewise-linear boundary carving out a region rather than a half-plane.

![Three panels: the trained linear network outputs 0.5 everywhere, the ReLU network draws a folded boundary that separates XOR, and their loss curves diverge by twenty-eight orders of magnitude](figures/02-xor-collapse.png)

The hand-set network is worth keeping in your head, because it needs no
training at all:

```
W1 = [[1, 1], [1, 1]]   b1 = [0, -1]      2 ReLU hidden units
W2 = [1, -2]            b2 = 0            1 linear output
->  0, 1, 1, 0, exactly XOR

delete the ReLU: W_eff = [-1, -1], b_eff = 2, i.e. y = -x1 - x2 + 2
->  2, 1, 1, 0, not XOR, and no weights could fix it
```

---

## Stage 3 — backpropagation through a two-layer MLP

`examples/03_backprop_mlp.py`

The network is 2 inputs, 2 ReLU hidden units, 1 linear output, mean squared
error, one sample. Nine scalar parameters, nine gradients, and every one of them
checkable with a pencil.

Forward:

```
  z1 = x @ W1.T + b1 = [-1.5  2. ]      <- neuron 0's pre-activation is negative
  h  = relu(z1)      = [0. 2.]          <- so neuron 0 is SWITCHED OFF
  z2 = h @ W2.T + b2 = [-0.5]
  L  = (z2 - y)^2    = 2.25
```

Backward, one node at a time, right to left:

```
  dL/dz2 = 2*(z2 - y)        = [-3.]     <- the seed; everything below is this
                                            number being routed and rescaled
  dL/dW2 = dz2.T @ h         = [ 0. -6.]
  dL/db2 = dz2.sum(axis=0)   = [-3.]
  dL/dh  = dz2 @ W2          = [-4.5  1.5]
  dL/dz1 = dh * (z1 > 0)     = [-0.   1.5]   <- entry 0 killed by the dead ReLU
  dL/dW1 = dz1.T @ x         = [[0.0, 0.0], [1.5, 3.0]]
  dL/db1 = dz1.sum(axis=0)   = [0.  1.5]
```

Read those seven lines as three rules applied five times. Addition distributes
(the bias gradients are the upstream gradient, unchanged). Multiplication swaps
(each weight gradient is the upstream gradient scaled by *the other* input,
which is why the forward pass must cache both). Max routes (the ReLU sends
everything to the winner and exactly zero to the loser).

Then all nine, against a central difference:

```
  param index       analytic           numeric      rel err
  W1    (0, 0)        0.0000      0.0000000000   exact-zero
  W1    (0, 1)        0.0000      0.0000000000   exact-zero
  W1    (1, 0)        1.5000      1.5000000000     6.55e-12
  W1    (1, 1)        3.0000      3.0000000000     6.55e-12
  b1    (0,)          0.0000      0.0000000000   exact-zero
  b1    (1,)          1.5000      1.5000000000     6.55e-12
  W2    (0, 0)        0.0000      0.0000000000   exact-zero
  W2    (0, 1)       -6.0000     -6.0000000000     8.50e-13
  b2    (0,)         -3.0000     -3.0000000000     6.55e-12
```

**The four exact zeros are the most instructive rows in the table.** The first
hidden unit's ReLU is off, so it receives no gradient at all -- not small,
zero. No optimiser can move its incoming weights, so its pre-activation cannot
change, so it can never come back on. That is the dying-ReLU problem in four
numbers, and stage 9 finds one in a network that was actually trained.

The numerical check agrees for the same reason: perturbing `W1[0,0]` by 1e-5
moves `z1[0]` from -1.5 to -1.50001, which is still negative, so the ReLU stays
off and the loss does not move. Both methods are measuring the same dead switch.

### Shapes, and why they are what they are

Shape errors are the commonest bug in a hand-written network, and the batched
case is the dangerous one: a wrong shape in the single-sample case raises
immediately, while in the batched case NumPy broadcasts and hands you a
numerically wrong gradient with no exception. The script prints the table and
then asserts it against the arrays the network actually produced:

```
  X                   (8, 4)      one sample per ROW. Batch first, always.
  W1                  (6, 4)      one NEURON per row. Forward is X @ W1.T, so the
                                  shared dimension d_in is what the transpose lines up.
  Z1 = X @ W1.T + b1  (8, 6)      b1 is (hidden,) and broadcasts down the batch.
  H = relu(Z1)        (8, 6)      elementwise: shape cannot change.
  Z2 = H @ W2.T + b2  (8, 3)      the logits.
  dZ2                 (8, 3)      same shape as Z2: one derivative per entry.
  dW2 = dZ2.T @ H     (3, 6)      must equal W2's shape. There is exactly one way
                                  to multiply (n,d_out) and (n,hidden) into (d_out,hidden).
  db2 = dZ2.sum(0)    (3,)        SUM over the batch: one bias, added n times, so
                                  it is a multi-path variable and the paths add.
  dH = dZ2 @ W2       (8, 6)      back through the same multiply, other side.
  dZ1 = dH * (Z1>0)   (8, 6)      the ReLU switch, elementwise.
```

`Layer._store` asserts `grad.shape == param.shape` on every backward pass, for
the same reason: it is free, and it converts a silent broadcast into a loud
failure.

### And then it trains

Two moons, 450 training points, 82 parameters:

```
  loss at init 0.8345, expected about ln(2) = 0.6931
  final train loss 0.0064  test accuracy 1.0000
  a single linear layer on the same data: test accuracy 0.8467
```

![A single linear layer draws one straight boundary and scores 0.847; a 2-16-2 ReLU network draws a piecewise-linear boundary around both moons and scores 1.000; the MLP's train and test loss fall together over 60 epochs](figures/03-backprop-mlp.png)

Look at the middle panel's boundary: it is made of straight segments. A ReLU
network *is* a piecewise-linear function -- each unit is a hyperplane that folds
the input space, and within any one region of that folding the network is
exactly affine. That is the honest answer to "why do we need activation
functions", and it is more useful than "to introduce nonlinearity".

---

## Stage 4 — gradient checking

`examples/04_gradient_check.py`

This is the stage the whole repository rests on. If you write a backward pass by
hand you will get one wrong, and it will not raise: the network still runs, the
loss still falls a bit, and you lose two days on the learning rate.

Every layer in the package, analytic against central difference:

```
  layer                             dW          db      dinput
  Linear(4->3)                1.22e-11    1.42e-11    8.90e-12
  ReLU                               -           -    2.57e-11
  Sigmoid                            -           -    3.23e-11
  Tanh                               -           -    1.94e-11
  Flatten                            -           -    3.05e-11
  Conv2D(2->3, k3, p1)        1.61e-11    2.65e-11    5.96e-11
  Conv2D(2->3, k3, s2)        1.48e-11    1.93e-11    7.87e-11
  MaxPool2D(2, 2)                    -           -    2.04e-11
  GlobalAvgPool2D                    -           -    7.23e-11
  worst relative error anywhere: 7.87e-11
```

A layer returns an array, not a scalar, and a gradient is only defined for a
scalar. So `check_layer` invents one: `L = sum(out * G)` for a fixed random `G`,
whose derivative with respect to `out` is exactly `G`. Random and not all-ones
on purpose -- summing is invariant to permutation and transposition, so an
all-ones seed cannot detect a backward pass that permutes its output, which is a
real bug class in conv layers where the forward pass ends in a reshape and a
transpose.

### The step size

```
  best h = 3.2e-05, relative error 3.36e-11
    h =  1.0e-01   rel err = 1.71e-04
    h =  1.0e-03   rel err = 1.71e-08
    h =  1.0e-05   rel err = 8.20e-11
    h =  1.0e-07   rel err = 8.78e-09
    h =  1.0e-09   rel err = 8.67e-07
    h =  1.0e-11   rel err = 5.14e-05
    h =  1.0e-13   rel err = 9.35e-03
```

Two errors fight. Truncation error falls as `h^2` (the centred difference
cancels the first-order Taylor terms; the one-sided difference does not, which
is why it is never used). Floating-point cancellation grows as `1/h`, because
`L(t+h)` and `L(t-h)` agree in their leading digits and subtracting them throws
those digits away. The minimum is around 1e-5. **If your relative error grows as
you shrink h, stop shrinking** -- you are on the cancellation branch and the
code is probably fine.

### Three real bugs, introduced on purpose

A test that has never failed has unknown sensitivity, so the script breaks three
backward passes and reports what the check says:

```
  correct Linear                    6.91e-12   PASS
  db without sum(axis=0)            1.05e+00   CAUGHT
  dW transposed                     8.44e-01   CAUGHT
  ReLU'(0) = 1 instead of 0         5.00e-01   CAUGHT
```

Each is a bug people really write. `db = dout[0]` instead of
`dout.sum(axis=0)` has the right shape and drops N-1 of the N contributions.
`dW` transposed raises a clean shape error on a non-square layer and passes the
shape assertion silently on a square one. `ReLU'(0) = 1` differs from the
correct version only where the pre-activation is exactly zero -- which sounds
like never, until you feed it zero-padded feature maps, at which point a large
fraction of the entries are exactly zero. Stage 9 hits this for real.

Note the magnitudes. A broken gradient is not off by 1e-5; it is off by a factor
of order one. That is why any threshold between 1e-7 and 1e-4 separates the two
cases with several orders of magnitude to spare.

![Three panels: every layer's worst relative error sits between 1e-12 and 1e-10, far below the 1e-7 line; the step-size sweep traces a clean V with its minimum at 3e-5; three deliberately sabotaged backward passes all land near relative error 1](figures/04-gradient-check.png)

---

## Stage 5 — softmax, cross-entropy, and the fusion

`examples/05_softmax_stability.py`

```
  z = [2.0, 1.0, 0.1], true class 0
  p        = [0.659  0.2424 0.0986]   (sums to 1.0000)
  L        = -ln(0.6590) = 0.4170
  dL/dz    = p - onehot  = [-0.341   0.2424  0.0986]
  the gradient sums to +0.0e+00 -- exactly zero, for every input, always.
```

That last line is a free correctness check you can apply to any softmax
gradient forever: `p` sums to 1 and the one-hot sums to 1, so the difference
sums to 0. Probability mass is conserved; the gradient can only move it between
classes.

The interesting part is why `dL/dz = p - onehot` at all, because the
intermediate steps are much uglier than the answer. Differentiating the softmax
gives a CxC Jacobian, `dp_i/dz_k = p_i(delta_ik - p_k)`; differentiating
`-log p_y` gives `-1/p_y`; and multiplying them, **the `p_y` cancels**
(DERIVATIONS.md section 6). That cancellation is the entire argument for
fusing the two operations, because `p_y` is exactly the quantity that
underflows.

```
  max logit      2.0 | naive p[0]      0.659 | naive loss      0.417 | fused loss     0.4170
  max logit    100.0 | naive p[0]   3.72e-44 | naive loss        100 | fused loss   100.0000
  max logit    700.0 | naive p[0]  9.86e-305 | naive loss        700 | fused loss   700.0000
  max logit   1000.0 | naive p[0]          0 | naive loss        inf | fused loss  1000.0000
```

`exp(709.79)` is the last finite exponential in float64. Past it the naive
denominator is `inf`, so the largest class reads `inf/inf = nan` and every
other class reads `0/inf = 0.0` -- and `-log(0.0)` is `+inf`. Whichever entry
you look at, the run is over, and nothing raised.

The fix for the overflow is an exact identity: subtracting a constant from
every logit leaves the softmax unchanged, so subtract the largest and every
exponent becomes at most 0. The fix for the underflow is to never form a
probability: `L = logsumexp(z) - z_y` is a subtraction of finite numbers.

```
  logits [0, 900], true class 0
  naive p = [ 0. nan]  ->  -log(p[0]) = inf
  fused loss = logsumexp(z) - z_0 = 900.0000, gradient [-1.  1.]
```

A loss of 900 and a gradient whose entries are still inside [-1, 1], because
`p - onehot` is bounded by construction no matter how wrong the prediction was.

![Three panels: p(correct class) from the naive softmax becomes nan at a logit scale of 717 while the stable one continues; the largest exponent handed to exp() grows linearly for the naive version and is pinned at 0 for the stable one, crossing the float64 limit of 709.78; and a confidently wrong prediction sends the unfused loss to +inf while the fused loss grows linearly](figures/05-softmax-stability.png)

---

## Stage 6 — convolution as a layer

`examples/06_convolution.py`

Start on paper. A 4x4 input, a 3x3 vertical-edge kernel, stride 1, no padding,
so the output is `(4-3)/1 + 1 = 2`. The kernel's middle column is zero, so each
row contributes `left - right`:

```
  out[0,0] = (1-3) + (0-2) + (3-1) = -2
```

```
  by hand    : [[-2, -2], [2, -2]]
  loops      : [[-2.0, -2.0], [2.0, -2.0]]
  im2col GEMM: [[-2.0, -2.0], [2.0, -2.0]]
  scipy correlate2d (no flip): [[-2.0, -2.0], [2.0, -2.0]]
  scipy convolve2d  (flips)  : [[2.0, 2.0], [-2.0, 2.0]]
```

Four implementations agree and a fifth disagrees in a specific, informative
way. **What every framework calls convolution is cross-correlation**: the
kernel is not flipped. Look at the index in the definition --

```
out[n, f, i, j] = sum_c sum_a sum_b  xp[n, c, i*s + a, j*s + b] * W[f, c, a, b] + bias[f]
```

-- both spatial indices run forward. Mathematical convolution uses
`x[i - a, j - b]`. `cv2.filter2D` also correlates; `scipy.signal.convolve2d`
genuinely flips. It is irrelevant for learned filters, because backprop simply
learns the flipped kernel, and it matters enormously the moment you port a
hand-designed Sobel from a classical pipeline into a network.

The other sentence worth saying out loud: **the channel axis is summed away,
not slid over.** A "3x3 filter" on a 64-channel input is a 3x3x64 tensor; the
kernel slides over height and width only, and output channels come from having
`C_out` independent filters. `tests/test_conv.py` settles it by building the
reference as a sum of per-channel scipy correlations.

The backward pass, gradient-checked at two stride/padding combinations:

```
  Conv2D(3->4, k3, stride=1, pad=1):  dW 6.34e-12   db 5.38e-11   dX 1.17e-10
  Conv2D(3->4, k3, stride=2, pad=0):  dW 1.69e-11   db 7.20e-12   dX 7.41e-11
```

And the counting arguments that every CNN question opens with:

```
  dense layer, 224x224x3 -> 1000 units : 150,528,000 weights
  conv 3x3, 3 -> 64 channels           : 1,792 weights, and independent of image size
  ratio                                : about 84,000x
  ResNet-18 stem: 224 -> 112, 118,013,952 MACs = 0.236 GFLOPs
  two stacked 3x3 (C=64): 73,728 weights, 5x5 receptive field
  one 5x5         (C=64): 102,400 weights, the same receptive field
  the stack is 28% cheaper AND has one extra nonlinearity in it.
```

![Four panels: a synthetic test card with hard and soft vertical edges and one horizontal edge; the netfs convolution's response; scipy's convolve2d with every colour reversed because it flips the kernel; and the difference against scipy's correlate2d, which is exactly zero everywhere](figures/06-conv-vs-reference.png)

---

## Stage 7 — im2col

`examples/07_im2col_speedup.py`

Nested loops are the honest way to understand convolution and a bad way to
compute it. im2col rewrites the problem as a matrix multiply: gather every patch
the kernel will ever see into the rows of one matrix, reshape the weights, and
multiply once.

```python
def conv2d_im2col(x, w, b=None, stride=1, pad=0):
    cols, ho, wo = im2col(x, kh, kw, stride, pad)
    out = cols @ w.reshape(c_out, -1).T     # THE GEMM
```

The correctness argument is that a patch and a filter are flattened in the
**same** `(channel, kernel row, kernel column)` order. If those two orders ever
drift apart, the convolution computes a permuted kernel, and nothing raises --
which is precisely why `tests/test_conv.py` asserts the im2col result equals the
naive loops exactly at six stride/padding combinations rather than
approximately.

The index trace, printed for the first two output pixels:

```
  output is 2x2, so the column matrix is (4, 9) = (Ho*Wo, kh*kw*Cin)
  output pixel 0 gathers rows [0 0 0 1 1 1 2 2 2] cols [0 1 2 0 1 2 0 1 2]
  output pixel 1 gathers rows [0 0 0 1 1 1 2 2 2] cols [1 2 3 1 2 3 1 2 3]
  cols[0] = [1. 2. 3. 0. 1. 2. 3. 0. 1.]
  cols[0] . kernel.ravel() = -2.0   <- that is out[0,0]
```

### The measurement

Three implementations, so that "faster than what" has an answer:

```
  configuration              full loops    einsum    im2col  vs full  vs einsum
  32x32, 3->16, k3               0.0626    0.0039   0.00046     135x         8x
  64x64, 3->16, k3               0.3114    0.0152   0.00229     136x         7x
  128x128, 3->16, k3             1.0927    0.0636   0.00785     139x         8x
  64x64, 16->32, k3              0.6528    0.0445   0.00887      74x         5x
  batch 8, 28x28, 1->8, k5       0.2041    0.0048   0.00052     389x         9x
  im2col vs the fully looped version : 74x to 389x
  im2col vs the einsum reference     : 5x to 9x
```

Both numbers are reported because quoting only the first would be true and
misleading. The losses and accuracies in this file are bit-identical between
runs; the timings move by 20-30% with machine load, which is why the hardware
and the run count are stated with them. Most of the win is ordinary vectorisation; the GEMM is worth a
further single-digit factor at these sizes. (Agreement to 1e-12 is asserted
before anything is timed -- a benchmark between two functions that compute
different things is a number with nothing attached to it.)

### The cost

Each input element is copied once per output position that reads it, so the
column matrix is roughly `k^2` times the activation:

```
  the standard example: a 224x224x64 float32 activation is 12.8 MB;
  its 3x3 column matrix is 50176 x 576 = 116 MB.
```

That 9x is why real frameworks do not always use im2col -- they keep direct,
Winograd and FFT paths and choose per shape -- and why a `MemoryError` can
appear on an input the naive loop handled comfortably.

![Three panels: forward-pass times on a log axis for three implementations across five shapes; the speedup of im2col against each of the other two; and the memory blow-up of the column matrix, clustered around the k^2 = 9 line for 3x3 kernels and reaching 18x for a 5x5](figures/07-im2col-speedup.png)

### The backward pass, and one specific trap

Once the forward pass is a matrix multiply, the backward pass is stage 3's
dense-layer backward pass followed by undoing the gather:

```python
dflat = dout.transpose(0, 2, 3, 1).reshape(-1, c_out)
self._store("W", (dflat.T @ self.cols).reshape(w.shape))
self._store("b", dflat.sum(axis=0))
dcols = dflat @ w.reshape(c_out, -1)
return col2im(dcols, self.x_shape, self.k, self.k, self.stride, self.pad)
```

`col2im` scatters with `np.add.at` and not with `dxp[..., ii, jj] += dcols`.
NumPy's fancy-index assignment is *buffered*: with repeated indices it applies
one of the duplicate updates and silently discards the rest. Every overlapping
window guarantees repeated indices, so the fast spelling produces a gradient
that is wrong only in the interior of the image and only where windows overlap.
There is no warning. `tests/test_gradients.py` pins it directly: with an
all-ones 3x3 kernel and an all-ones upstream gradient, an interior pixel's
gradient must be exactly 9 and a corner's exactly 4.

---

## Stage 8 — pooling, stride, padding

`examples/08_pooling_shapes.py`

Max pooling has no parameters, halves the spatial size, and its backward pass is
the third flow pattern: **max routes**.

```
  rows 0-1, cols 0-1: max 6 at absolute position (1, 1)
  rows 0-1, cols 2-3: max 4 at absolute position (0, 3)
  rows 2-3, cols 0-1: max 7 at absolute position (2, 0)
  rows 2-3, cols 2-3: max 9 at absolute position (3, 3)
  forward  = [[6.0, 4.0], [7.0, 9.0]]
  upstream = [[1.0, 2.0], [3.0, 4.0]]
  backward (max ROUTES: everything to the winner, nothing to the rest):
     0   0   0   2
     0   1   0   0
     3   0   0   0
     0   0   0   4
  12 of 16 entries are exactly zero.
  the routed total 10 equals the upstream total 10 -- nothing is created or lost.
```

If you did not store the winning positions on the forward pass you cannot do
this at all. That is the same cache argument as stage 3, in a new costume.

The implementation folds the channels into the batch axis and reuses im2col:

```python
folded = x.reshape(n * c, 1, h, w)          # channels into the batch axis
cols, _, _ = im2col(folded, self.k, self.k, self.stride, 0)
self.argmax = np.argmax(cols, axis=1)
```

That is not just code reuse. It says something true: pooling treats each channel
independently, unlike convolution, which sums the channel axis away. And it
means the backward scatter inherits `col2im`'s `np.add.at`, which matters as
soon as `stride < k` and one input can win two windows:

```
  MaxPool2D(k=2, stride=2) dX relative error 1.73e-11
  MaxPool2D(k=3, stride=1) dX relative error 4.15e-11   <- windows overlap, gradients accumulate
  MaxPool2D(k=2, stride=1) dX relative error 2.39e-11   <- windows overlap, gradients accumulate
```

### The off-by-one

```
out = floor( (in + 2*pad - k) / stride ) + 1
```

```
     in   k  pad  stride   out   note
     32   3    1       1    32   'same' padding
     32   3    0       1    30   valid: you lose k-1
     32   3    1       2    16   stride 2 halves it
      7   3    0       2     3   windows at columns 0-2, 2-4, 4-6
      8   3    0       2     3   SAME answer -- column 7 is read by nothing
    224   7    3       2   112   the ResNet-18 stem
```

The floor is where the bugs live, and the script makes it concrete by listing
exactly which input columns any window ever touches:

```
  in=7, k=3, s=2 -> out=3;  columns read: [0, 1, 2, 3, 4, 5, 6];  all read
  in=8, k=3, s=2 -> out=3;  columns read: [0, 1, 2, 3, 4, 5, 6];  NEVER READ: [7]
  in=9, k=3, s=2 -> out=4;  columns read: [0..8];  all read
```

A test suite that only ever feeds 7 and 9 never sees this. Feed it 8 and a whole
column of the image is invisible to the layer, silently.

![Four panels: a 4x4 input with each pooling window's winner circled; the 2x2 pooled output; the backward pass showing the upstream gradient routed to exactly four positions with twelve zeros; and a coverage diagram showing that at k=3 s=2 an input of width 8 has one column no window ever reads](figures/08-pooling-and-shapes.png)

---

## Stage 9 — a CNN, trained

`examples/09_train_cnn.py`

```
layer                 output shape              params
input                 (1, 1, 8, 8)
Conv2D                (1, 8, 8, 8)                  80
ReLU                  (1, 8, 8, 8)                   0
MaxPool2D             (1, 8, 4, 4)                   0
Conv2D                (1, 16, 4, 4)              1,168
ReLU                  (1, 16, 4, 4)                  0
MaxPool2D             (1, 16, 2, 2)                  0
Flatten               (1, 64)                        0
Linear                (1, 10)                      650
TOTAL                                            1,898
```

The dataset is scikit-learn's bundled copy of the UCI optical-digits set: 1797
real handwritten digits at 8x8, ten classes, shipped inside the wheel so nothing
downloads. Split 70/30, stratified, seed 0: 1,258 train and 539 test.

### The gradient check that "fails", and why that is the lesson

```
  on the raw images        : worst relative error 4.29e-01   <- looks broken
  on the same images + 1e-2 of noise: 4.74e-09   <- fine
  fraction of first-layer pre-activations that are EXACTLY 0.0: 10.2%
```

Nothing is broken. An 8x8 digit is mostly blank; with zero padding and a bias
initialised to zero, every patch that is entirely blank has a pre-activation of
exactly 0.0 -- precisely the point where ReLU has no derivative. The analytic
rule picks 0 there (PyTorch also picks 0); the centred difference measures the
chord across the kink and gets 0.5. They disagree because the function is not
differentiable, not because the code is wrong.

The standard test is to re-run the check at a different point: **a bad row that
MOVES is a kink; a bad row that STAYS PUT is a bug.** Here it moves -- a
hundredth of noise on the pixels lifts every pre-activation off the kink and the
error drops by eight orders of magnitude. This is the single case where a
failing gradient check means nothing is wrong, and it is worth having met it
once on real data rather than reading about it.

### The number you write down before pressing go

```
  loss at initialisation : 2.3139
  -ln(1/10) = ln(10)     : 2.3026
  steps per epoch        : 40
```

An untrained classifier spreads its confidence evenly, so every class gets 1/C
and the loss is `ln(C)`. If the first number were far from the second, the bug
would be in the wiring -- labels, normalisation, or the loss being handed the
wrong argument -- and finding that now costs seconds instead of half an hour.
(Getting this right is also why the final layer is initialised at
`weight_scale=0.01` rather than with the He default; see ADR-011.)

### Training

```
  epoch  0  train_loss 2.0634  val_loss 1.6129  val_acc 0.6215  (0.32s)
  epoch  9  train_loss 0.1212  val_loss 0.1421  val_acc 0.9573  (0.31s)
  epoch 19  train_loss 0.0387  val_loss 0.1013  val_acc 0.9647  (0.28s)
  epoch 29  train_loss 0.0130  val_loss 0.0803  val_acc 0.9814  (0.19s)

  CNN: 1,898 parameters
  training time            : 9.8 s for 30 epochs (0.27 s/epoch)
  final train accuracy     : 0.9992
  final TEST accuracy      : 0.9814   (loss 0.0803)
  MLP baseline             : 0.9647 with 1,810 parameters
```

The dense baseline is there because "the CNN is better" is not worth saying
without the other number. At a nearly identical parameter count the convolution
is worth about 1.7 accuracy points here -- a real but modest margin, which is
what you should expect on 8x8 images where the spatial prior has little room to
pay off. On 224x224 photographs the gap is enormous. Reporting the small number
honestly is more useful than implying the large one.

![Three panels: CNN train and test loss falling from ln(10) over 30 epochs; test accuracy for the CNN and a parameter-matched dense MLP, ending at 0.9814 and 0.9647; and accuracy against wall-clock seconds, showing the whole run takes about ten seconds of pure NumPy](figures/09-cnn-training.png)

### What it learned, including one thing it did not

```
  first-layer filters that never fire on ANY test image: 1 of 8  (filters [1])
```

One of the eight first-layer filters is dead. Its pre-activation is negative for
every image in the test set, so it outputs zero everywhere, so it receives
exactly zero gradient, so no optimiser can revive it. The model reached 98.14%
with seven working filters and one piece of dead weight. That is the dying-ReLU
problem measured in a network that was actually trained, and it is the argument
for LeakyReLU in one line of output.

![Two rows: the eight 3x3 kernels the first conv layer learned, one of them flagged dead, and underneath each one its response to a single input digit -- the dead filter's response is uniformly black](figures/10-learned-filters.png)

### The confusion matrix

```
  confusion matrix -- rows are truth, columns are the guess
         0      1      2      3      4      5      6      7      8      9
  0     52      0      0      0      1      0      0      0      0      0
  1      0     54      0      0      0      0      0      0      1      0
  2      0      1     52      0      0      0      0      0      0      0
  3      0      0      0     54      0      0      0      0      1      0
  4      0      0      0      0     54      0      0      0      0      0
  5      0      0      0      0      0     54      0      0      0      1
  6      0      0      0      0      0      0     54      0      0      0
  7      0      0      0      0      0      0      0     54      0      0
  8      0      3      0      0      0      0      0      0     49      0
  9      0      0      0      0      0      1      0      0      1     52
  trace / total = 529/539 = 0.9814
```

Rows are truth, columns are the guess, so the diagonal is correct and every
off-diagonal entry is one specific, nameable mistake. The largest is three 8s
called 1 -- and looking at the actual images, they are 8s drawn with a narrow
waist. The mistakes are the mistakes a person would make, which is the
strongest available evidence that the model learned something about digits
rather than something about the dataset.

![A ten-class confusion matrix with 529 of 539 on the diagonal, beside every one of the ten misclassified digits with its true label, predicted label and the confidence the model assigned](figures/11-confusion-matrix.png)

---

## Stage 10 — bridge: from classifier to detector

Nothing in this section is implemented here. It is the shape of the next step,
written down so that the last stage does not end in mid-air.

A classifier answers a question with a fixed-size answer: one score per class,
always the same shape, batchable, comparable to a label element by element. A
**detector** answers with a **set**: `(box, class, score)` triples, three of them
for one photo, forty-one for the next, none for the one after that. Every
awkward thing in detection follows from that one difference -- no fixed output
shape, no canonical ordering, and no one-to-one correspondence between
predictions and ground truth unless you build one.

**Dense prediction is the bridge.** The last spatial feature map in stage 9's
CNN is `(16, 2, 2)` and gets flattened into one vector, which throws the
positions away. Do not flatten it. Attach a small head that runs at *every*
spatial position of that map, and each position now emits its own prediction for
the region of the input it can see -- its receptive field, computed by exactly
the arithmetic in stage 8. A classifier becomes a detector by replacing "one
answer per image" with "one answer per cell", and the convolution's translation
equivariance is what makes that legitimate: shift the input and the feature map
shifts identically, so a head that works at one position works at all of them.

**What each cell predicts** has changed twice in the field's short history, and
naming the generation before answering is what separates a real answer from a
memorised one:

- **Anchor-based** (the YOLOv3 era). Each cell carries a few fixed prior box
  shapes and predicts a *correction* to each prior -- offsets for the centre and
  multiplicative scale factors for the size -- plus an "objectness" score and a
  class distribution. The priors exist because regressing an absolute box size
  from scratch is hard and correcting a sensible guess is easy.
- **Anchor-free** (FCOS, YOLOX, YOLOv8, YOLO11). The priors vanish. Each cell is
  a *point*, and the head regresses four distances from that point to the box's
  left, top, right and bottom edges. Simpler, fewer hyperparameters, and now the
  default. Whether a separate objectness branch survives depends on the model,
  and getting that wrong shifts every channel you decode by one.

**Then the duplicates.** Training marks several cells near an object as positive
-- one positive per object gives far too little gradient signal -- so at
inference several of them fire, with near-identical boxes. **Non-maximum
suppression** is the greedy cleanup: sort by score, keep the best box, delete
everything overlapping it beyond an IoU threshold, repeat. It is not a model
component; it is a post-processing pass that exists because of a training
choice, and the current direction of the field is to remove the cause instead --
one-to-one label assignment during training (DETR's Hungarian matching,
YOLOv10's parallel one-to-one head) means no duplicates appear and no
suppression is needed. The practical reason people care is not accuracy: NMS is
non-differentiable, has a data-dependent output shape, and therefore usually
cannot live inside an exported ONNX or TensorRT graph.

**And then the measurement changes too.** Accuracy is meaningless for a set:
there is no denominator. Detection is scored with intersection-over-union to
decide which prediction matches which ground-truth box, precision and recall
computed at every score threshold, and the area under that precision-recall
curve (average precision) averaged over classes and IoU thresholds.

That measured side of the story is the subject of the portfolio's
[`object-detection-benchmark`](https://github.com/Pratyush150/object-detection-benchmark)
repository, which fine-tunes a detector on a small licensed dataset and reports
mAP, AP_small, and median and p95 latency on named hardware. This repository
stops at the point where you know exactly what the boxes and scores coming out
of that model *are*, and how the gradients that produced them were computed.
