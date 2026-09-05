# DERIVATIONS

Every gradient used by `src/netfs/` is derived here, in order, in plain text.
No LaTeX: the maths is written the way you would write it on paper, because
that is the form you need to be able to reproduce.

If a line of code disagrees with a derivation here, one of the two is wrong and
it is worth finding out which before you change anything. Every result below is
also asserted numerically in `tests/test_gradients.py`.

**Contents**

1. [Notation and the one convention that matters](#1-notation)
2. [One neuron, squared error](#2-one-neuron-squared-error)
3. [Why stacked linear layers collapse](#3-why-stacked-linear-layers-collapse)
4. [The dense layer, batched](#4-the-dense-layer-batched)
5. [Activations](#5-activations)
6. [Softmax and cross-entropy, and why they are fused](#6-softmax-and-cross-entropy)
7. [Convolution: forward, dW, db, dX](#7-convolution)
8. [Max pooling and global average pooling](#8-pooling)
9. [The numerical gradient, and why it is centred](#9-the-numerical-gradient)
10. [Output-size arithmetic](#10-output-size-arithmetic)
11. [Optimisers](#11-optimisers)

---

## 1. Notation

```
X       input,  shape (N, D_in)      N samples, ONE SAMPLE PER ROW
W       weights of a dense layer, shape (D_out, D_in)   ONE NEURON PER ROW
b       bias,   shape (D_out,)
Z       pre-activation,  Z = X @ W.T + b,   shape (N, D_out)
H       activation,      H = f(Z)
L       the loss, a single number
dA      shorthand for dL/dA, and it always has exactly A's shape
@       matrix multiply
*       elementwise multiply
.T      transpose
```

**The one convention that matters.** There are two ways to write a linear layer
and both are correct:

```
single sample, column vector :   z = W x + b        x is (D_in,)
batched, row-major           :   Z = X @ W.T + b    X is (N, D_in)
```

This package uses the second one everywhere, because it is what PyTorch uses
and because it makes the batch dimension explicit. Mixing the two inside one
derivation is the single most expensive mistake available here, and the reason
is that it does not raise: NumPy will broadcast an `(N, N)` matrix into
existence where you wanted `(N, D_out)` and hand you a plausible, wrong answer.

**The one rule that is backpropagation.** For any node in the graph:

```
gradient flowing out of a node = upstream gradient x that node's local derivative
```

The upstream gradient is dL/d(this node's output), handed to you by the node
downstream, which has already run. The local derivative is d(output)/d(input),
evaluated at the values cached on the forward pass. That is the chain rule used
as a mechanical procedure rather than quoted as a theorem, and it is applied in
reverse topological order: never visit a node until everything consuming its
output has been visited.

**The three patterns that cover most of a network:**

- **Addition distributes.** `c = a + b` sends the upstream gradient unchanged to
  both inputs. A router that copies.
- **Multiplication swaps.** `c = a * b` sends `upstream * b` to `a` and
  `upstream * a` to `b`. Each input's gradient is scaled by *the other* input,
  which is why the forward pass must cache both.
- **Max routes.** `c = max(a, b)` sends the whole upstream gradient to whichever
  input won and exactly zero to the loser. A switch, not a splitter.

**Multi-path gradients ADD.** If a value is consumed by two downstream
operations, its gradient is the sum of the contributions coming back along both
paths. Not the last one, not the average. In code that is the difference
between `grad = ...` and `grad += ...`, and getting it wrong silently drops a
path with no error message. It is the reason a bias gradient is a sum over the
batch (section 4) and the reason `col2im` scatters with `np.add.at` (section 7).

**Shape discipline.** dL/dW has exactly W's shape, always -- there is one
partial derivative per entry. If a derivation hands you the transpose, you
multiplied in the wrong order. `Layer._store` asserts this on every backward
pass.

---

## 2. One neuron, squared error

The smallest complete example, and the one in `examples/01_one_neuron.py`.
One input x, one weight w, one bias b, one target y, N samples:

```
prediction   p_i = w*x_i + b
loss         L   = (1/N) * sum_i (p_i - y_i)^2
```

Work the chain in two steps. First the loss with respect to the prediction:

```
dL/dp_i = (1/N) * 2*(p_i - y_i)
```

Then the prediction with respect to each parameter -- these are the *local*
derivatives, and they are trivial:

```
dp_i/dw = x_i
dp_i/db = 1
```

Multiply and sum over the samples, because w appears in all N predictions and a
variable used N times collects N contributions:

```
dL/dw = (1/N) * sum_i 2*(p_i - y_i)*x_i
dL/db = (1/N) * sum_i 2*(p_i - y_i)
```

Two things to notice, both of which appear again at full scale:

- The factor 2 is real. Dropping it does not move the minimum, so the model
  still trains -- at half the learning rate you think you set. That is exactly
  the class of bug that is invisible without a reference to compare against.
- `dL/db` is `dL/dw` with `x_i` replaced by 1, because the bias's local
  derivative is 1. Every bias gradient in this package is the layer's
  pre-activation gradient summed over everything the bias was broadcast across.

---

## 3. Why stacked linear layers collapse

Two linear layers with nothing between them:

```
y = W2 (W1 x + b1) + b2
  = W2 W1 x + W2 b1 + b2
  = W_eff x + b_eff        with W_eff = W2 W1,  b_eff = W2 b1 + b2
```

This is an identity, not an approximation. Stack a hundred linear layers and
you still have one matrix multiply: you have spent a hundred layers' worth of
parameters to buy exactly zero extra expressive power.

**The XOR consequence.** XOR is 0,1,1,0 on the corners (0,0), (0,1), (1,0),
(1,1). Any affine function f satisfies

```
f(0,0) + f(1,1) = f(0,1) + f(1,0)
```

(both sides equal `2*b + w1 + w2`). XOR's targets give `0 + 0 = 0` on the left
and `1 + 1 = 2` on the right. The constraint is violated by 2, so no affine
function can fit XOR -- and the best it can do is split that discrepancy evenly:
an error of 1/2 at each of the four corners, giving a mean squared error of
`4 * (1/2)^2 / 4 = 0.25`.

That 0.25 is a floor derived on paper, not a number that a training run
happened to reach. `examples/02_why_nonlinearity.py` trains a two-layer linear
stack on XOR and it converges to 0.25000; adding one ReLU between the same two
layers and changing nothing else drives it below 1e-28.

**The hand-set network that does work:**

```
W1 = [[1, 1], [1, 1]]   b1 = [0, -1]      2 ReLU hidden units
W2 = [1, -2]            b2 = 0            1 linear output
```

```
x=(0,0):  z1 = [ 0, -1] -> h = [0, 0] -> y = 0
x=(0,1):  z1 = [ 1,  0] -> h = [1, 0] -> y = 1
x=(1,0):  z1 = [ 1,  0] -> h = [1, 0] -> y = 1
x=(1,1):  z1 = [ 2,  1] -> h = [2, 1] -> y = 1*2 + (-2)*1 = 0
```

0, 1, 1, 0 from four hand-set weights and no training. Delete the ReLU and the
same weights collapse to `W_eff = [-1, -1]`, `b_eff = 2`, i.e.
`y = -x1 - x2 + 2`, which outputs 2, 1, 1, 0.

---

## 4. The dense layer, batched

```
forward:   Z = X @ W.T + b
           X (N, D_in),  W (D_out, D_in),  b (D_out,),  Z (N, D_out)
```

Write one entry with explicit indices, because that is where the transposes
come from:

```
Z[n, j] = sum_k X[n, k] * W[j, k] + b[j]
```

**dL/dW.** W[j, k] appears in `Z[n, j]` for every sample n, so it is a
multi-path variable and the contributions sum over n:

```
dL/dW[j, k] = sum_n dZ[n, j] * (dZ[n,j]/dW[j,k]) = sum_n dZ[n, j] * X[n, k]
```

That sum over the shared index n is a matrix multiply with n contracted:

```
dW = dZ.T @ X                 (D_out, N) @ (N, D_in) -> (D_out, D_in)  = W's shape
```

**dL/db.** b[j] appears in `Z[n, j]` for every n, with local derivative 1:

```
dL/db[j] = sum_n dZ[n, j]     ->   db = dZ.sum(axis=0)     -> (D_out,)  = b's shape
```

The `sum(axis=0)` is not optional and it is not a mean. Leave it out and `db`
keeps shape `(N, D_out)`, which then broadcasts during the update and turns the
bias into a matrix -- the parameter silently changes shape between iterations.
(With a mean-reduction loss the 1/N is already inside dZ, so `dZ.sum(axis=0)`
and `dZ.mean(axis=0)` differ by exactly the factor that the loss reduction
already applied. Both spellings appear in the wild, for different losses.)

**dL/dX**, needed to continue backwards into the previous layer. X[n, k]
appears in `Z[n, j]` for every output unit j:

```
dL/dX[n, k] = sum_j dZ[n, j] * W[j, k]    ->   dX = dZ @ W     -> (N, D_in)
```

Three lines, and the shapes force them: there is exactly one way to multiply
`(N, D_out)` and `(N, D_in)` into `(D_out, D_in)`.

**The interview sentence.** Notice that the true derivative dZ/dW is a rank-4
tensor with `N * D_out * D_out * D_in` entries and is almost entirely zeros. It
is never formed. `dW = dZ.T @ X` is a vector-Jacobian product: one matmul,
about the cost of the forward pass. Every autograd framework is a VJP engine,
not a Jacobian engine, and that is the whole reason reverse-mode
differentiation is affordable -- one backward sweep for one scalar output,
regardless of how many parameters there are.

### The worked 2-2-1 example

The example in `examples/03_backprop_mlp.py`, small enough to check with a
pencil. One sample:

```
x = [1, 2]   y = 1
W1 = [[0.5, -1.0],   b1 = [0, -1]     W2 = [[1.5, -0.5]]   b2 = [0.5]
      [1.0,  1.0]]
```

Forward:

```
z1 = x @ W1.T + b1 = [0.5*1 + (-1)*2 + 0,  1*1 + 1*2 - 1] = [-1.5, 2.0]
h  = relu(z1)      = [0.0, 2.0]                       <- unit 0 is switched off
z2 = h @ W2.T + b2 = 1.5*0 + (-0.5)*2 + 0.5 = -0.5
L  = (z2 - y)^2    = (-1.5)^2 = 2.25
```

Backward:

```
dz2 = 2*(z2 - y) = -3.0                       the seed
dW2 = dz2.T @ h  = [-3.0*0.0, -3.0*2.0] = [[0.0, -6.0]]
db2 = dz2        = [-3.0]
dh  = dz2 @ W2   = [-3.0*1.5, -3.0*(-0.5)] = [-4.5, +1.5]
dz1 = dh * (z1 > 0) = [-4.5*0, 1.5*1] = [0.0, 1.5]
dW1 = dz1.T @ x  = [[0.0*1, 0.0*2], [1.5*1, 1.5*2]] = [[0.0, 0.0], [1.5, 3.0]]
db1 = dz1        = [0.0, 1.5]
```

**The answer sheet:**

```
dW1 = [[0.0, 0.0], [1.5, 3.0]]      db1 = [0.0, 1.5]
dW2 = [[0.0, -6.0]]                 db2 = [-3.0]
```

The whole first row of dW1 is exactly zero, along with db1[0], because that
neuron's ReLU was off. A dead unit receives precisely zero gradient -- not
small, zero -- so no optimiser can move its incoming weights, so its
pre-activation cannot change, so it can never revive on its own. That is the
dying-ReLU problem in four numbers, and `examples/09_train_cnn.py` reports one
real instance of it in a trained network.

---

## 5. Activations

All of these are elementwise, so the local derivative is elementwise too and
the backward pass is a multiply rather than a matmul.

**ReLU.** `f(z) = max(0, z)`

```
f'(z) = 1  if z > 0
        0  if z < 0
        undefined at z = 0     -- NumPy's (z > 0) gives 0; PyTorch also gives 0
```

```
dZ = dH * (Z > 0)
```

The undefined point is not pedantry. `examples/09_train_cnn.py` finds that
10.2% of the first layer's pre-activations on real 8x8 digit images are exactly
0.0 at initialisation -- an image that is mostly blank, zero padding, and a
bias initialised to zero make a lot of exactly-zero patches -- and that is
enough to make a naive gradient check on raw data look broken when nothing is.

**Sigmoid.** `f(z) = 1 / (1 + exp(-z))`

```
f'(z) = f(z) * (1 - f(z))          maximum 0.25, at z = 0
dZ = dH * H * (1 - H)              expressed in the OUTPUT, so cache that
```

Derivation: write `f = (1 + e)^-1` with `e = exp(-z)`, so
`f' = -(1+e)^-2 * de/dz = (1+e)^-2 * e`. Then
`f(1-f) = (1+e)^-1 * (e/(1+e)) = e (1+e)^-2`, the same thing.

That maximum of 0.25 is the seed of the vanishing-gradient story: ten chained
sigmoids multiply the gradient by at most `0.25^10 = 9.5e-7`.

The textbook formula overflows. `1/(1+exp(-z))` at z = -800 evaluates
`exp(800)`, which is `inf` in float64, and returns 0.0 after a warning. The
stable form branches on the sign so the exponent is never positive:

```
z >= 0:  1 / (1 + exp(-z))
z <  0:  exp(z) / (1 + exp(z))
```

**tanh.** `f(z) = tanh(z)`, `f'(z) = 1 - tanh(z)^2`, maximum 1.0 at z = 0, and
zero-centred. The zero-centring matters because `dW = dZ.T @ X`: if every entry
of X is positive (as it is for a sigmoid's output), every entry of dW shares
the sign of dZ, and the weight vector can only move along one diagonal at a
time. That is the classic zig-zag descent path.

---

## 6. Softmax and cross-entropy

**Forward.**

```
p_i = exp(z_i) / sum_j exp(z_j)
L   = -log(p_y)                     y is the index of the true class
```

**The shift identity.** For any constant c,

```
exp(z_i - c) / sum_j exp(z_j - c) = exp(-c) exp(z_i) / (exp(-c) sum_j exp(z_j)) = p_i
```

The result is unchanged for ANY c. Choosing `c = max_j z_j` makes the largest
exponent exactly 0, so every call to exp lands in (0, 1] and cannot overflow.
Underflow of the small terms to 0.0 is harmless -- they were negligible --
while overflow of the large terms to inf is fatal, because `inf/inf = nan`.

**The fused loss.** Take the log of the softmax directly:

```
L = -log( exp(z_y) / sum_j exp(z_j) )
  = -z_y + log sum_j exp(z_j)
  = logsumexp(z) - z_y
```

and compute logsumexp with the same shift:

```
logsumexp(z) = m + log sum_j exp(z_j - m),   m = max_j z_j
```

No probability is ever formed. That kills both failure modes at once:

- **Overflow:** exp is never called on a positive number.
- **Underflow:** the unfused route computes `p_y` and then `-log(p_y)`. A
  confidently wrong prediction underflows `p_y` to exactly 0.0 and
  `-log(0.0) = +inf`, which poisons every parameter on the next update. The
  fused form is a subtraction of two finite numbers, so a logit gap of 900
  simply gives a loss of 900.

**The gradient.** This is the derivation worth doing once by hand, because the
answer is so much simpler than the intermediate steps.

```
L = -log p_y,   p_i = exp(z_i) / S,   S = sum_j exp(z_j)
```

Differentiate `p_i` with respect to `z_k`, two cases:

```
i = k:   dp_i/dz_i = [exp(z_i) S - exp(z_i) exp(z_i)] / S^2 = p_i (1 - p_i)
i != k:  dp_i/dz_k = [0 - exp(z_i) exp(z_k)] / S^2          = -p_i p_k
```

Both cases at once: `dp_i/dz_k = p_i (delta_ik - p_k)`.

Now `L = -log p_y`, so `dL/dp_y = -1/p_y`, and by the chain rule:

```
dL/dz_k = (-1/p_y) * dp_y/dz_k
        = (-1/p_y) * p_y (delta_yk - p_k)
        = p_k - delta_yk
```

**The p_y cancels.** That single cancellation is the whole argument for fusing
the two operations: the unfused route computes `1/p_y` explicitly, and `p_y` is
exactly the quantity that just underflowed to zero. Composed algebraically
first, it never appears.

```
dL/dz = p - onehot(y)        (divided by N for a mean-reduction loss)
```

**A free correctness check, valid for every input.** `p` sums to 1 and the
one-hot sums to 1, so every row of the gradient sums to exactly 0. Probability
mass is conserved: the gradient can only MOVE mass between classes, never
create it. If your softmax gradient rows do not sum to zero, stop and read the
code rather than tuning the learning rate.

Note also that every entry of `p - onehot` lies in [-1, 1] by construction, no
matter how wrong the prediction was. The loss can be 900; the gradient cannot
exceed 1 in magnitude.

**Worked example.** `z = [2.0, 1.0, 0.1]`, true class 0:

```
exp: 7.3891, 2.7183, 1.1052    sum = 11.2125
p  = [0.6590, 0.2424, 0.0986]  (sums to 1.0000)
L  = -ln(0.6590) = 0.4170
dL/dz = [0.6590 - 1, 0.2424, 0.0986] = [-0.3410, +0.2424, +0.0986]   (sums to 0)
```

**The loss at initialisation.** An untrained classifier spreads its confidence
evenly, so `p_i = 1/C` for every class and `L = -ln(1/C) = ln(C)`. That is
2.3026 for ten classes and 1.0986 for three. It is a number you can write down
before pressing go, and a first loss far from it means the labels, the
normalisation or the loss's arguments are wrong -- not the architecture.

---

## 7. Convolution

### Forward

```
out[n, f, i, j] = sum_c sum_a sum_b  xp[n, c, i*s + a, j*s + b] * W[f, c, a, b]  +  bias[f]
```

where `xp` is the zero-padded input, `s` is the stride, and the sums run over
input channels c and kernel positions (a, b).

**Both spatial indices run forward.** Mathematical convolution flips the
kernel and uses `x[i - a, j - b]`. What every deep learning framework calls
convolution -- and what `cv2.filter2D` computes -- is cross-correlation, with
no flip. `scipy.signal.convolve2d` genuinely does flip, and every sign of its
output is opposite. This is irrelevant for learned filters (backprop simply
learns the flipped kernel) and it matters enormously the moment you port a
hand-designed Sobel from a classical pipeline into a network.

**The channel axis is summed away, not slid over.** A "3x3 filter" on a
64-channel input is a 3x3x64 tensor. The kernel slides over height and width
only; depth is consumed entirely, every time. Output channels come from having
`C_out` independent filters. That is why the parameter count is
`k*k*C_in*C_out + C_out` with no depth stride in it.

### Worked by hand

Input 4x4, kernel 3x3, stride 1, no padding, so the output is
`(4-3)/1 + 1 = 2`:

```
input                kernel (vertical edge)
1  2  3  0            1  0 -1
0  1  2  3            1  0 -1
3  0  1  2            1  0 -1
2  3  0  1
```

The middle column of the kernel is zero, so each kernel row contributes
`left - right`:

```
out[0,0]: rows 0-2, cols 0-2 -> (1-3) + (0-2) + (3-1) = -2
out[0,1]: rows 0-2, cols 1-3 -> (2-0) + (1-3) + (0-2) = -2
out[1,0]: rows 1-3, cols 0-2 -> (0-2) + (3-1) + (2-0) = +2
out[1,1]: rows 1-3, cols 1-3 -> (1-3) + (0-2) + (3-1) = -2

ANSWER: [[-2, -2], [2, -2]]
```

Flip the kernel 180 degrees and `out[0,0]` becomes `+2`. That sign flip in four
numbers is the proof that a convolution layer computes a correlation.

### im2col

Nested loops are the honest way to understand convolution and a bad way to
compute it. im2col rearranges the problem into one matrix multiply:

- there are `N*Ho*Wo` output positions; each reads a patch of `C_in*kh*kw`
  numbers;
- gather all of those patches into a matrix `cols` of shape
  `(N*Ho*Wo, C_in*kh*kw)`, one patch per row;
- reshape the weights to `(C_out, C_in*kh*kw)`;
- then `out = cols @ W.reshape(C_out, -1).T + b`, one GEMM.

The correctness argument is that the flattening order of a patch and the
flattening order of a filter are the same `(c, a, b)` order. If they ever drift
apart, the convolution computes a permuted kernel and nothing raises. The
column indices are:

```
f in [0, C_in*kh*kw):
    c = f // (kh*kw)        which input channel
    a = (f // kw) % kh      which kernel row
    b = f % kw              which kernel column

r in [0, Ho*Wo):
    row of xp    = (r // Wo)*stride + a
    column of xp = (r %  Wo)*stride + b
```

**Cost.** Each input element is copied once per output position that reads it,
so `cols` is roughly `k^2` times the activation. A 224x224x64 float32
activation is 12.8 MB; its 3x3 column matrix is `50176 x 576 = 116 MB`, a 9x
blow-up. That is why real frameworks keep direct, Winograd and FFT paths as
well and choose per shape, and why a MemoryError can appear on an input the
naive loop handled comfortably.

### Backward

Once the forward pass is `out_flat = cols @ Wr.T + b` with
`Wr = W.reshape(C_out, -1)`, the convolution backward pass is just section 4's
dense-layer backward pass, followed by undoing the gather.

Let `dflat` be the upstream gradient reshaped to `(N*Ho*Wo, C_out)` -- the
exact inverse of the reshape and transpose the forward pass ended with.

```
dWr   = dflat.T @ cols          (C_out, N*Ho*Wo) @ (N*Ho*Wo, K) -> (C_out, K)
dW    = dWr.reshape(W.shape)
db    = dflat.sum(axis=0)       -> (C_out,)
dcols = dflat @ Wr              (N*Ho*Wo, C_out) @ (C_out, K)   -> (N*Ho*Wo, K)
dX    = col2im(dcols)
```

**dW sums over every position the filter visited.** That is weight sharing seen
from the backward side: one filter used at 4096 positions collects 4096
contributions to its gradient, which is why conv layers learn from far less
data than their dense equivalents.

**db sums over the batch and both spatial axes**, for the same reason: the same
scalar bias was added at every position of every sample.

**col2im is a scatter-ADD, and the ADD is the whole point.** An interior pixel
of a 3x3 stride-1 convolution is read by nine output positions, so it is a
multi-path variable and its gradient is the sum of nine contributions. In code
this must be `np.add.at`, not `dxp[..., ii, jj] += dcols`: buffered fancy-index
assignment with repeated indices applies ONE of the duplicate updates and
silently discards the rest. The resulting gradient is wrong only in the
interior, only where windows overlap -- a magnificent bug to find by reading
code, and a trivial one to find with a gradient check.

Finally, the padded border receives gradient too, and it is discarded. Those
entries are constants we invented; nobody can change them.

**Verification** (`tests/test_gradients.py`): with an all-ones 3x3 kernel,
pad 1, and an all-ones upstream gradient, `dX` at an interior pixel is exactly
9 and at a corner exactly 4 -- the corner's other five windows fell in the
padding.

### Counting

```
dense, 224x224x3 -> 1000 units : 150,528 * 1000 = 150,528,000 weights, ONE layer
conv 3x3, 3 -> 64 channels     : 3*3*3*64 + 64  = 1,792 weights
ratio                          : about 84,000x, and the conv number does not
                                 depend on the image size AT ALL
```

MACs (one multiply-accumulate per output element per kernel weight; 2 FLOPs per
MAC). ResNet-18's stem, in = 224, k = 7, pad = 3, stride = 2:

```
out  = (224 + 6 - 7)//2 + 1 = 112
MACs = 112*112*64 * (7*7*3) = 118,013,952  =  0.236 GFLOPs
```

The stacked-3x3 argument, at 64 channels in and out:

```
two 3x3   : 2 * 3*3*64*64 =  73,728       one 5x5 : 25*64*64 = 102,400   -> 28% fewer
three 3x3 : 3 * 3*3*64*64 = 110,592       one 7x7 : 49*64*64 = 200,704   -> 45% fewer
```

Same receptive field, fewer parameters, and one extra nonlinearity per extra
layer.

Receptive field, computed back to front from `r = 1`:

```
r_{i-1} = (r_i - 1) * stride_i + k_i
```

Two stacked 3x3s at stride 1 give 5; three give 7.

---

## 8. Pooling

### Max pooling

```
forward:  out[n, c, i, j] = max over the window of x[n, c, i*s + a, j*s + b]
```

The local derivative is 1 for the input that won and 0 for every other input in
the window, so:

```
backward: dx[argmax position] += dout[n, c, i, j],   everything else 0
```

`+=` and not `=`, because with `stride < k` the windows overlap and one input
can win two windows -- multi-path again.

**Worked by hand.** 4x4 input, 2x2 window, stride 2:

```
1  3  2  4        windows and winners:
5  6  1  2          rows 0-1, cols 0-1 -> 6 at (1,1)
7  2  8  3          rows 0-1, cols 2-3 -> 4 at (0,3)
1  4  2  9          rows 2-3, cols 0-1 -> 7 at (2,0)
                    rows 2-3, cols 2-3 -> 9 at (3,3)

forward  = [[6, 4], [7, 9]]
```

Push an upstream gradient of `[[1, 2], [3, 4]]` back:

```
[[0, 0, 0, 2],
 [0, 1, 0, 0],
 [3, 0, 0, 0],
 [0, 0, 0, 4]]
```

Twelve of the sixteen entries are exactly zero. If you did not store the
winning positions on the forward pass, you cannot do this at all -- the same
cache argument as section 4, in a new costume.

**Ties.** If two values in a window are equal, `argmax` takes the first in flat
order and the entire gradient goes to it. It is not split. Rarely
consequential; it is why two "identical" runs on integer-valued data can
diverge.

**The window that does not divide the input** drops the remainder: 5x5 with a
2x2 stride-2 window gives 2x2, and the last row and column are read by nothing.

### Global average pooling

```
forward:  out[n, c] = (1/(H*W)) * sum_{i,j} x[n, c, i, j]
backward: dx[n, c, i, j] = dout[n, c] / (H*W)
```

Each input contributed 1/(H*W) of the output, so each receives 1/(H*W) of the
gradient. Contrast with max pooling, where one input gets everything.

This is the layer that makes a backbone resolution-independent: `Flatten`
hard-codes the training resolution into the next linear layer's weight shape,
while global average pooling produces a C-vector at any H and W.

---

## 9. The numerical gradient

Taylor-expand the loss around a parameter value in both directions:

```
L(t + h) = L(t) + h L'(t) + (h^2/2) L''(t) + (h^3/6) L'''(t) + ...
L(t - h) = L(t) - h L'(t) + (h^2/2) L''(t) - (h^3/6) L'''(t) + ...
```

Subtract. The `L(t)` and `h^2` terms cancel:

```
L(t + h) - L(t - h) = 2h L'(t) + (h^3/3) L'''(t) + ...

L'(t) = [L(t + h) - L(t - h)] / (2h)  +  O(h^2)
```

The one-sided difference `[L(t+h) - L(t)]/h` leaves the `h^2/2 * L''` term
uncancelled and so has error `O(h)`. The centred version costs one extra
forward pass and buys an order of accuracy: at `h = 1e-5` that is roughly the
difference between 5 correct digits and 10.

**Why h must not be too small.** The truncation error shrinks as `h^2`, and the
floating-point cancellation error grows as `1/h`: `L(t+h)` and `L(t-h)` agree
in their leading digits, and subtracting them discards those digits. In float64
the total error is minimised somewhere around `h = 1e-6` to `1e-4`.
`examples/04_gradient_check.py` sweeps h across nine orders of magnitude and
plots the U-curve; on this machine the minimum is at `h = 3.2e-5` with a
relative error of 3.4e-11, and at `h = 1e-13` the error is 9.4e-3.

**If your relative error grows as you shrink h, stop shrinking.** You are on
the cancellation branch, and the code is probably fine.

**Cost.** Two forward passes per parameter. Nine parameters is eighteen forward
passes; a hundred million parameters is two hundred million forward passes per
step, which is not a slow method but an impossible one. Backpropagation gets
all of them in one backward sweep costing about as much as one forward pass.
Gradient checking is how you earn the right to trust that trade.

**Kinks.** If a perturbation of size h flips a ReLU across zero, the numeric
gradient measures a chord across the kink and disagrees with the analytic one
for a good reason. At exactly `z = 0`, the analytic rule gives 0 and the centred
difference gives `(relu(h) - relu(-h))/(2h) = 0.5`. Neither is wrong; the
function has no derivative there. The standard test is to re-run the check at a
different point: **a bad row that MOVES is a kink; a bad row that STAYS PUT is a
bug.**

**Thresholds** (CS231n's, and about right): below 1e-7, be happy; 1e-4 is
acceptable only for objectives with kinks in them; above 1e-2 is a bug.

---

## 10. Output-size arithmetic

```
out = floor( (in + 2*pad - dilation*(k - 1) - 1) / stride ) + 1
```

which at dilation 1 is the familiar

```
out = floor( (in + 2*pad - k) / stride ) + 1
```

Read it as: `in + 2*pad` positions exist after padding; the window physically
reaches `dilation*(k-1) + 1` of them; the subtraction leaves the number of
*extra* positions the window can slide into; dividing by the stride counts the
slides; the `+1` is the window's own starting position, which needs no slide.

Hand-computed cases (all asserted in `tests/test_shapes.py`):

```
(32, k3, p1, s1) -> 32     'same' padding
(32, k3, p0, s1) -> 30     valid: you lose k-1
(32, k3, p1, s2) -> 16     stride 2 halves it
( 7, k3, p0, s2) ->  3     windows at columns 0-2, 2-4, 4-6
( 8, k3, p0, s2) ->  3     SAME answer -- column 7 is read by NO window
(224, k7, p3, s2) -> 112   the ResNet-18 stem
```

The floor is where the bugs live. On an input of 8 the last column vanishes
silently: no warning, no error, just a spatial dimension that drifts by one
against a reference implementation somewhere deep in a network, with the
exception surfacing three layers after the mistake.

'Same' padding solves `in = (in + 2p - k) + 1` for p, giving `p = (k-1)/2`,
which is an integer only for odd k. That is the real reason essentially every
kernel you meet is 3x3, 5x5 or 7x7: an even kernel cannot be centred on a
pixel, so 'same' padding has to be asymmetric and implementations disagree
about which side gets the extra row.

---

## 11. Optimisers

**Plain gradient descent.** `w <- w - lr * dL/dw`. The gradient points uphill,
so the step goes against it.

On a quadratic `L(w) = (2w - 6)^2 = 4w^2 - 24w + 36`:

```
dL/dw = 8w - 24        zero at w = 3
L''   = 8              the curvature, constant
```

The update `w <- w - lr(8w - 24)` rearranges to `w <- w(1 - 8*lr) + 24*lr`, so
the distance to the optimum is multiplied by `(1 - lr*L'')` every step.
Convergence therefore requires

```
|1 - lr*L''| < 1     i.e.    0 < lr < 2/L''   = 0.25 here
optimal:  lr = 1/L'' = 0.125, which lands exactly on the optimum in ONE step
```

Starting at `w = 0`:

```
lr = 0.05   factor +0.60 : 1.2, 1.92, 2.352, 2.6112, 2.7667, 2.86   smooth
lr = 0.125  factor  0.00 : 3, 3, 3, 3, 3, 3                         one step
lr = 0.2    factor -0.60 : 4.8, 1.92, 3.648, 2.6112, 3.2333, 2.86   overshoots, converges
lr = 0.25   factor -1.00 : 6, 0, 6, 0, 6, 0                         orbits forever
lr = 0.3    factor -1.40 : 7.2, -2.88, 11.232, ...                  diverges
```

The `lr = 0.25` row is a trap worth memorising: `L(0) = L(6) = 36`, so the
printed loss reads 36.0000 on every single step while w swings by 6 each
iteration. **A flat loss curve does not mean nothing is happening.** All five
rows are asserted in `tests/test_train.py`.

**Momentum.**

```
v <- beta*v + g ;   w <- w - lr*v
```

Consistent directions accumulate, oscillating ones cancel. A constant gradient
g drives v towards `g/(1 - beta)`, so `beta = 0.9` multiplies the eventual step
by ten. Turning momentum on without lowering the learning rate is a standard
way to blow up a run that was fine.

**Adam.**

```
m <- b1*m + (1-b1)*g              running mean of the gradient
v <- b2*v + (1-b2)*g*g            running mean of the SQUARED gradient
m_hat = m / (1 - b1^t)            bias correction
v_hat = v / (1 - b2^t)
w <- w - lr * m_hat / (sqrt(v_hat) + eps)
```

The division is what "adaptive" means, and it is not a hand-wave. A parameter
whose gradient is consistently about 0.01 gets `m ~ 0.01` and
`sqrt(v) ~ 0.01`, so the ratio is about 1 and it moves a full `lr`. A parameter
whose gradient is consistently about 10 gets the same ratio and the same step.
The raw magnitudes cancel.

The bias correction exists because m and v both start at exactly zero: after
one step with `b1 = 0.9`, m is `0.1*g` rather than g, so the first steps read
about ten times too small. Dividing by `(1 - b1^t)` removes exactly that
startup shortfall.
