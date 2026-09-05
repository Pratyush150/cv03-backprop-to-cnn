"""Sequential: the smallest thing that can be called a network.

A network is a list of layers, a forward pass that walks it left to right, and
a backward pass that walks it right to left. That really is all a framework's
`nn.Sequential` is, once autograd is not doing the bookkeeping for you.

The reverse walk is the part worth staring at. Backprop is one rule --

    gradient out of a node = upstream gradient x that node's local Jacobian

-- applied in reverse topological order, where "reverse topological order"
means you never visit a node until everything that consumes its output has
already been visited. For a straight chain that is just `reversed(layers)`, and
the whole of `backward()` below is four lines. A branching graph (a ResNet skip
connection, say) needs the general version, and the only new rule it needs is
that a value consumed twice receives the SUM of the two gradients coming back.
"""

from __future__ import annotations

import numpy as np


class Sequential:
    def __init__(self, *layers) -> None:
        # Accept either Sequential(a, b, c) or Sequential([a, b, c]).
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = tuple(layers[0])
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    __call__ = forward

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Walk the chain backwards, handing each layer the gradient of the
        loss with respect to ITS output, and receiving the gradient with
        respect to its input -- which is, by definition, the gradient with
        respect to the previous layer's output. That handoff is the entire
        algorithm; each layer knows nothing about the rest of the network, and
        that locality is why backprop scales to a hundred million parameters.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def backward_from_loss(self, loss_fn, x: np.ndarray, y) -> float:
        """forward -> loss -> backward, in one call. Returns the loss.

        The forward pass is re-run here rather than assumed, because the
        backward pass depends on caches that must correspond to the CURRENT
        parameters. Reusing a stale cache after an optimiser step is a bug that
        shows up as a gradient check that passes on step 0 and fails on step 1.
        """
        out = self.forward(x)
        loss, dout = loss_fn(out, y)
        self.backward(dout)
        return loss

    def parameters(self):
        """Yield (layer, name) for every trainable array. Optimisers walk this
        rather than a flat list of arrays, because they must be able to write
        the update back into the layer that owns it.
        """
        for layer in self.layers:
            for name in layer.params:
                yield layer, name

    def n_params(self) -> int:
        return sum(layer.params[name].size for layer, name in self.parameters())

    def describe(self, input_shape) -> str:
        """Run a dummy batch through and report the shape at every stage.

        Not decoration. Spatial sizes drift off by one somewhere deep in a
        network and the exception surfaces three layers later; the fastest way
        to find where it started is to print the shape after every layer and
        compare against the arithmetic you did on paper.
        """
        x = np.zeros((1, *input_shape))
        lines = [f"{'layer':<22}{'output shape':<22}{'params':>10}"]
        lines.append(f"{'input':<22}{str(x.shape):<22}{'':>10}")
        for layer in self.layers:
            x = layer.forward(x)
            n = sum(p.size for p in layer.params.values())
            lines.append(f"{type(layer).__name__:<22}{str(x.shape):<22}{n:>10,}")
        lines.append(f"{'TOTAL':<22}{'':<22}{self.n_params():>10,}")
        return "\n".join(lines)
