"""netfs -- net from scratch.

Neural networks and convolutional networks implemented in NumPy, with every
gradient derived by hand in docs/DERIVATIONS.md and checked against a
central-difference numerical gradient in tests/.

The hard rule of this package: **nothing in src/netfs imports a deep learning
framework.** Forward passes, backward passes, losses, convolution, pooling and
the optimisers are NumPy. PyTorch appears in tests/test_torch_oracle.py and
nowhere else, as an independent second opinion on gradients we already
computed. scipy appears in tests/test_conv.py, as an independent second opinion
on convolution. Neither of them ever does the learning. See docs/DECISIONS.md
ADR-001.
"""

from .conv import Conv2D, col2im, conv2d_im2col, conv2d_naive, im2col, im2col_indices
from .data import ImageDataset, iterate_minibatches, load_image_dataset, two_moons, xor_dataset
from .gradcheck import check_layer, check_model, numerical_gradient, relative_error
from .layers import Flatten, Linear, ReLU, Sigmoid, Tanh
from .losses import (accuracy, confusion_matrix, cross_entropy_unfused, log_sum_exp, mse_loss,
                     softmax, softmax_cross_entropy, softmax_naive)
from .model import Sequential
from .optim import SGD, Adam
from .pool import GlobalAvgPool2D, MaxPool2D
from .shapes import conv_macs, conv_out_size, conv_params, receptive_field, same_padding
from .train import evaluate, train

__version__ = "0.1.0"

__all__ = [
    "Conv2D", "col2im", "conv2d_im2col", "conv2d_naive", "im2col", "im2col_indices",
    "ImageDataset", "iterate_minibatches", "load_image_dataset", "two_moons", "xor_dataset",
    "check_layer", "check_model", "numerical_gradient", "relative_error",
    "Flatten", "Linear", "ReLU", "Sigmoid", "Tanh",
    "accuracy", "confusion_matrix", "cross_entropy_unfused", "log_sum_exp", "mse_loss",
    "softmax", "softmax_cross_entropy", "softmax_naive",
    "Sequential", "SGD", "Adam", "GlobalAvgPool2D", "MaxPool2D",
    "conv_macs", "conv_out_size", "conv_params", "receptive_field", "same_padding",
    "evaluate", "train", "__version__",
]
