# [MODULE DOC]
# kernrl problem (level 2): Conv2d (3 -> 16, 3x3), tanh, scale by 2.0, add a
# per-channel bias and 2x2 max pool on a (128, 3, 32, 32) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model with conv, a random (out_channels, 1, 1) bias parameter and
    # MaxPool2d.
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        scaling_factor,
        bias_shape,
        pool_kernel_size,
    ):
        # [FUNCTION DOC]
        # Creates the conv, the bias parameter (random normal of bias_shape) and the
        # max-pool submodule; stores the scale.
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns max_pool(tanh(conv(x)) * scaling_factor + bias).
        # Convolution
        x = self.conv(x)
        # Tanh activation
        x = torch.tanh(x)
        # Scaling
        x = x * self.scaling_factor
        # Bias addition
        x = x + self.bias
        # Max-pooling
        x = self.max_pool(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 2


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_channels, height, width).
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_channels, out_channels, kernel_size, scaling_factor, bias_shape,
    # pool_kernel_size].
    return [
        in_channels,
        out_channels,
        kernel_size,
        scaling_factor,
        bias_shape,
        pool_kernel_size,
    ]
