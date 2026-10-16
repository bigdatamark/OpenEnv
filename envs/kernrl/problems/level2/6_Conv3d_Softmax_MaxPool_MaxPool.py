# [MODULE DOC]
# kernrl problem (level 2): Conv3d (3 -> 16, 3x3x3), channel softmax and two
# successive 2x2x2 max pools on a (128, 3, 16, 32, 32) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: Conv3d, softmax over channels, two MaxPool3d layers.
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """

    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        # [FUNCTION DOC]
        # Creates the Conv3d and two MaxPool3d submodules.
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns pool2(pool1(softmax(conv(x), dim=1))).
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch, in_channels, depth, height, width).
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_channels, out_channels, kernel_size, pool_kernel_size].
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
