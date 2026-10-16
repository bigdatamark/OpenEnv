# [MODULE DOC]
# kernrl problem (level 2): Conv2d (3 -> 16, 3x3), Mish-style activation
# (tanh(softplus(x)) * x) and BatchNorm2d on a (128, 3, 32, 32) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: conv, activation, BatchNorm2d (eps and momentum
    # configurable). In eval mode (as used by the evaluator) BatchNorm uses its
    # running statistics (initially mean 0, var 1).
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """

    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        # [FUNCTION DOC]
        # Creates the conv and BatchNorm2d submodules.
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns bn(tanh(softplus(conv(x))) * conv(x)).
        x = self.conv(x)
        x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
        x = self.bn(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_channels, height, width).
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_channels, out_channels, kernel_size].
    return [in_channels, out_channels, kernel_size]
