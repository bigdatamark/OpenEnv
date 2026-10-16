# [MODULE DOC]
# kernrl problem (level 2): Conv2d (3 -> 16, 3x3), BatchNorm2d and scaling by
# 2.0 on a (128, 3, 32, 32) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: conv, BatchNorm2d (running stats in eval mode), scale.
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        # [FUNCTION DOC]
        # Creates the conv and BatchNorm2d submodules and stores the scale.
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns bn(conv(x)) * scaling_factor.
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scaling_factor
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_channels, height, width).
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_channels, out_channels, kernel_size, scaling_factor].
    return [in_channels, out_channels, kernel_size, scaling_factor]
