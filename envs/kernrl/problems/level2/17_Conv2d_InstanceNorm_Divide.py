# [MODULE DOC]
# kernrl problem (level 2, fused operators): Conv2d (3 -> 16, 3x3) followed by
# InstanceNorm2d and division by 2.0 on a (128, 3, 32, 32) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: Conv2d, InstanceNorm2d (no affine, no running stats), then
    # division by a constant. Conv weights are randomly initialized.
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        # [FUNCTION DOC]
        # Creates the conv and instance-norm submodules and stores the divisor.
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns instance_norm(conv(x)) / divide_by.
        x = self.conv(x)
        x = self.instance_norm(x)
        x = x / self.divide_by
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divide_by = 2.0


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_channels, height, width).
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_channels, out_channels, kernel_size, divide_by].
    return [in_channels, out_channels, kernel_size, divide_by]
