# [MODULE DOC]
# kernrl problem (level 2): Conv2d (3 -> 16, 3x3), GroupNorm (8 groups),
# per-channel scale, 2x2 max pool and clamp to [0, 1] on a (128, 3, 32, 32)
# input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model with conv, GroupNorm, a learnable scale parameter (initialized
    # to ones, shape scale_shape), MaxPool2d and clamp bounds.
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ):
        # [FUNCTION DOC]
        # Creates the conv, GroupNorm, scale parameter and max-pool submodules and
        # stores the clamp bounds.
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns clamp(maxpool(group_norm(conv(x)) * scale), clamp_min, clamp_max).
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_channels, height, width).
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_channels, out_channels, kernel_size, num_groups, scale_shape,
    # maxpool_kernel_size, clamp_min, clamp_max].
    return [
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ]
