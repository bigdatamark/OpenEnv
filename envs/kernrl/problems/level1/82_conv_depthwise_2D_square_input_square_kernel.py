# [MODULE DOC]
# kernrl problem (level 1): depthwise 2D convolution (3 channels, 3x3 kernel,
# stride 1, no padding, no bias) on a (16, 3, 256, 256) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model wrapping nn.Conv2d with groups = in_channels (one filter per
    # channel).
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        # [FUNCTION DOC]
        # Creates the depthwise Conv2d submodule (in_channels in and out, groups =
        # in_channels) with the given kernel, stride, padding and bias.
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns the depthwise convolution of x, shape (B, in_channels, H_out, W_out).
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return self.conv2d(x)


# Test code
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_channels, height, width).
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_channels, kernel_size, stride, padding].
    return [in_channels, kernel_size, stride, padding]
