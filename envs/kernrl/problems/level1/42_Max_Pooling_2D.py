# [MODULE DOC]
# kernrl problem (level 1): 2D max pooling (kernel 2, stride 2, padding 1,
# dilation 3) over a (16, 32, 128, 128) tensor.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model wrapping nn.MaxPool2d with the given window parameters.
    """
    Simple model that performs Max Pooling 2D.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        # [FUNCTION DOC]
        # Creates the MaxPool2d submodule from kernel_size, stride, padding, dilation.
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns max-pooled x of shape (batch, channels, pooled_h, pooled_w).
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool(x)


batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, channels, height, width).
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [kernel_size, stride, padding, dilation].
    return [kernel_size, stride, padding, dilation]
