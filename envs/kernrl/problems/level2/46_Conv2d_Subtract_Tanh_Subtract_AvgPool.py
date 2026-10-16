# [MODULE DOC]
# kernrl problem (level 2): Conv2d (3 -> 16, 3x3), subtract 0.5, tanh,
# subtract 0.2, 2x2 average pool on a (128, 3, 32, 32) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: conv, two constant subtractions around tanh, average pool.
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        subtract1_value,
        subtract2_value,
        kernel_size_pool,
    ):
        # [FUNCTION DOC]
        # Creates the conv and AvgPool2d submodules and stores the two constants.
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns avgpool(tanh(conv(x) - subtract1) - subtract2).
        x = self.conv(x)
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_channels, height, width).
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_channels, out_channels, kernel_size, subtract1_value,
    # subtract2_value, kernel_size_pool].
    return [
        in_channels,
        out_channels,
        kernel_size,
        subtract1_value,
        subtract2_value,
        kernel_size_pool,
    ]
