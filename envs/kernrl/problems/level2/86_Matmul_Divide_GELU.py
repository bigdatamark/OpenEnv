# [MODULE DOC]
# kernrl problem (level 2): Linear (512 -> 1024), division by 10.0 and GELU on
# a (128, 512) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: Linear, scalar division, GELU.
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """

    def __init__(self, input_size, output_size, divisor):
        # [FUNCTION DOC]
        # Creates the Linear layer and stores the divisor.
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns gelu(linear(x) / divisor), shape (batch, output_size).
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = self.linear(x)
        x = x / self.divisor
        x = torch.nn.functional.gelu(x)
        return x


batch_size = 128
input_size = 512
output_size = 1024
divisor = 10.0


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, input_size).
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [input_size, output_size, divisor].
    return [input_size, output_size, divisor]
