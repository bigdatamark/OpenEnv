# [MODULE DOC]
# kernrl problem (level 2): Linear (64 -> 128) followed by scaling and a
# residual add of the unscaled result, i.e. output = (1 + 0.5) * linear(x).
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model with a Linear layer and a scaling factor.
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """

    def __init__(self, in_features, out_features, scaling_factor):
        # [FUNCTION DOC]
        # Creates the Linear layer and stores the scaling factor.
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # [FUNCTION DOC]
        # Computes y = linear(x) and returns y * scaling_factor + y (the residual copy
        # is detached, which only matters for gradients).
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        original_x = x.clone().detach()
        x = x * self.scaling_factor
        x = x + original_x
        return x


batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_features).
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_features, out_features, scaling_factor].
    return [in_features, out_features, scaling_factor]
