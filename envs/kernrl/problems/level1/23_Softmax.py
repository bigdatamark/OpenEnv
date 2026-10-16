# [MODULE DOC]
# kernrl problem (level 1): row-wise softmax over a (16, 16384) float tensor.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model applying softmax along dimension 1. No parameters.
    """
    Simple model that performs a Softmax activation.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns softmax(x) over dim 1, same shape as x.
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)


batch_size = 16
dim = 16384


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size=16, dim=16384).
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []  # No special initialization inputs needed
