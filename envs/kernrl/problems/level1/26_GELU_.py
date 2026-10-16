# [MODULE DOC]
# kernrl problem (level 1): elementwise GELU (exact erf form, PyTorch default)
# over a (16, 16384) tensor.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model applying GELU. No parameters.
    """
    Simple model that performs a GELU activation.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns GELU(x) elementwise, same shape.
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return torch.nn.functional.gelu(x)


batch_size = 16
dim = 16384


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (16, 16384).
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []  # No special initialization inputs needed
