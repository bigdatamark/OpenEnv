# [MODULE DOC]
# kernrl problem (level 1): sum reduction over dimension 1 of a (16, 256, 256)
# tensor, keeping the reduced dimension.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model summing over a configured dimension with keepdim.
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        # [FUNCTION DOC]
        # Stores the dimension to reduce.
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns sum of x over self.dim with keepdim True (size 1 in that dim).
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return torch.sum(x, dim=self.dim, keepdim=True)


batch_size = 16
dim1 = 256
dim2 = 256
reduce_dim = 1


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, dim1, dim2).
    x = torch.randn(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [reduce_dim] (1).
    return [reduce_dim]
