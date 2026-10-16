# [MODULE DOC]
# kernrl problem (level 1): LayerNorm over the last three dims (64, 256, 256) of
# a (16, 64, 256, 256) tensor, with learnable affine parameters.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model wrapping nn.LayerNorm with the given normalized shape
    # (elementwise weight and bias initialized to 1 and 0).
    """
    Simple model that performs Layer Normalization.
    """

    def __init__(self, normalized_shape: tuple):
        # [FUNCTION DOC]
        # Creates the LayerNorm submodule for normalized_shape.
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns the layer-normalized x, same shape.
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, features, dim1, dim2).
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [(features, dim1, dim2)] as the normalized shape argument.
    return [(features, dim1, dim2)]
