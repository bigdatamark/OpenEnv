# [MODULE DOC]
# kernrl problem (level 1): RMS normalization over the channel dimension of a
# (16, 64, 256, 256) tensor, without a learnable scale.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference RMSNorm model with num_features and eps (default 1e-5); no
    # learnable parameters (num_features is stored but unused).
    """
    Simple model that performs RMS Normalization.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        # [FUNCTION DOC]
        # Stores num_features and eps.
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns x divided by sqrt(mean(x^2 over dim 1, kept) + eps); same shape.
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the feature dimension
        rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)

        # Normalize the input by dividing by the RMS
        return x / rms


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
    # Returns [features] (64) as constructor arguments.
    return [features]
