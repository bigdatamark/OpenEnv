# [MODULE DOC]
# kernrl problem (level 2): Linear (10 -> 5), 1D max pool (kernel 2) over the
# feature axis, sum over features and scale by 0.5 on a (128, 10) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: Linear, MaxPool1d over features, sum, scale.
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """

    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        # [FUNCTION DOC]
        # Creates the Linear and MaxPool1d submodules and stores the scale factor.
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        # [FUNCTION DOC]
        # Computes linear(x), max-pools along the feature axis (treated as a single
        # channel), sums the pooled features per row and multiplies by scale_factor.
        # Returns a (batch,) tensor.
        # KNOWN ISSUE: the docstring says the output is (batch_size, out_features), but
        # the sum reduces it to (batch_size,).
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.max_pool(x.unsqueeze(1)).squeeze(1)
        x = torch.sum(x, dim=1)
        x = x * self.scale_factor
        return x


batch_size = 128
in_features = 10
out_features = 5
kernel_size = 2
scale_factor = 0.5


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_features).
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_features, out_features, kernel_size, scale_factor].
    return [in_features, out_features, kernel_size, scale_factor]
