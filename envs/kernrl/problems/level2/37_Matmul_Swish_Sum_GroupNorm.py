# [MODULE DOC]
# kernrl problem (level 2): Linear (512 -> 1024), Swish, bias add and
# GroupNorm (32 groups) on a (128, 512) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model with a Linear layer, an extra randomly initialized bias
    # parameter of shape bias_shape, and GroupNorm.
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """

    def __init__(self, in_features, out_features, num_groups, bias_shape):
        # [FUNCTION DOC]
        # Creates the Linear layer, the extra bias parameter (random normal) and the
        # GroupNorm with num_groups over out_features channels.
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns group_norm(swish(linear(x)) + bias), where swish(v) = sigmoid(v) * v.
        # Shape (batch, out_features).
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = x + self.bias
        x = self.group_norm(x)
        return x


batch_size = 128
in_features = 512
out_features = 1024
num_groups = 32
bias_shape = (out_features,)


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_features).
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_features, out_features, num_groups, bias_shape].
    return [in_features, out_features, num_groups, bias_shape]
