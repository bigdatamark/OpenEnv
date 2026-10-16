# [MODULE DOC]
# kernrl problem (level 2): Linear (1024 -> 512), Swish and scaling by 2.0 on a
# (128, 1024) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: Linear, Swish, constant scale.
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """

    def __init__(self, in_features, out_features, scaling_factor):
        # [FUNCTION DOC]
        # Creates the Linear layer and stores the scaling factor.
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns swish(linear(x)) * scaling_factor, swish(v) = v * sigmoid(v).
        x = self.matmul(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = x * self.scaling_factor
        return x


batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_features).
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_features, out_features, scaling_factor].
    return [in_features, out_features, scaling_factor]
