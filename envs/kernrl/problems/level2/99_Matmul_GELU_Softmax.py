# [MODULE DOC]
# kernrl problem (level 2): Linear (100 -> 10), GELU and row softmax on a
# (128, 100) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: Linear, GELU, softmax over features.
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """

    def __init__(self, in_features, out_features):
        # [FUNCTION DOC]
        # Creates the Linear layer.
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns softmax(gelu(linear(x)), dim=1), shape (batch, out_features).
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


batch_size = 128
in_features = 100
out_features = 10


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_features).
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_features, out_features].
    return [in_features, out_features]
