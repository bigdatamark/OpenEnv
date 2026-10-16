# [MODULE DOC]
# kernrl problem (level 2): Linear (100 -> 50), dropout (p=0.2), mean over
# features and softmax on a (128, 100) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: Linear, Dropout (identity in eval mode), mean, softmax.
    """
    A model that performs matrix multiplication, applies dropout, calculates the mean, and then applies softmax.
    """

    def __init__(self, in_features, out_features, dropout_p):
        # [FUNCTION DOC]
        # Creates the Linear and Dropout submodules.
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # [FUNCTION DOC]
        # Computes linear(x), dropout, the mean over dim 1 with keepdim, then softmax
        # over dim 1. Returns a (batch, 1) tensor.
        # KNOWN ISSUE: softmax over a size-1 dimension always yields 1.0, and the
        # docstring's (batch_size, out_features) output shape is wrong.
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.softmax(x, dim=1)
        return x


batch_size = 128
in_features = 100
out_features = 50
dropout_p = 0.2


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_features).
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_features, out_features, dropout_p].
    return [in_features, out_features, dropout_p]
