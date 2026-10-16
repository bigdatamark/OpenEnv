# [MODULE DOC]
# kernrl problem (level 2): Linear (512 -> 256), 1D average pool (kernel 4)
# over features, GELU, scale by 2.0 and max over features on a (128, 512) input.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model: Linear, AvgPool1d over features, GELU, scale, row max.
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """

    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        # [FUNCTION DOC]
        # Creates the Linear and AvgPool1d submodules and stores the scale factor.
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        # [FUNCTION DOC]
        # Computes linear(x), average-pools along features (as a single channel),
        # applies GELU, multiplies by scale_factor and returns the per-row maximum, a
        # (batch,) tensor.
        # KNOWN ISSUE: the docstring says the output is (batch_size, out_features),
        # but the final max reduces it to (batch_size,).
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        x = torch.nn.functional.gelu(x)
        x = x * self.scale_factor
        x = torch.max(x, dim=1).values
        return x


batch_size = 128
in_features = 512
out_features = 256
pool_kernel_size = 4
scale_factor = 2.0


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, in_features).
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [in_features, out_features, pool_kernel_size, scale_factor].
    return [in_features, out_features, pool_kernel_size, scale_factor]
