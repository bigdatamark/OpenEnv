# [MODULE DOC]
# kernrl problem (level 1): batched matmul (batch 128, m=128, k=256, n=512).
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model computing a batched matrix product. No parameters.
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns bmm(A, B): (b, m, k) x (b, k, n) -> (b, m, n).
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return torch.bmm(A, B)


batch_size = 128
m = 128
k = 256
n = 512


def get_inputs():
    # [FUNCTION DOC]
    # Returns random A (batch, m, k) and B (batch, k, n).
    A = torch.randn(batch_size, m, k)
    B = torch.randn(batch_size, k, n)
    return [A, B]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []  # No special initialization inputs needed
