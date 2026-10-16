# [MODULE DOC]
# kernrl problem (level 1): matmul with irregular, non-power-of-two shapes
# (M=8205, K=2949, N=5921).
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model computing A @ B. No parameters.
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns A (M x K) @ B (K x N), shape (M, N).
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        return torch.matmul(A, B)


M = 8205
K = 2949
N = 5921


def get_inputs():
    # [FUNCTION DOC]
    # Returns random A of shape (M, K) and B of shape (K, N).
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []  # No special initialization inputs needed
