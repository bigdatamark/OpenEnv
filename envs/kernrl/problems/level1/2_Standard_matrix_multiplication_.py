# [MODULE DOC]
# kernrl problem (level 1): rectangular matmul (M=1024, K=4096, N=2048).
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model computing A @ B. No parameters.
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns A (M x K) @ B (K x N), shape (M, N).
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B)


M = 1024
K = 4096
N = 2048


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
