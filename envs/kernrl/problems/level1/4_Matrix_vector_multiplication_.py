# [MODULE DOC]
# kernrl problem (level 1): matrix-vector product with M=256, K=131072.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model computing A @ B where B is a column vector. No parameters.
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns A (M x K) @ B (K x 1), shape (M, 1).
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return torch.matmul(A, B)


M = 256
K = 131072


def get_inputs():
    # [FUNCTION DOC]
    # Returns random A of shape (M, K) and B of shape (K, 1).
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []  # No special initialization inputs needed
