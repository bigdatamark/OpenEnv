# [MODULE DOC]
# kernrl problem (level 1): square matrix multiplication. Reference PyTorch
# module plus input factories used by the evaluator; agents must produce a
# faster Model with the same interface. N = 2048.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model computing C = A @ B for square matrices. No parameters.
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns the matrix product of A (N x N) and B (N x N).
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)


N = 2048


def get_inputs():
    # [FUNCTION DOC]
    # Returns two random N x N float32 CPU tensors [A, B] (moved to the device by
    # the evaluator).
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []  # No special initialization inputs needed
