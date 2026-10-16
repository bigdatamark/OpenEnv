# [MODULE DOC]
# kernrl problem (level 1): tall-skinny matmul with M=16384, N=16; the product
# is a large (16384, 16384) matrix built from a rank-16 inner dimension.
import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model computing A @ B. No parameters.
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, A, B):
        # [FUNCTION DOC]
        # Returns A @ B. With the provided inputs A is (M, N) and B is (N, M), so the
        # result is (M, M).
        # KNOWN ISSUE: the docstring describes shapes in terms of K and an (M, N) or
        # (N, M) output, which does not match the actual (M, M) result.
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return torch.matmul(A, B)


M = 16384
N = 16


def get_inputs():
    # [FUNCTION DOC]
    # Returns random A of shape (M, N) and B of shape (N, M).
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []  # No special initialization inputs needed
