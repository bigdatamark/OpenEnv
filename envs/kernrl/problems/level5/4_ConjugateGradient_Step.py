# [MODULE DOC]
# kernrl problem (level 5): one Conjugate Gradient iteration on a dense
# 4096 x 4096 symmetric positive definite system.
"""
Conjugate Gradient Solver Step

One iteration of the Conjugate Gradient method for solving Ax = b.
This combines multiple BLAS operations that can be fused:
- Matrix-vector product (SpMV or dense)
- Multiple dot products
- Vector updates (AXPY)

Optimization opportunities:
- Kernel fusion to reduce memory traffic
- Persistent threads to keep intermediate results in registers
- Overlapping computation with memory operations
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference CG step model. No parameters.
    """
    One iteration of the Conjugate Gradient method.

    Given current state (x, r, p, rsold), computes next iteration.
    This is a key building block for large-scale linear system solvers.

    CG iteration:
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @ r
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(
        self,
        A: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        p: torch.Tensor,
        rsold: torch.Tensor,
    ) -> tuple:
        # [FUNCTION DOC]
        # Computes Ap = A @ p, alpha = rsold / (p . Ap), x_new = x + alpha p,
        # r_new = r - alpha Ap, rsnew = r_new . r_new, p_new = r_new + (rsnew/rsold) p.
        # Returns the tuple (x_new, r_new, p_new, rsnew).
        # KNOWN ISSUE: the evaluator compares outputs by converting non-tensor outputs
        # with torch.tensor, which fails for a tuple of tensors, so this problem's
        # correctness check cannot pass as written.
        """
        Perform one CG iteration.

        Args:
            A: (N, N) symmetric positive definite matrix
            x: (N,) current solution estimate
            r: (N,) current residual (b - Ax)
            p: (N,) current search direction
            rsold: scalar, r @ r from previous iteration

        Returns:
            x_new, r_new, p_new, rsnew: updated CG state
        """
        # Matrix-vector product
        Ap = A @ p

        # Compute step size
        pAp = torch.dot(p, Ap)
        alpha = rsold / pAp

        # Update solution
        x_new = x + alpha * p

        # Update residual
        r_new = r - alpha * Ap

        # Compute new residual norm squared
        rsnew = torch.dot(r_new, r_new)

        # Update search direction
        beta = rsnew / rsold
        p_new = r_new + beta * p

        return x_new, r_new, p_new, rsnew


# Problem configuration
matrix_size = 4096


def get_inputs():
    # [FUNCTION DOC]
    # Builds an SPD matrix A = Q D Q^T (Q orthogonal via QR, eigenvalues in
    # [0.1, 1.1)), random x and b, residual r = b - A x, p = r and rsold = r . r.
    # Returns [A, x, r, p, rsold].
    # Create a symmetric positive definite matrix
    # A = Q @ D @ Q.T where D has positive eigenvalues
    Q = torch.randn(matrix_size, matrix_size)
    Q, _ = torch.linalg.qr(Q)  # Orthogonal matrix
    D = torch.diag(torch.rand(matrix_size) + 0.1)  # Positive eigenvalues
    A = Q @ D @ Q.T

    # Random initial state
    x = torch.randn(matrix_size)
    b = torch.randn(matrix_size)
    r = b - A @ x  # Initial residual
    p = r.clone()  # Initial search direction
    rsold = torch.dot(r, r)

    return [A, x, r, p, rsold]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
