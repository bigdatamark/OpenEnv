# [MODULE DOC]
# kernrl problem (level 5): one Jacobi iteration of the 5-point 2D stencil on
# a 2048 x 2048 grid with fixed boundaries.
"""
2D Stencil Computation - Heat Equation / Jacobi Iteration

Classic 5-point stencil for solving the 2D heat equation or Laplace equation.
u_new[i,j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

This is a memory-bound kernel with regular access patterns.

Optimization opportunities:
- Shared memory caching to reduce redundant loads
- Loop tiling / temporal blocking
- Vectorized loads for coalescing
- Register blocking across multiple rows
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference stencil model. No parameters.
    """
    Applies one iteration of 2D Jacobi stencil (5-point Laplacian).

    This is the core operation in iterative solvers for PDEs like:
    - Heat equation (diffusion)
    - Laplace equation (steady-state)
    - Poisson equation (with source term)
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns a copy of u where every interior cell is 0.25 times the sum of its
        # four neighbors (computed from the old values) and boundary cells are
        # unchanged.
        """
        Apply one Jacobi iteration.

        Args:
            u: (H, W) 2D grid values

        Returns:
            u_new: (H, W) updated grid (interior updated, boundary unchanged)
        """
        H, W = u.shape

        # Allocate output (copy boundary)
        u_new = u.clone()

        # Apply 5-point stencil to interior
        # u_new[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
        u_new[1:-1, 1:-1] = (
            0.25
            * (
                u[:-2, 1:-1]  # North
                + u[2:, 1:-1]  # South
                + u[1:-1, :-2]  # West
                + u[1:-1, 2:]  # East
            )
        )

        return u_new


# Problem configuration - large grid
grid_height = 2048
grid_width = 2048


def get_inputs():
    # [FUNCTION DOC]
    # Returns [u] with random interior values and Dirichlet boundaries: top and
    # left edges 1.0, bottom and right edges 0.0 (set in that order).
    # Initialize with random values (or could be zeros with boundary conditions)
    u = torch.randn(grid_height, grid_width)
    # Set boundary conditions (Dirichlet - fixed boundary)
    u[0, :] = 1.0  # Top
    u[-1, :] = 0.0  # Bottom
    u[:, 0] = 1.0  # Left
    u[:, -1] = 0.0  # Right
    return [u]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
