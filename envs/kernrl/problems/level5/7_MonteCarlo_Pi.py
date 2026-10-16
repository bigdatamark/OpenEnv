# [MODULE DOC]
# kernrl problem (level 5): Monte Carlo estimate of pi from 10 million
# pre-generated random points (classification plus reduction).
"""
Monte Carlo Pi Estimation

Estimates Pi using Monte Carlo integration: count random points falling inside unit circle.
This represents a broader class of Monte Carlo integration problems.

The kernel involves:
- Random number generation (or processing pre-generated randoms)
- Point classification (inside/outside circle)
- Reduction to count hits

Optimization opportunities:
- Parallel random number generation (cuRAND)
- Warp-level reductions
- Persistent kernel for streaming random numbers
- Fused generation + classification + reduction
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model. No parameters.
    """
    Monte Carlo estimation of Pi using random sampling.

    Points (x, y) in [0, 1]^2 that satisfy x^2 + y^2 <= 1 fall inside
    the quarter circle. Ratio of hits to total * 4 estimates Pi.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, random_points: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Counts points with x^2 + y^2 <= 1 and returns 4 * count / N as a scalar
        # tensor.
        """
        Compute Pi estimate from random points.

        Args:
            random_points: (N, 2) random points in [0, 1]^2

        Returns:
            pi_estimate: scalar tensor with Pi estimate
        """
        # Compute distance from origin
        x = random_points[:, 0]
        y = random_points[:, 1]

        # Points inside quarter circle: x^2 + y^2 <= 1
        dist_sq = x * x + y * y
        inside = (dist_sq <= 1.0).float()

        # Ratio * 4 = Pi estimate
        N = random_points.shape[0]
        pi_estimate = 4.0 * inside.sum() / N

        return pi_estimate


# Problem configuration - many samples for accuracy
num_samples = 10_000_000


def get_inputs():
    # [FUNCTION DOC]
    # Returns [points] uniform in [0, 1)^2, shape (num_samples, 2).
    # Pre-generate random points
    random_points = torch.rand(num_samples, 2)
    return [random_points]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
