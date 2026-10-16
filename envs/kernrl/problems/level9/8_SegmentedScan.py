# [MODULE DOC]
# kernrl problem (level 9): segmented exclusive prefix sum over 16M floats with
# 1000 segments; the reference loops over segments in Python.
"""
Segmented Prefix Sum

Computes prefix sum within segments defined by a flag array.
Resets accumulator at segment boundaries.

Used in graph algorithms, sparse operations, and more.

Optimization opportunities:
- Head flags for segment boundaries
- Warp-level segmented scan
- Decoupled lookback with segment handling
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference segmented scan model. No parameters.
    """
    Segmented exclusive prefix sum.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(
        self, values: torch.Tensor, segment_heads: torch.Tensor
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Collects segment start indices from the head flags (adding 0 if missing),
        # and for each segment writes cumsum(segment) - segment, so each segment's
        # exclusive scan restarts at 0. Returns an (N,) tensor.
        """
        Compute segmented exclusive prefix sum.

        Args:
            values: (N,) input values
            segment_heads: (N,) boolean tensor, True at segment starts

        Returns:
            output: (N,) segmented exclusive prefix sums
        """
        N = values.shape[0]
        output = torch.zeros_like(values)

        # Find segment starts
        segment_starts = torch.where(segment_heads)[0].tolist()
        if 0 not in segment_starts:
            segment_starts = [0] + segment_starts
        segment_starts.append(N)

        # Process each segment
        for i in range(len(segment_starts) - 1):
            start = segment_starts[i]
            end = segment_starts[i + 1]
            segment = values[start:end]
            output[start:end] = torch.cumsum(segment, dim=0) - segment

        return output


# Problem configuration
array_size = 16 * 1024 * 1024
num_segments = 1000


def get_inputs():
    # [FUNCTION DOC]
    # Returns uniform random values and a boolean head-flag array with index 0 and
    # num_segments - 1 other distinct random positions set.
    values = torch.rand(array_size)
    # Random segment heads
    segment_heads = torch.zeros(array_size, dtype=torch.bool)
    segment_heads[0] = True  # First element always starts a segment
    head_positions = torch.randperm(array_size - 1)[: num_segments - 1] + 1
    segment_heads[head_positions] = True
    return [values, segment_heads]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
