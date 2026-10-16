# [MODULE DOC]
# kernrl problem (level 9, parallel primitives): exclusive prefix sum of 16M
# floats.
"""
Exclusive Prefix Sum (Scan)

Computes exclusive prefix sum: out[i] = sum(in[0:i])
Fundamental building block for parallel algorithms.

out[0] = 0
out[i] = in[0] + in[1] + ... + in[i-1]

Optimization opportunities:
- Kogge-Stone or Blelloch algorithm
- Warp-level scan using shuffle
- Multi-level scan for large arrays
- Bank conflict-free shared memory
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference exclusive scan model. No parameters.
    """
    Exclusive prefix sum (scan).
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns cumsum(input) - input, i.e. out[i] = sum of input[0..i-1] (subject to
        # float rounding of the subtraction).
        """
        Compute exclusive prefix sum.

        Args:
            input: (N,) input array

        Returns:
            output: (N,) exclusive prefix sum
        """
        return torch.cumsum(input, dim=0) - input


# Problem configuration
array_size = 16 * 1024 * 1024  # 16M elements


def get_inputs():
    # [FUNCTION DOC]
    # Returns [data] uniform random of length array_size.
    data = torch.rand(array_size)
    return [data]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
