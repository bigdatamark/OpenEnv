# [MODULE DOC]
# kernrl problem (level 9): max reduction of 64M floats.
"""
Parallel Reduction - Maximum

Finds the maximum element in an array.
Similar structure to sum reduction but with max operation.

Optimization opportunities:
- Same as sum reduction
- Can use warp vote for early termination
- Max with index tracking (argmax)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference max reduction model. No parameters.
    """
    Parallel max reduction.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns the maximum element as a scalar tensor.
        """
        Find maximum element.

        Args:
            input: (N,) input array

        Returns:
            max_val: scalar tensor
        """
        return input.max()


# Problem configuration
array_size = 64 * 1024 * 1024


def get_inputs():
    # [FUNCTION DOC]
    # Returns [data] uniform random of length array_size.
    data = torch.rand(array_size)
    return [data]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
