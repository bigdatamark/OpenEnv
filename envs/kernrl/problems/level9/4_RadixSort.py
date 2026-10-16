# [MODULE DOC]
# kernrl problem (level 9): ascending sort of 4M non-negative 31-bit integers
# (stored as int64); the reference uses torch.sort.
"""
Radix Sort (32-bit integers)

Sorts array of 32-bit integers using radix sort.
Processes bits in groups, using counting sort for each digit.

Optimization opportunities:
- Per-block radix sort + global merge
- 4-bit or 8-bit radix for fewer passes
- Local sort using shared memory
- Warp-level sort for small segments
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference sort model. No parameters.
    """
    Radix sort for 32-bit integers.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns the input values sorted ascending.
        """
        Sort array using radix sort.

        Args:
            input: (N,) array of 32-bit integers

        Returns:
            sorted: (N,) sorted array
        """
        return torch.sort(input)[0]


# Problem configuration
array_size = 4 * 1024 * 1024  # 4M elements


def get_inputs():
    # [FUNCTION DOC]
    # Returns [data] random int64 values in [0, 2^31).
    # Random 32-bit integers
    data = torch.randint(0, 2**31, (array_size,), dtype=torch.int64)
    return [data]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
