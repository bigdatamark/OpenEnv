# [MODULE DOC]
# kernrl problem (level 9): scatter of 4M values into a 1M-element output by
# random (colliding) indices.
"""
Scatter Operation

Scatters values to specified indices in output array.
out[indices[i]] = values[i]

Challenge: Multiple values may scatter to same index (race condition).

Optimization opportunities:
- Atomic operations for conflicts
- Sorting by destination for coalescing
- Segmented scatter
- Conflict detection with warp ballot
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference scatter model storing the output size.
    """
    Scatter values to indices.
    """

    def __init__(self, output_size: int = 1000000):
        # [FUNCTION DOC]
        # Stores output_size (default 1000000).
        super(Model, self).__init__()
        self.output_size = output_size

    def forward(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Creates a zero output of output_size in the values' dtype and writes
        # values[i] to output[indices[i]]; positions never written stay 0.
        # KNOWN ISSUE: with duplicate indices, which value wins is unspecified
        # (nondeterministic on GPU), so reference and solution may legitimately differ.
        """
        Scatter values to indices.

        Args:
            values: (N,) values to scatter
            indices: (N,) destination indices

        Returns:
            output: (output_size,) scattered values
        """
        output = torch.zeros(self.output_size, device=values.device, dtype=values.dtype)
        output.scatter_(0, indices, values)
        return output


# Problem configuration
num_values = 4 * 1024 * 1024
output_size = 1000000


def get_inputs():
    # [FUNCTION DOC]
    # Returns uniform random values (num_values,) and random int64 indices in
    # [0, output_size).
    values = torch.rand(num_values)
    indices = torch.randint(0, output_size, (num_values,))
    return [values, indices]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [output_size].
    return [output_size]
