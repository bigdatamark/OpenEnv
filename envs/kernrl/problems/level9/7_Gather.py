# [MODULE DOC]
# kernrl problem (level 9): gather of 16M values from a 1M-element source by
# random indices.
"""
Gather Operation

Gathers values from source array based on index array.
out[i] = source[indices[i]]

Optimization opportunities:
- Coalesced reads by sorting indices
- Texture memory for cached reads
- Prefetching for sequential access patterns
- Vectorized loads when possible
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference gather model. No parameters.
    """
    Gather values from indices.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns source[indices], shape of indices.
        """
        Gather values from source at indices.

        Args:
            source: (M,) source array
            indices: (N,) indices into source

        Returns:
            output: (N,) gathered values
        """
        return source[indices]


# Problem configuration
source_size = 1000000
num_gathers = 16 * 1024 * 1024


def get_inputs():
    # [FUNCTION DOC]
    # Returns a uniform random source (source_size,) and random int64 indices in
    # [0, source_size) of length num_gathers.
    source = torch.rand(source_size)
    indices = torch.randint(0, source_size, (num_gathers,))
    return [source, indices]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
