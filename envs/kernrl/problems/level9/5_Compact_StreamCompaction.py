# [MODULE DOC]
# kernrl problem (level 9): stream compaction keeping elements >= 0.5 from 16M
# floats.
"""
Stream Compaction (Filter)

Removes elements that don't satisfy a predicate, compacting the result.
Also known as filtering or partition.

Example: Remove all zeros from array.

Optimization opportunities:
- Scan-based compaction
- Warp-level ballot for predicate evaluation
- Per-block compaction + global gather
- Decoupled lookback for single-pass
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference compaction model storing the threshold (default 0.5).
    """
    Stream compaction - removes elements not satisfying predicate.
    """

    def __init__(self, threshold: float = 0.5):
        # [FUNCTION DOC]
        # Stores the threshold.
        super(Model, self).__init__()
        self.threshold = threshold

    def forward(self, input: torch.Tensor) -> tuple:
        # [FUNCTION DOC]
        # Returns the tuple (elements >= threshold in original order, count of kept
        # elements as a scalar tensor).
        # KNOWN ISSUE: the evaluator's correctness check cannot convert a tuple of
        # tensors, so this output format fails comparison.
        """
        Compact array keeping only elements >= threshold.

        Args:
            input: (N,) input array

        Returns:
            output: (M,) compacted array (M <= N)
            count: number of elements kept
        """
        mask = input >= self.threshold
        output = input[mask]
        count = mask.sum()
        return output, count


# Problem configuration
array_size = 16 * 1024 * 1024


def get_inputs():
    # [FUNCTION DOC]
    # Returns [data] uniform random of length array_size.
    data = torch.rand(array_size)
    return [data]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [0.5] (threshold).
    return [0.5]  # threshold
