# [MODULE DOC]
# kernrl problem (level 7): direct 1D filtering of a 1M-sample signal with a
# 127-tap random kernel, "same" zero padding.
"""
1D Convolution (Direct)

Direct implementation of 1D convolution without using FFT.
Common in signal processing, audio effects, and neural networks.

Optimization opportunities:
- Shared memory tiling
- Loop unrolling for fixed kernel sizes
- Vectorized loads
- Register blocking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # [CLASS DOC]
    # Reference 1D filter model with a random (1, 1, kernel_size) kernel buffer.
    """
    1D convolution with a filter kernel.
    """

    def __init__(self, kernel_size: int = 127):
        # [FUNCTION DOC]
        # Stores kernel_size and registers a random normal kernel.
        super(Model, self).__init__()
        self.kernel_size = kernel_size

        # Random kernel (for FIR filter, this would be designed coefficients)
        kernel = torch.randn(1, 1, kernel_size)
        self.register_buffer("kernel", kernel)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Reshapes (N,) or (B, N) input to (B, 1, N), applies conv1d (cross-correlation,
        # i.e. unflipped kernel) with padding kernel_size // 2, and restores the input
        # rank. Returns the filtered signal (same length for odd kernel sizes).
        """
        Apply 1D convolution.

        Args:
            signal: (N,) or (B, N) 1D signal

        Returns:
            result: (N,) or (B, N) convolved signal (same size with padding)
        """
        original_shape = signal.shape

        if signal.dim() == 1:
            signal = signal.unsqueeze(0).unsqueeze(0)
        elif signal.dim() == 2:
            signal = signal.unsqueeze(1)

        # Same padding
        padding = self.kernel_size // 2
        result = F.conv1d(signal, self.kernel, padding=padding)

        # Restore shape
        if len(original_shape) == 1:
            result = result.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            result = result.squeeze(1)

        return result


# Problem configuration
signal_length = 1024 * 1024


def get_inputs():
    # [FUNCTION DOC]
    # Returns [signal] random normal of length signal_length.
    signal = torch.randn(signal_length)
    return [signal]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [127] (kernel_size).
    return [127]  # kernel_size
