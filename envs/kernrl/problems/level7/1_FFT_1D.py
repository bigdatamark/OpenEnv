# [MODULE DOC]
# kernrl problem (level 7, signal processing): 1D FFT of a 1M-sample real
# signal.
"""
1D Fast Fourier Transform (FFT)

Computes the Discrete Fourier Transform using the Cooley-Tukey algorithm.
Fundamental operation in signal processing, audio analysis, and convolution.

Optimization opportunities:
- Radix-2/4/8 algorithms
- Shared memory for butterfly operations
- Bank-conflict-free shared memory access
- Warp-synchronous programming
- Stockham auto-sort algorithm
"""

import torch
import torch.fft
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference FFT model. No parameters.
    """
    1D Fast Fourier Transform.

    Computes DFT of complex or real signals.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns torch.fft.fft(signal) along the last dimension (complex output, same
        # length).
        # KNOWN ISSUE: the evaluator casts outputs with float() for comparison, which
        # drops the imaginary part of complex results.
        """
        Compute 1D FFT.

        Args:
            signal: (N,) or (B, N) real or complex signal

        Returns:
            spectrum: (N,) or (B, N) complex frequency components
        """
        return torch.fft.fft(signal)


# Problem configuration
signal_length = 1024 * 1024  # 1M samples


def get_inputs():
    # [FUNCTION DOC]
    # Returns [signal] random normal of length signal_length.
    # Real signal
    signal = torch.randn(signal_length)
    return [signal]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
