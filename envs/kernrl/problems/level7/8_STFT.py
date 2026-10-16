# [MODULE DOC]
# kernrl problem (level 7): short-time Fourier transform of a 10 s 16 kHz
# signal (n_fft 1024, hop 256, Hann window, centered reflect padding).
"""
Short-Time Fourier Transform (STFT)

Computes the STFT of a signal using sliding window analysis.
Fundamental for audio processing, speech recognition, and spectrograms.

STFT(t, f) = sum_n x[n] * w[n-t] * exp(-j*2*pi*f*n/N)

Optimization opportunities:
- Batched FFTs for all windows
- Shared memory for window overlap
- Fused windowing + FFT
- Streaming for long signals
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference STFT model with a window buffer (Hann, Hamming, or rectangular
    # for any other name).
    """
    Short-Time Fourier Transform.
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, window: str = "hann"):
        # [FUNCTION DOC]
        # Stores n_fft and hop_length and registers the chosen window.
        super(Model, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Create window function
        if window == "hann":
            w = torch.hann_window(n_fft)
        elif window == "hamming":
            w = torch.hamming_window(n_fft)
        else:
            w = torch.ones(n_fft)

        self.register_buffer("window", w)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns torch.stft of the signal with return_complex True, center True and
        # reflect padding: a complex tensor of shape (n_fft // 2 + 1, num_frames).
        # KNOWN ISSUE: the docstring gives the shape as (num_frames, n_fft//2+1), which
        # is transposed; complex output also loses its imaginary part in the evaluator's
        # float() comparison.
        """
        Compute STFT.

        Args:
            signal: (N,) time-domain signal

        Returns:
            stft: (num_frames, n_fft//2+1) complex spectrogram
        """
        return torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )


# Problem configuration
signal_length = 16000 * 10  # 10 seconds at 16kHz


def get_inputs():
    # [FUNCTION DOC]
    # Returns [signal] random normal of length signal_length.
    # Audio signal
    signal = torch.randn(signal_length)
    return [signal]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [1024, 256, "hann"] (n_fft, hop_length, window).
    return [1024, 256, "hann"]  # n_fft, hop_length, window
