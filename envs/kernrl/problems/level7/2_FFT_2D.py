# [MODULE DOC]
# kernrl problem (level 7): 2D FFT of a 2048 x 2048 real image.
"""
2D Fast Fourier Transform

Computes 2D DFT, commonly used in image processing for:
- Frequency domain filtering
- Convolution via multiplication
- Pattern detection

Optimization opportunities:
- Row-column decomposition
- Shared memory for partial transforms
- Batched 1D FFTs
- Tiled computation for large images
"""

import torch
import torch.fft
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference 2D FFT model. No parameters.
    """
    2D Fast Fourier Transform.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns torch.fft.fft2(image) over the last two dims (complex output).
        # KNOWN ISSUE: complex outputs lose their imaginary part in the evaluator's
        # float() comparison.
        """
        Compute 2D FFT.

        Args:
            image: (H, W) real or complex 2D array

        Returns:
            spectrum: (H, W) complex 2D frequency components
        """
        return torch.fft.fft2(image)


# Problem configuration
image_height = 2048
image_width = 2048


def get_inputs():
    # [FUNCTION DOC]
    # Returns [image] random normal of shape (image_height, image_width).
    image = torch.randn(image_height, image_width)
    return [image]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
