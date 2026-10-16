# [MODULE DOC]
# kernrl problem (level 7): 5 x 5 median filter of a 512 x 512 image with
# salt-and-pepper noise. Note: torch.nn.functional is imported below the class
# definition, which works because it is only used at call time.
"""
2D Median Filter

Non-linear filter that replaces each pixel with the median of its neighborhood.
Excellent for removing salt-and-pepper noise while preserving edges.

Challenge: Finding median requires sorting - not a simple convolution.

Optimization opportunities:
- Histogram-based median (for byte images)
- Partial sort using selection networks
- Running median algorithms
- Separable approximations
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference median filter model storing kernel size and radius.
    """
    2D median filter for noise removal.
    """

    def __init__(self, kernel_size: int = 5):
        # [FUNCTION DOC]
        # Stores kernel_size (default 5) and radius (k // 2).
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.radius = kernel_size // 2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Reflect-pads the (H, W) image by the radius, extracts all k x k windows with
        # unfold, and returns the per-pixel median (the lower median for even counts)
        # as an (H, W) tensor.
        # KNOWN ISSUE: reflect padding of a plain 2D tensor with a 4-value pad may be
        # rejected by some PyTorch versions.
        """
        Apply median filter.

        Args:
            image: (H, W) input image

        Returns:
            filtered: (H, W) median filtered image
        """
        H, W = image.shape
        radius = self.radius
        ks = self.kernel_size

        # Pad image
        padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")

        # Unfold to get all windows
        # Shape: (H, W, ks, ks)
        windows = padded.unfold(0, ks, 1).unfold(1, ks, 1)

        # Flatten windows and compute median
        # Shape: (H, W, ks*ks)
        windows_flat = windows.reshape(H, W, -1)

        # Median along last dimension
        filtered = windows_flat.median(dim=-1)[0]

        return filtered


# Need F for padding
import torch.nn.functional as F

# Problem configuration
image_height = 512
image_width = 512


def get_inputs():
    # [FUNCTION DOC]
    # Returns [image] uniform random of shape (image_height, image_width) with
    # about 5% of pixels set to 0.0 and 5% set to 1.0.
    # Image with salt-and-pepper noise
    image = torch.rand(image_height, image_width)
    # Add noise
    noise_mask = torch.rand(image_height, image_width)
    image[noise_mask < 0.05] = 0.0  # Salt
    image[noise_mask > 0.95] = 1.0  # Pepper
    return [image]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [5] (kernel_size).
    return [5]  # kernel_size
