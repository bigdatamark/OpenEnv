# [MODULE DOC]
# kernrl problem (level 6): 11 x 11 box (mean) filter of a 1920 x 1080 image
# with zero padding.
"""
Box Filter (Moving Average)

Computes local mean in a rectangular window.
Very common image processing operation (smoothing, feature computation).

Optimization opportunities:
- Integral images (summed area tables)
- Separable implementation (row sum + column sum)
- Sliding window with running sum
- O(1) computation per pixel using prefix sums
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # [CLASS DOC]
    # Reference box filter model with a uniform normalized kernel buffer.
    """
    Box filter (uniform averaging filter).

    Computes the mean of all pixels in a rectangular window.
    """

    def __init__(self, kernel_size: int = 11):
        # [FUNCTION DOC]
        # Stores kernel size and padding and registers the (1, 1, k, k) kernel of
        # value 1 / k^2.
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Uniform kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (
            kernel_size * kernel_size
        )
        self.register_buffer("kernel", kernel)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Convolves the (H, W) image with the uniform kernel (zero padding, same size)
        # and returns the (H, W) result.
        """
        Apply box filter.

        Args:
            image: (H, W) input image

        Returns:
            filtered: (H, W) box filtered image
        """
        x = image.unsqueeze(0).unsqueeze(0)
        filtered = F.conv2d(x, self.kernel, padding=self.padding)
        return filtered.squeeze(0).squeeze(0)


# Problem configuration
image_height = 1920
image_width = 1080


def get_inputs():
    # [FUNCTION DOC]
    # Returns [image] uniform random of shape (image_height, image_width).
    image = torch.rand(image_height, image_width)
    return [image]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [11] (kernel_size).
    return [11]  # kernel_size
