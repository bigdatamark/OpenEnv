# [MODULE DOC]
# kernrl problem (level 6): Sobel gradient magnitude and direction of a
# 1920 x 1080 grayscale image (zero padding).
"""
Sobel Edge Detection

Computes image gradients using Sobel operators and combines into edge magnitude.
Classic image processing operation.

Optimization opportunities:
- Separable convolution (Sobel is separable)
- Shared memory tiling
- Fusing Gx, Gy computation with magnitude
- Vectorized processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # [CLASS DOC]
    # Reference Sobel model with 3x3 x- and y-gradient kernels registered as
    # (1, 1, 3, 3) buffers.
    """
    Sobel edge detection filter.

    Computes horizontal and vertical gradients, then magnitude.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Registers the Sobel x and y kernels.
        super(Model, self).__init__()

        # Sobel kernels
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, image: torch.Tensor) -> tuple:
        # [FUNCTION DOC]
        # Convolves the (H, W) image with both kernels (padding 1, cross-correlation as
        # in conv2d) and returns the tuple (magnitude = sqrt(gx^2 + gy^2),
        # angle = atan2(gy, gx)), each (H, W).
        # KNOWN ISSUE: the evaluator's correctness check cannot convert a tuple of
        # tensors, so this output format fails comparison.
        """
        Apply Sobel edge detection.

        Args:
            image: (H, W) grayscale image

        Returns:
            magnitude: (H, W) edge magnitude
            angle: (H, W) gradient direction in radians
        """
        # Add batch and channel dimensions
        x = image.unsqueeze(0).unsqueeze(0)

        # Compute gradients
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)

        # Remove batch and channel dimensions
        gx = gx.squeeze(0).squeeze(0)
        gy = gy.squeeze(0).squeeze(0)

        # Compute magnitude and angle
        magnitude = torch.sqrt(gx**2 + gy**2)
        angle = torch.atan2(gy, gx)

        return magnitude, angle


# Problem configuration
image_height = 1920
image_width = 1080


def get_inputs():
    # [FUNCTION DOC]
    # Returns [image] uniform random of shape (image_height, image_width).
    # Grayscale image
    image = torch.rand(image_height, image_width)
    return [image]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
