# [MODULE DOC]
# kernrl problem (level 6): per-pixel RGB to YUV (BT.601-style coefficients)
# conversion of a 1920 x 1080 x 3 image.
"""
Color Space Conversion - RGB to YUV

Converts RGB image to YUV color space.
This is a pixel-wise matrix multiplication, highly parallelizable.

Y =  0.299*R + 0.587*G + 0.114*B
U = -0.147*R - 0.289*G + 0.436*B
V =  0.615*R - 0.515*G - 0.100*B

Optimization opportunities:
- Vectorized pixel processing
- Fused multiply-add (FMA) instructions
- Coalesced memory access patterns
- In-place conversion if possible
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model holding the 3 x 3 RGB-to-YUV matrix as a buffer.
    """
    RGB to YUV color space conversion.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Registers the conversion matrix.
        super(Model, self).__init__()

        # Conversion matrix
        rgb_to_yuv = torch.tensor(
            [[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.100]],
            dtype=torch.float32,
        )

        self.register_buffer("conversion_matrix", rgb_to_yuv)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns rgb @ matrix^T over the last (channel) dimension, same shape as the
        # input.
        """
        Convert RGB to YUV.

        Args:
            rgb: (H, W, 3) or (B, H, W, 3) RGB image in [0, 1]

        Returns:
            yuv: same shape, YUV values
        """
        # Matrix multiply on last dimension
        yuv = torch.matmul(rgb, self.conversion_matrix.T)
        return yuv


# Problem configuration
image_height = 1920
image_width = 1080


def get_inputs():
    # [FUNCTION DOC]
    # Returns [rgb] uniform random of shape (image_height, image_width, 3).
    # RGB image
    rgb = torch.rand(image_height, image_width, 3)
    return [rgb]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
