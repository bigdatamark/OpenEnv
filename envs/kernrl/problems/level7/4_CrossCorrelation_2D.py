# [MODULE DOC]
# kernrl problem (level 7): 2D cross-correlation (template matching) of a
# 1024 x 1024 image with a random 32 x 32 template.
"""
2D Cross-Correlation (Template Matching)

Slides a template over an image and computes correlation at each position.
Used for template matching, feature detection, and pattern recognition.

Optimization opportunities:
- FFT-based correlation for large templates
- Shared memory for template caching
- Normalized cross-correlation variants
- Integral images for sum computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model with a random (1, 1, th, tw) template buffer.
    """
    2D cross-correlation for template matching.
    """

    def __init__(self, template_height: int = 32, template_width: int = 32):
        # [FUNCTION DOC]
        # Stores template dimensions and registers a random normal template.
        super(Model, self).__init__()
        self.template_height = template_height
        self.template_width = template_width

        # Random template (in practice, this would be a pattern to find)
        template = torch.randn(1, 1, template_height, template_width)
        self.register_buffer("template", template)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Cross-correlates the image with the template via conv2d with padding
        # (th // 2, tw // 2) and returns the 2D correlation map.
        # KNOWN ISSUE: with even template sizes the output is (H + 1, W + 1), not the
        # (H, W) the docstring claims.
        """
        Compute cross-correlation between image and template.

        Args:
            image: (H, W) input image

        Returns:
            correlation: (H, W) correlation map (same size with padding)
        """
        x = image.unsqueeze(0).unsqueeze(0)

        # Valid padding would give (H-Th+1, W-Tw+1)
        # Use same padding for consistent size
        pad_h = self.template_height // 2
        pad_w = self.template_width // 2

        correlation = F.conv2d(x, self.template, padding=(pad_h, pad_w))

        return correlation.squeeze(0).squeeze(0)


# Problem configuration
image_height = 1024
image_width = 1024


def get_inputs():
    # [FUNCTION DOC]
    # Returns [image] random normal of shape (image_height, image_width).
    image = torch.randn(image_height, image_width)
    return [image]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [32, 32] (template height and width).
    return [32, 32]  # template_height, template_width
