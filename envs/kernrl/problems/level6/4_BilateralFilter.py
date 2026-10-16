# [MODULE DOC]
# kernrl problem (level 6): 9 x 9 bilateral filter (sigma_spatial 3,
# sigma_range 0.1) of a 256 x 256 grayscale image; the reference loops over
# pixels in Python.
"""
Bilateral Filter

Edge-preserving smoothing filter that considers both spatial proximity
and intensity similarity. Widely used in image denoising and tone mapping.

Challenge: Non-separable, data-dependent weights make optimization harder than Gaussian.

Optimization opportunities:
- Approximate bilateral using bilateral grid
- Constant-time bilateral filter
- Shared memory for local window
- Vectorized weight computation
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference bilateral filter model storing kernel size, spatial and range
    # sigmas, and radius (k // 2).
    """
    Bilateral filter for edge-preserving smoothing.

    Weight = exp(-spatial_dist^2 / (2*sigma_s^2)) * exp(-intensity_diff^2 / (2*sigma_r^2))
    """

    def __init__(
        self, kernel_size: int = 9, sigma_spatial: float = 3.0, sigma_range: float = 0.1
    ):
        # [FUNCTION DOC]
        # Stores the filter parameters.
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range
        self.radius = kernel_size // 2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Reflect-pads the image by the radius; for each pixel computes spatial
        # Gaussian weights over the window and range weights from the intensity
        # difference to the center pixel, and writes the weighted average of the
        # window. Returns the (H, W) filtered image.
        # KNOWN ISSUE: reflect padding of a plain 2D tensor with a 4-value pad may be
        # rejected by some PyTorch versions, which expect a batched/channel dimension
        # for 2D reflect padding.
        """
        Apply bilateral filter.

        Args:
            image: (H, W) grayscale image in [0, 1]

        Returns:
            filtered: (H, W) filtered image
        """
        H, W = image.shape
        radius = self.radius
        sigma_s_sq = 2 * self.sigma_spatial**2
        sigma_r_sq = 2 * self.sigma_range**2

        # Pad image
        padded = torch.nn.functional.pad(
            image, (radius, radius, radius, radius), mode="reflect"
        )

        # Output
        output = torch.zeros_like(image)

        # For each pixel
        for i in range(H):
            for j in range(W):
                center_val = image[i, j]

                # Extract window
                window = padded[i : i + self.kernel_size, j : j + self.kernel_size]

                # Compute spatial weights
                y_coords = torch.arange(self.kernel_size).float() - radius
                x_coords = torch.arange(self.kernel_size).float() - radius
                Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
                spatial_weight = torch.exp(-(X**2 + Y**2) / sigma_s_sq)

                # Compute range weights
                intensity_diff = window - center_val
                range_weight = torch.exp(-(intensity_diff**2) / sigma_r_sq)

                # Combined weight
                weight = spatial_weight.to(image.device) * range_weight

                # Normalize and apply
                weight_sum = weight.sum()
                if weight_sum > 0:
                    output[i, j] = (window * weight).sum() / weight_sum

        return output


# Problem configuration - smaller for bilateral due to O(n^2) per pixel
image_height = 256
image_width = 256


def get_inputs():
    # [FUNCTION DOC]
    # Returns [image] uniform random of shape (image_height, image_width).
    # Grayscale image with some noise
    image = torch.rand(image_height, image_width)
    return [image]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [9, 3.0, 0.1] (kernel_size, sigma_spatial, sigma_range).
    return [9, 3.0, 0.1]  # kernel_size, sigma_spatial, sigma_range
