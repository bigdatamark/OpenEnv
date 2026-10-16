# [MODULE DOC]
# kernrl problem (level 8): homography warp of a 1080 x 1920 frame (small
# rotation plus translation) for video stabilization.
"""
Video Stabilization Transform

Applies homography transformations to stabilize video frames.
Core operation in video stabilization pipelines.

Optimization opportunities:
- Batched homography warping
- Texture memory for source frame
- Bilinear/bicubic interpolation
- Parallel per-pixel transform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # [CLASS DOC]
    # Reference homography warp model. No parameters.
    """
    Applies homography transformation to stabilize a frame.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(Model, self).__init__()

    def forward(self, frame: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Maps every destination pixel through the inverse homography (homogeneous
        # divide with +1e-10), normalizes source coordinates to [-1, 1] (align_corners
        # convention), and samples the frame bilinearly with zero padding. Returns the
        # warped frame in the input's shape ((H, W) or (C, H, W)).
        """
        Warp frame using homography matrix.

        Args:
            frame: (H, W) or (C, H, W) input frame
            homography: (3, 3) homography matrix (source to destination)

        Returns:
            warped: same shape as input, warped frame
        """
        if frame.dim() == 2:
            frame = frame.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        C, H, W = frame.shape

        # Create destination coordinates
        y_coords = torch.arange(H, device=frame.device).float()
        x_coords = torch.arange(W, device=frame.device).float()
        Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Homogeneous coordinates (3, H*W)
        ones = torch.ones_like(X)
        dst_coords = torch.stack([X, Y, ones], dim=0).reshape(3, -1)

        # Apply inverse homography to get source coordinates
        H_inv = torch.linalg.inv(homography)
        src_coords = H_inv @ dst_coords

        # Normalize by homogeneous coordinate
        src_coords = src_coords[:2] / (src_coords[2:3] + 1e-10)

        # Reshape to (H, W, 2)
        src_x = src_coords[0].reshape(H, W)
        src_y = src_coords[1].reshape(H, W)

        # Normalize to [-1, 1] for grid_sample
        src_x_norm = 2 * src_x / (W - 1) - 1
        src_y_norm = 2 * src_y / (H - 1) - 1
        grid = torch.stack([src_x_norm, src_y_norm], dim=-1)

        # Warp
        frame_batch = frame.unsqueeze(0)
        grid_batch = grid.unsqueeze(0)

        warped = F.grid_sample(
            frame_batch,
            grid_batch,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped = warped.squeeze(0)

        if squeeze:
            warped = warped.squeeze(0)

        return warped


# Problem configuration
frame_height = 1080
frame_width = 1920


def get_inputs():
    # [FUNCTION DOC]
    # Returns a uniform random frame and a 3x3 homography of a 0.02 rad rotation
    # with translation (5, 3).
    frame = torch.rand(frame_height, frame_width)
    # Small rotation + translation homography
    angle = 0.02  # Small angle
    tx, ty = 5.0, 3.0  # Small translation
    cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    homography = torch.tensor(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty], [0.0, 0.0, 1.0]]
    )
    return [frame, homography]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns an empty list (no constructor arguments).
    return []
