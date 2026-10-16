# [MODULE DOC]
# kernrl problem (level 8): simplified H.264-style deblocking of 8 x 8 block
# edges in a 720 x 1280 frame; the reference loops over edge pixels in Python.
"""
Deblocking Filter (H.264/H.265 Style)

Reduces blocking artifacts at block boundaries in compressed video.
Applied in-loop during video decoding.

The filter adaptively smooths edges based on local gradient analysis.

Optimization opportunities:
- Parallel edge filtering (horizontal and vertical)
- SIMD comparisons for filter decisions
- Shared memory for boundary pixels
- In-place filtering with proper dependencies
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference deblocking model with block size, edge threshold alpha and neighbor
    # threshold beta.
    """
    Simple deblocking filter for 8x8 block boundaries.

    Smooths block edges adaptively based on gradient strength.
    """

    def __init__(self, block_size: int = 8, alpha: float = 0.1, beta: float = 0.05):
        # [FUNCTION DOC]
        # Stores block_size (default 8), alpha (default 0.1) and beta (default 0.05).
        super(Model, self).__init__()
        self.block_size = block_size
        self.alpha = alpha  # Edge threshold
        self.beta = beta  # Neighbor threshold

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # First filters vertical block edges (x = bs, 2bs, ... below W - 1) reading from
        # the input, then horizontal edges reading from the vertically filtered result.
        # At each edge pixel pair (p0 | q0) with outer neighbors p1, q1, if
        # |p0 - q0| < alpha and both |p1 - p0| and |q1 - q0| < beta, replaces p0 with
        # (2p1 + p0 + q0 + 2) / 4 and q0 with (2q1 + q0 + p0 + 2) / 4. Returns the
        # filtered (H, W) frame.
        # KNOWN ISSUE: the "+2" rounding constant comes from integer pixel arithmetic;
        # with float pixels in [0, 1] it adds a 0.5 bias to every filtered pixel.
        """
        Apply deblocking filter.

        Args:
            frame: (H, W) reconstructed frame with blocking artifacts

        Returns:
            filtered: (H, W) deblocked frame
        """
        H, W = frame.shape
        bs = self.block_size
        output = frame.clone()

        # Filter vertical edges (at x = k * block_size for k > 0)
        for x in range(bs, W - 1, bs):
            for y in range(H):
                # Get boundary pixels
                p0 = frame[y, x - 1]
                p1 = frame[y, x - 2] if x >= 2 else p0
                q0 = frame[y, x]
                q1 = frame[y, x + 1] if x < W - 1 else q0

                # Check if filtering should be applied
                if abs(p0 - q0) < self.alpha:
                    if abs(p1 - p0) < self.beta and abs(q1 - q0) < self.beta:
                        # Apply 4-tap filter
                        p0_new = (2 * p1 + p0 + q0 + 2) / 4
                        q0_new = (2 * q1 + q0 + p0 + 2) / 4
                        output[y, x - 1] = p0_new
                        output[y, x] = q0_new

        # Filter horizontal edges (at y = k * block_size for k > 0)
        frame = output.clone()
        for y in range(bs, H - 1, bs):
            for x in range(W):
                p0 = frame[y - 1, x]
                p1 = frame[y - 2, x] if y >= 2 else p0
                q0 = frame[y, x]
                q1 = frame[y + 1, x] if y < H - 1 else q0

                if abs(p0 - q0) < self.alpha:
                    if abs(p1 - p0) < self.beta and abs(q1 - q0) < self.beta:
                        p0_new = (2 * p1 + p0 + q0 + 2) / 4
                        q0_new = (2 * q1 + q0 + p0 + 2) / 4
                        output[y - 1, x] = p0_new
                        output[y, x] = q0_new

        return output


# Problem configuration
frame_height = 720
frame_width = 1280


def get_inputs():
    # [FUNCTION DOC]
    # Returns [frame] uniform random of shape (frame_height, frame_width).
    # Simulated blocky frame
    frame = torch.rand(frame_height, frame_width)
    return [frame]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [8, 0.1, 0.05] (block_size, alpha, beta).
    return [8, 0.1, 0.05]  # block_size, alpha, beta
