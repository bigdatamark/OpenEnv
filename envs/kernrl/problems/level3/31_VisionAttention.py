# [MODULE DOC]
# kernrl problem (level 3, full blocks): multi-head self-attention plus
# residual LayerNorm over the spatial positions of a (2, 128, 128, 128) feature
# map (sequence length 16384, embed dim 128, 4 heads).
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # [CLASS DOC]
    # Reference attention block wrapping nn.MultiheadAttention (sequence-first
    # layout) and a LayerNorm over the embedding dim.
    def __init__(self, embed_dim, num_heads):
        # [FUNCTION DOC]
        # Creates the MultiheadAttention (embed_dim, num_heads) and LayerNorm modules.
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # [FUNCTION DOC]
        # Flattens (B, C, H, W) to a (H*W, B, C) sequence, applies self-attention,
        # adds the residual, applies LayerNorm, and reshapes back to (B, C, H, W).
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)  # (seq_len, batch_size, embed_dim)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x


embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, num_channels, height, width).
    return [torch.randn(batch_size, num_channels, image_height, image_width)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [embed_dim, num_heads].
    return [embed_dim, num_heads]
