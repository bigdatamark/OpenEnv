# [MODULE DOC]
# kernrl problem (level 4, modern LLM operators): DeepSeek-V3 Multi-head Latent
# Attention (low-rank compressed Q and KV projections, decoupled RoPE and
# non-RoPE head dims, causal attention), scaled down (batch 4, sequence 2048,
# hidden 2048, 16 heads).
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# DeepSeek-V3 Multi-head Latent Attention (MLA)
# Source: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
#
# MLA compresses KV projections through low-rank decomposition:
# - Standard attention: Q, K, V each projected from hidden_size to num_heads * head_dim
# - MLA: KV compressed to kv_lora_rank, then expanded. Q optionally compressed via q_lora_rank.
# - Decoupled RoPE: Separate rope/nope head dimensions for positional vs non-positional attention
#
# This HuggingFace implementation uses naive PyTorch ops - a fused CUDA kernel can
# significantly accelerate the compression/expansion and attention computation.


class DeepSeekRMSNorm(nn.Module):
    # [CLASS DOC]
    # HELPER: RMSNorm with a learnable weight (initialized to ones).
    def __init__(self, hidden_size, eps=1e-6):
        # [FUNCTION DOC]
        # Creates the weight parameter and stores eps (default 1e-6).
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # [FUNCTION DOC]
        # Normalizes in float32 by the root mean square over the last dim, casts back
        # to the input dtype and multiplies by the weight.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    # [FUNCTION DOC]
    # HELPER: Returns x with its last-dim halves swapped and the new first half
    # negated (concat of -second_half and first_half).
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # [FUNCTION DOC]
    # HELPER: Applies rotary embedding: q*cos + rotate_half(q)*sin and the same for
    # k, after unsqueezing cos/sin at unsqueeze_dim for head broadcasting.
    # position_ids is unused. Returns (q_embed, k_embed).
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepSeekRotaryEmbedding(nn.Module):
    # [CLASS DOC]
    # HELPER: Rotary embedding table generator with inverse frequencies
    # base^(-2i/dim) stored as a non-persistent buffer.
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        # [FUNCTION DOC]
        # Stores dim, max positions and base and registers the inverse frequencies.
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        # [FUNCTION DOC]
        # Without gradients, builds position angles for positions 0..seq_len-1
        # (default x.shape[-2]) and returns (cos, sin) tensors of shape (seq_len, dim).
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class Model(nn.Module):
    # [CLASS DOC]
    # Reference MLA layer: q path hidden -> q_lora_rank (RMSNorm) -> heads x
    # (nope + rope dims); kv path hidden -> kv_lora_rank + shared rope key, RMSNorm,
    # then expanded to per-head nope keys and values; rotary embeddings; output
    # projection. Softmax scale is q_head_dim^-0.5.
    """
    DeepSeek-V3 Multi-head Latent Attention (MLA)

    Key optimizations targets:
    1. Fused LoRA compression/expansion for Q and KV
    2. Fused RoPE application with decoupled nope/rope heads
    3. Fused attention with softmax scaling
    4. Memory-efficient KV compression pathway
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        # [FUNCTION DOC]
        # Stores dimensions and creates all bias-free projections, the two RMSNorms and
        # the rotary embedding.
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.attention_dropout = attention_dropout
        self.softmax_scale = self.q_head_dim ** (-0.5)

        # Query projection with LoRA compression
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = DeepSeekRMSNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(
            q_lora_rank, num_attention_heads * self.q_head_dim, bias=False
        )

        # KV projection with LoRA compression (MQA-style: shared across heads initially)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = DeepSeekRMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            num_attention_heads * v_head_dim, hidden_size, bias=False
        )

        # Rotary embeddings
        self.rotary_emb = DeepSeekRotaryEmbedding(
            qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Computes per-head queries split into nope/rope parts, the compressed KV and a
        # single shared rope key, expands KV into nope keys and values, applies RoPE to
        # the rope parts, concatenates into full query/key states (rope key broadcast to
        # all heads), computes scaled scores with a causal mask, softmax in float32,
        # dropout (training only), weighted values, and the output projection.
        # Returns (batch, seq_len, hidden_size).
        bsz, q_len, _ = hidden_states.size()

        # Query projection with LoRA compression
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # Split query into nope (non-positional) and rope (positional) components
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # KV projection with compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Expand compressed KV
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(
            bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        kv = kv.transpose(1, 2)

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # Apply rotary embeddings to positional components only
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        # Assemble full query and key states
        query_states = torch.empty(
            bsz,
            self.num_heads,
            q_len,
            self.q_head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = torch.empty(
            bsz,
            self.num_heads,
            q_len,
            self.q_head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        # Compute attention
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


# DeepSeek-V3 style configuration (scaled down for single H100)
batch_size = 4
seq_len = 2048
hidden_size = 2048
num_attention_heads = 16
q_lora_rank = 1536
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
max_position_embeddings = 4096


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, seq_len, hidden_size).
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns the eight constructor arguments (hidden size, heads, q/kv LoRA ranks,
    # nope/rope/value head dims, max positions).
    return [
        hidden_size,
        num_attention_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        max_position_embeddings,
    ]
