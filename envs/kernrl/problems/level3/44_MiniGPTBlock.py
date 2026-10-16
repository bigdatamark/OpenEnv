# [MODULE DOC]
# kernrl problem (level 3): a full minGPT transformer block (pre-LN causal
# attention plus GELU MLP with residuals), batch 128, sequence 512, embedding
# 768, 8 heads.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py


class NewGELU(nn.Module):
    # [CLASS DOC]
    # Tanh-approximation GELU module (BERT / GPT style).
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module; no state.
        super(NewGELU, self).__init__()

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    # [CLASS DOC]
    # Causal multi-head self-attention identical to the minGPT reference: fused
    # QKV projection, causal mask buffer, dropouts, output projection.
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        # [FUNCTION DOC]
        # Creates projections, dropouts and the causal mask buffer; asserts n_embd is
        # divisible by n_head.
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(
                1, 1, max_seqlen, max_seqlen
            ),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        # [FUNCTION DOC]
        # Computes masked scaled dot-product attention over heads and returns the
        # projected (B, T, C) output (see the level 3 causal attention problem).
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Model(nn.Module):
    # [CLASS DOC]
    # Reference transformer block: LayerNorm, causal attention, LayerNorm, and an
    # MLP (Linear to 4x width, GELU, Linear back, dropout) exposed via a stored
    # lambda.
    """an unassuming Transformer block"""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        # [FUNCTION DOC]
        # Creates the two LayerNorms, the attention module and the MLP ModuleDict, and
        # stores a lambda applying the MLP layers in order.
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns x + attn(ln_1(x)), then adds mlp(ln_2(...)) as a second residual.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0


def get_inputs():
    # [FUNCTION DOC]
    # Returns [x] with x random of shape (batch_size, seq_len, n_embd).
    return [torch.randn(batch_size, seq_len, n_embd)]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen].
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
