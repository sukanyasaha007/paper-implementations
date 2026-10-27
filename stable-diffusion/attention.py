import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # One projection produces Q, K and V together
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x - (batch, seq, dim)
        batch, seq_len, d_embed = x.shape
        interim_shape = (batch, seq_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        # (batch, seq, dim) -> (batch, heads, seq, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        scores = q @ k.transpose(-1, -2)
        if causal_mask:
            # Mask out future positions (upper triangle above the diagonal)
            mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
        scores = scores / math.sqrt(self.d_head)
        scores = scores.softmax(dim=-1)
        values = scores @ v  # (batch, heads, seq, d_head)
        # Back to (batch, seq, heads, d_head) before merging heads; a plain view without the transpose
        # would interleave values from different heads
        values = values.transpose(1, 2).contiguous()
        values = values.view(x.shape)
        return self.out_proj(values)


class CrossAttention(nn.Module):
    """The image asks the prompt questions: queries come from the latent pixels, keys and values from CLIP's text."""

    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        # Keys and values are projected from the 768-dim text context into the image's channel size
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x latent - (B, seq_len_q, dim_q)
        # y context - (B, 77, 768)
        input_shape = x.shape
        batch = input_shape[0]
        interim_shape = (batch, -1, self.n_heads, self.d_head)

        q = self.q_proj(x).view(interim_shape).transpose(1, 2)
        k = self.k_proj(y).view(interim_shape).transpose(1, 2)
        v = self.v_proj(y).view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).contiguous().view(input_shape)
        return self.out_proj(output)
