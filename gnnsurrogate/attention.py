"""Scaled dot-product attention and multi-head self-attention, written out by hand.

These are the reference implementations used by the slice-attention surrogate (transformer.py).
tests/test_attention.py checks them against torch.nn.functional.scaled_dot_product_attention and
torch.nn.MultiheadAttention.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax: subtracting the row maximum does not change the result but keeps
    exp() from overflowing."""
    z = x - x.amax(dim=dim, keepdim=True)
    e = torch.exp(z)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """softmax(Q K^T / sqrt(d)) V.

    q: (..., Lq, d), k: (..., Lk, d), v: (..., Lk, dv) -> (..., Lq, dv).
    mask: optional bool tensor broadcastable to (..., Lq, Lk); True means "may attend".
    Every query must be allowed to attend to at least one key.
    """
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ v


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention on a (B, L, D) sequence.

    One linear layer produces Q, K and V (same layout as nn.MultiheadAttention's in_proj), the
    D features are split into `heads` groups of D / heads, each group attends separately, and the
    heads are concatenated and mixed by an output projection.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} is not divisible by heads={heads}")
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.reshape(b, l, self.heads, self.dh).transpose(1, 2)      # (B, H, L, dh)

    @staticmethod
    def merge_heads(x: torch.Tensor) -> torch.Tensor:
        b, h, l, dh = x.shape
        return x.transpose(1, 2).reshape(b, l, h * dh)                   # (B, L, D)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        o = scaled_dot_product_attention(self.split_heads(q), self.split_heads(k), self.split_heads(v), mask)
        return self.out(self.merge_heads(o))
