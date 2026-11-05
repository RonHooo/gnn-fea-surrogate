"""Slice-attention surrogate (after the physics-attention of Transolver, Wu et al. 2024).

Full self-attention over 3-5k mesh nodes costs O(N^2) per layer. Instead, every layer
  1. softly assigns each node to M "slices" (softmax over M learned scores per node),
  2. forms one token per slice as the weighted average of its nodes' features,
  3. runs multi-head self-attention among the M tokens (gnnsurrogate.attention, written by hand),
  4. sends the updated tokens back to the nodes with the same assignment weights,
which costs O(N M). Every node can receive information from every other node in one layer, while a
message-passing layer only reaches the direct neighbours.

Simplifications compared with Transolver: one set of slice weights is shared by all attention heads
(Transolver computes separate slice weights per head), and the global outputs come from attention
pooling over the nodes.

The model uses the same batches as the GNN (gnnsurrogate.data.collate: several graphs concatenated,
padded with a dummy graph). Slices are formed per graph, so designs in a batch never mix.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .attention import MultiHeadSelfAttention, softmax
from .model import mlp, scatter_mean


class FourierFeatures(nn.Module):
    """[sin(2^k pi x), cos(2^k pi x)] for k = 0..n_freq-1 and each coordinate (positions are in units
    of 100 mm, so the shortest wavelength with n_freq = 6 is 2 / 32 * 100 mm = 6 mm)."""

    def __init__(self, n_freq: int = 6):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq, dtype=torch.float32))

    @property
    def n_out(self) -> int:
        return 3 * 2 * len(self.freqs)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        a = p[:, :, None] * self.freqs                                   # (N, 3, F)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1).flatten(1)


class SliceAttention(nn.Module):
    def __init__(self, dim: int, heads: int, n_slices: int):
        super().__init__()
        self.to_logits = nn.Linear(dim, n_slices)
        self.to_fx = nn.Linear(dim, dim)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.5)))       # softmax temperature (Transolver init)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.to_out = nn.Linear(dim, dim)

    def slice_weights(self, h: torch.Tensor) -> torch.Tensor:
        return softmax(self.to_logits(h) / torch.exp(self.log_tau), dim=-1)      # (N, M), rows sum to 1

    def forward(self, h: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
        """h: (N, D) node features of all graphs in the batch; member: (B, N) 0/1 graph membership."""
        wb = member[:, :, None] * self.slice_weights(h)[None]           # (B, N, M), zero outside graph b
        mass = wb.sum(dim=1)                                             # (B, M)
        tokens = torch.einsum("bnm,nd->bmd", wb, self.to_fx(h)) / (mass[..., None] + 1e-5)
        tokens = self.attn(tokens)                                       # (B, M, D)
        return self.to_out(torch.einsum("bnm,bmd->nd", wb, tokens))     # back to the nodes


class LocalMP(nn.Module):
    """Optional local step (ablation A4): each node mixes in the mean of its mesh neighbours."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, h, edge_index):
        src, dst = edge_index
        return self.net(torch.cat([h, scatter_mean(h[src], dst, h.shape[0])], dim=1))


class Block(nn.Module):
    """Pre-norm transformer block: h + SliceAttention(LN(h)), then h + FFN(LN(h))."""

    def __init__(self, dim: int, heads: int, n_slices: int, mlp_ratio: int = 2, local: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SliceAttention(dim, heads, n_slices)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.local = None
        if local:
            self.ln0 = nn.LayerNorm(dim)
            self.local = LocalMP(dim)

    def forward(self, h, member, edge_index):
        if self.local is not None:
            h = h + self.local(self.ln0(h), edge_index)
        h = h + self.attn(self.ln1(h), member)
        return h + self.ffn(self.ln2(h))


class AttentionPool(nn.Module):
    """Per-graph weighted average of the node features, with `heads` separate learned weightings."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.score = nn.Linear(dim, heads)

    def forward(self, h: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
        s = self.score(h).T[None]                                        # (1, P, N)
        s = s.masked_fill(member[:, None, :] == 0, float("-inf"))        # (B, P, N)
        a = softmax(s, dim=-1)
        return (a @ h).flatten(1)                                        # (B, P * D)


class SliceTransformer(nn.Module):
    def __init__(self, node_in: int = 6, n_params: int = 6, hidden: int = 96, layers: int = 6,
                 heads: int = 4, n_slices: int = 32, mlp_ratio: int = 2, n_freq: int = 6,
                 use_fourier: bool = True, use_params: bool = True, local: bool = False,
                 node_out: int = 9, global_out: int = 3, pool_heads: int = 4):
        super().__init__()
        self.use_params = use_params
        self.fourier = FourierFeatures(n_freq) if use_fourier else None
        n_in = node_in + (n_params if use_params else 0) + (self.fourier.n_out if use_fourier else 0)
        self.embed = mlp(n_in, hidden, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads, n_slices, mlp_ratio, local) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(hidden)
        self.node_dec = mlp(hidden, hidden, node_out, norm=False)
        self.pool = AttentionPool(hidden, pool_heads)
        self.glob_dec = mlp(pool_heads * hidden, hidden, global_out, norm=False)

    def forward(self, b):
        x = b.x
        parts = [x]
        if self.fourier is not None:
            parts.append(self.fourier(x[:, :3]))
        if self.use_params:
            parts.append(b.params[b.batch])
        h = self.embed(torch.cat(parts, dim=1))
        member = F.one_hot(b.batch, b.n_slots).T.to(h.dtype)             # (B, N)
        for blk in self.blocks:
            h = blk(h, member, b.edge_index)
        h = self.ln_out(h)
        return self.node_dec(h), self.glob_dec(self.pool(h, member))[:b.n_graphs]
