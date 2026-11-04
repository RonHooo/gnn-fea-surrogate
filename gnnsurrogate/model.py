"""MeshGraphNet-style encode-process-decode GNN (Pfaff et al. 2021) with an optional global latent.

Static linear elasticity is a global problem: the stress near the gussets depends on the load at the
mounting hole 80+ mm away, which is far more hops than the number of message-passing layers. The
global latent (a "virtual node" that reads the mean of all node latents and is fed back to every node
in every layer) gives each layer a path for long-range information.
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(n_in: int, n_hidden: int, n_out: int, layers: int = 2, norm: bool = True) -> nn.Sequential:
    mods, d = [], n_in
    for _ in range(layers - 1):
        mods += [nn.Linear(d, n_hidden), nn.SiLU()]
        d = n_hidden
    mods.append(nn.Linear(d, n_out))
    if norm:
        mods.append(nn.LayerNorm(n_out))
    return nn.Sequential(*mods)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, src.shape[1], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(n, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, src.shape[1], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def scatter_max(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.full((n, src.shape[1]), -1e9, device=src.device, dtype=src.dtype)
    return out.scatter_reduce(0, index.unsqueeze(1).expand_as(src), src, reduce="amax", include_self=True)


class ProcessorBlock(nn.Module):
    def __init__(self, h: int, use_global: bool):
        super().__init__()
        self.use_global = use_global
        k = 1 if use_global else 0
        # Edge MLP: Linear([e, v_src, v_dst, u]) -> SiLU -> Linear -> LayerNorm. The first layer is
        # split into per-block weights so the node and global terms are projected once per node
        # and then gathered; this is the same function but avoids storing the (E, 4h) concatenation.
        self.e_lin = nn.Linear(h, h)
        self.s_lin = nn.Linear(h, h, bias=False)
        self.d_lin = nn.Linear(h, h, bias=False)
        if use_global:
            self.u_lin = nn.Linear(h, h, bias=False)
        self.e_out = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.LayerNorm(h))
        self.node = mlp(2 * h + k * h, h, h)
        if use_global:
            self.glob = mlp(2 * h, h, h)

    def forward(self, v, e, u, edge_index, batch, n_graphs):
        src, dst = edge_index
        ps = self.s_lin(v)
        if self.use_global:
            ps = ps + self.u_lin(u)[batch]
        e = e + self.e_out(self.e_lin(e) + ps[src] + self.d_lin(v)[dst])
        agg = scatter_sum(e, dst, v.shape[0])
        parts = [v, agg]
        if self.use_global:
            parts.append(u[batch])
        v = v + self.node(torch.cat(parts, dim=1))
        if self.use_global:
            u = u + self.glob(torch.cat([u, scatter_mean(v, batch, n_graphs)], dim=1))
        return v, e, u


class MeshGNN(nn.Module):
    def __init__(self, node_in: int = 6, edge_in: int = 4, n_params: int = 6, hidden: int = 64,
                 layers: int = 10, node_out: int = 4, global_out: int = 3,
                 use_params: bool = True, use_global: bool = True):
        super().__init__()
        self.use_params, self.use_global = use_params, use_global
        self.node_enc = mlp(node_in + (n_params if use_params else 0), hidden, hidden)
        self.edge_enc = mlp(edge_in, hidden, hidden)
        self.glob_enc = mlp(n_params if use_params else 1, hidden, hidden)
        self.blocks = nn.ModuleList([ProcessorBlock(hidden, use_global) for _ in range(layers)])
        self.node_dec = mlp(hidden, hidden, node_out, norm=False)
        self.glob_dec = mlp(3 * hidden, hidden, global_out, norm=False)

    def forward(self, b):
        n = b.n_slots
        x = b.x
        if self.use_params:
            x = torch.cat([x, b.params[b.batch]], dim=1)
            gin = b.params
        else:
            gin = torch.zeros(n, 1, device=x.device)
        v = self.node_enc(x)
        e = self.edge_enc(b.edge_attr)
        u = self.glob_enc(gin)
        for blk in self.blocks:
            v, e, u = blk(v, e, u, b.edge_index, b.batch, n)
        node_out = self.node_dec(v)
        pooled = torch.cat([scatter_mean(v, b.batch, n), scatter_max(v, b.batch, n), u], dim=1)
        return node_out, self.glob_dec(pooled)[:b.n_graphs]


class PointMLP(nn.Module):
    """Baseline without mesh connectivity: each node is predicted from its own features + parameters."""

    def __init__(self, node_in: int = 6, n_params: int = 6, hidden: int = 256, layers: int = 6,
                 node_out: int = 4):
        super().__init__()
        self.net = mlp(node_in + n_params, hidden, node_out, layers=layers, norm=False)

    def forward(self, b):
        return self.net(torch.cat([b.x, b.params[b.batch]], dim=1)), None
