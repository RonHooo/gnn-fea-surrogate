"""Graphs from the bracket FE dataset (corner nodes of the C3D10 meshes, edges of the tetrahedra)."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PARAM_NAMES = ["t_b", "t_w", "r_f", "L_g", "t_g", "d_h"]
# training ranges of the design variables (for scaling to [0, 1]; OOD samples fall outside)
PARAM_LO = np.array([4.0, 4.0, 2.0, 0.0, 3.0, 0.0])
PARAM_HI = np.array([12.0, 10.0, 10.0, 45.0, 8.0, 20.0])
POS_SCALE = 100.0
TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
NODE_TARGETS = ["u_x", "u_y", "u_z", "von_mises", "s_xx", "s_yy", "s_zz", "s_xy", "s_yz", "s_zx"]
# columns of Graph.y that a model is trained on: von Mises directly, or the 6 stress components
# (von Mises is then computed from the predicted components)
TARGET_COLS = {"vm": [0, 1, 2, 3], "stress": [0, 1, 2, 4, 5, 6, 7, 8, 9]}
GLOBAL_TARGETS = ["log_f1", "log10_life", "log_vm_max"]


@dataclass
class Graph:
    x: torch.Tensor           # (N, F) node features
    edge_index: torch.Tensor  # (2, E) directed edges (both directions)
    edge_attr: torch.Tensor   # (E, 4) relative position / POS_SCALE and length
    y: torch.Tensor           # (N, 10) node targets: u (3), von Mises, stress components (6)
    g: torch.Tensor           # (3,) global targets
    params: torch.Tensor      # (6,) scaled design variables
    region: torch.Tensor      # (N,) bool, stress evaluation region
    pos: torch.Tensor         # (N, 3) mm
    meta: dict

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]


def tet_edges(tets: np.ndarray) -> np.ndarray:
    e = tets[:, TET_EDGES].reshape(-1, 2)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    return np.concatenate([e, e[:, ::-1]], axis=0).T          # (2, E)


def graph_inputs(pos: np.ndarray, tets: np.ndarray, flags: np.ndarray):
    """Node features, edges and edge features from a linear tet mesh (no FE results needed)."""
    pos = pos.astype(np.float64)
    flags = flags.astype(np.float32)                           # fix, mount, surface, region
    ei = tet_edges(tets.astype(np.int64))
    rel = (pos[ei[1]] - pos[ei[0]]) / POS_SCALE
    edge_attr = np.concatenate([rel, np.linalg.norm(rel, axis=1, keepdims=True)], axis=1)
    x = np.concatenate([pos / POS_SCALE, flags[:, :3]], axis=1)
    return x, ei, edge_attr


def graph_from_mesh(pos, tets, flags, params_raw, name="candidate") -> Graph:
    """Graph for a design that has not been analysed (targets are NaN)."""
    x, ei, edge_attr = graph_inputs(pos, tets, flags)
    params = (np.asarray(params_raw, dtype=np.float64) - PARAM_LO) / (PARAM_HI - PARAM_LO)
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)
    n = len(pos)
    return Graph(x=t(x), edge_index=t(ei, torch.long), edge_attr=t(edge_attr),
                 y=torch.full((n, 10), float("nan")), g=torch.full((3,), float("nan")), params=t(params),
                 region=t(flags[:, 3] > 0, torch.bool), pos=t(pos),
                 meta=dict(name=name, split="candidate", params_raw=list(map(float, params_raw))))


def load_graph(path: str | Path) -> Graph:
    z = np.load(path)
    pos = z["pos"].astype(np.float64)
    flags = z["flags"].astype(np.float32)
    x, ei, edge_attr = graph_inputs(pos, z["tets"], flags)
    y = np.concatenate([z["u"], z["vm"][:, None], z["s"]], axis=1)
    life = float(np.min(z["life"]))
    vm_corner = float(z["vm_max_fe_corner"]) if "vm_max_fe_corner" in z.files else float("nan")
    g = np.array([np.log(z["freq"][0]), np.log10(life), np.log(float(z["vm_max"]))])
    params = (z["params"] - PARAM_LO) / (PARAM_HI - PARAM_LO)
    name = Path(path).stem
    meta = dict(name=name, split=name.split("_")[0], params_raw=z["params"].tolist(),
                mass=float(z["mass"]), n_full=int(z["n_nodes_full"]),
                vm_max_full=float(z["vm_max"]), vm_max_fe_corner=vm_corner, f1=float(z["freq"][0]),
                life=life)
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)
    return Graph(x=t(x), edge_index=t(ei, torch.long), edge_attr=t(edge_attr), y=t(y), g=t(g),
                 params=t(params), region=t(flags[:, 3] > 0, torch.bool), pos=t(pos), meta=meta)


def load_dataset(raw_dir: str | Path, cache: str | Path | None = None) -> list[Graph]:
    if cache and Path(cache).exists():
        return torch.load(cache, weights_only=False)
    files = sorted(glob.glob(str(Path(raw_dir) / "*.npz")))
    graphs = [load_graph(f) for f in files]
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, cache)
    return graphs


def split_dataset(graphs: list[Graph], seed: int = 0, n_val: int = 60, n_test: int = 60):
    doe = [g for g in graphs if g.meta["split"] == "doe"]
    ood = [g for g in graphs if g.meta["split"] == "ood"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(doe))
    test = [doe[i] for i in idx[:n_test]]
    val = [doe[i] for i in idx[n_test:n_test + n_val]]
    train = [doe[i] for i in idx[n_test + n_val:]]
    return train, val, test, ood


class Normalizer:
    """Mean/std scaling of node and global targets, fitted on the training graphs."""

    def __init__(self, y_mean=None, y_std=None, g_mean=None, g_std=None):
        self.y_mean, self.y_std, self.g_mean, self.g_std = y_mean, y_std, g_mean, g_std

    @classmethod
    def fit(cls, graphs: list[Graph]) -> "Normalizer":
        y = torch.cat([g.y for g in graphs])
        G = torch.stack([g.g for g in graphs])
        return cls(y.mean(0), y.std(0), G.mean(0), G.std(0))

    def to(self, device):
        return Normalizer(*(t.to(device) for t in (self.y_mean, self.y_std, self.g_mean, self.g_std)))

    def state(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("y_mean", "y_std", "g_mean", "g_std")}

    @classmethod
    def from_state(cls, s: dict) -> "Normalizer":
        return cls(*(torch.tensor(s[k]) for k in ("y_mean", "y_std", "g_mean", "g_std")))


@dataclass
class Batch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    g: torch.Tensor
    params: torch.Tensor
    batch: torch.Tensor        # (N,) graph index of each node
    region: torch.Tensor
    mask: torch.Tensor         # (N,) True for real nodes, False for padding
    n_graphs: int
    n_slots: int               # n_graphs, +1 when padded (padding nodes form a dummy graph)

    def to(self, device):
        return Batch(*(getattr(self, f).to(device) for f in
                       ("x", "edge_index", "edge_attr", "y", "g", "params", "batch", "region", "mask")),
                     n_graphs=self.n_graphs, n_slots=self.n_slots)


def _round_up(n: int, k: int) -> int:
    return int(np.ceil(n / k) * k)


def collate(graphs: list[Graph], pad_nodes: int = 0, pad_edges: int = 0) -> Batch:
    """Concatenate graphs into one disconnected graph.

    With pad_nodes/pad_edges > 0 the node and edge counts are rounded up to multiples of these
    numbers (the MPS backend recompiles its kernels for every new tensor shape, so a small set of
    bucketed shapes is much faster than one shape per batch). Padding nodes belong to a dummy graph
    with index n_graphs and padding edges are self-loops on the first padding node.
    """
    offs = np.cumsum([0] + [g.num_nodes for g in graphs[:-1]])
    n = sum(g.num_nodes for g in graphs)
    x = torch.cat([g.x for g in graphs])
    ei = torch.cat([g.edge_index + int(o) for g, o in zip(graphs, offs)], dim=1)
    ea = torch.cat([g.edge_attr for g in graphs])
    y = torch.cat([g.y for g in graphs])
    batch = torch.cat([torch.full((g.num_nodes,), i, dtype=torch.long) for i, g in enumerate(graphs)])
    region = torch.cat([g.region for g in graphs])
    mask = torch.ones(n, dtype=torch.bool)
    n_slots = len(graphs)
    if pad_nodes:
        n_pad = _round_up(n + 1, pad_nodes) - n
        e = ei.shape[1]
        e_pad = _round_up(e, pad_edges) - e if pad_edges else 0
        x = torch.cat([x, torch.zeros(n_pad, x.shape[1])])
        y = torch.cat([y, torch.zeros(n_pad, y.shape[1])])
        batch = torch.cat([batch, torch.full((n_pad,), len(graphs), dtype=torch.long)])
        region = torch.cat([region, torch.zeros(n_pad, dtype=torch.bool)])
        mask = torch.cat([mask, torch.zeros(n_pad, dtype=torch.bool)])
        ei = torch.cat([ei, torch.full((2, e_pad), n, dtype=torch.long)], dim=1)
        ea = torch.cat([ea, torch.zeros(e_pad, ea.shape[1])])
        n_slots += 1
    params = torch.stack([g.params for g in graphs])
    if n_slots > len(graphs):
        params = torch.cat([params, torch.zeros(1, params.shape[1])])
    return Batch(x=x, edge_index=ei, edge_attr=ea, y=y, g=torch.stack([g.g for g in graphs]),
                 params=params, batch=batch, region=region, mask=mask, n_graphs=len(graphs),
                 n_slots=n_slots)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))
