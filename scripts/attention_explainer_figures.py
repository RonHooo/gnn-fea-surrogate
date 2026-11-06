"""Figures that explain the model: (1) which nodes a GNN can see after 1 and 10 message-passing layers
vs one attention layer, (2) the slices learned by the attention model."""

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnnsurrogate.data import load_dataset, split_dataset, collate, PARAM_NAMES
from gnnsurrogate.evaluate import load_model
from gnnsurrogate.train import pick_device, PAD

pv.OFF_SCREEN = True
OUT = ROOT / "results/attention"
OUT.mkdir(parents=True, exist_ok=True)


def hops_from(src: int, edge_index: np.ndarray, n: int) -> np.ndarray:
    nbrs = [[] for _ in range(n)]
    for a, b in edge_index.T:
        nbrs[a].append(b)
    d = np.full(n, -1)
    d[src] = 0
    q = deque([src])
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if d[j] < 0:
                d[j] = d[i] + 1
                q.append(j)
    return d


def surface(name):
    z = np.load(ROOT / "data/coarse" / f"{name}.npz")
    cells = np.hstack([np.full((len(z["tets"]), 1), 4), z["tets"]]).ravel()
    return pv.UnstructuredGrid(cells, np.full(len(z["tets"]), 10, dtype=np.uint8), z["pos"].astype(float))


def view(pl, pos):
    """Camera looking at the inner face of the upright (mounting hole and gussets visible)."""
    pl.set_background("white")
    c = pos.mean(0)
    pl.camera.focal_point = c
    pl.camera.position = c + np.array([230.0, -170.0, 140.0])
    pl.camera.up = (0, 0, 1)


def receptive_field(g, info):
    ei = g.edge_index.numpy()
    n = g.num_nodes
    flags = g.x[:, 3:6].numpy()
    pos = g.pos.numpy()
    mount = np.where(flags[:, 1] > 0)[0]
    src = int(mount[np.argmin(np.linalg.norm(pos[mount] - pos[mount].mean(0), axis=1))])
    d = hops_from(src, ei, n)
    fixed = np.where(flags[:, 0] > 0)[0]
    info.update(design=g.meta["name"], n_nodes=n, start_node=src, start_pos_mm=pos[src].round(1).tolist(),
                max_hops=int(d.max()), hops_to_nearest_bolt_hole=int(d[fixed].min()),
                frac_within_1=float((d <= 1).mean()), frac_within_10=float((d <= 10).mean()))
    grid = surface(g.meta["name"])
    pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 700))
    panels = ((d <= 1, f"GNN, 1 layer: {100 * (d <= 1).mean():.1f} % of nodes"),
              (d <= 10, f"GNN, 10 layers: {100 * (d <= 10).mean():.0f} % of nodes"),
              (np.ones(n, bool), "Attention, 1 layer: 100 % of nodes"))
    for i, (reach, title) in enumerate(panels):
        pl.subplot(0, i)
        gr = grid.copy()
        gr.point_data["reached"] = reach.astype(float)
        pl.add_mesh(gr.extract_surface(algorithm="dataset_surface"), scalars="reached", cmap=["#d9d9d9", "#e6550d"],
                    clim=(0, 1), show_scalar_bar=False)
        pl.add_mesh(pv.Sphere(radius=4, center=pos[src]), color="black")
        pl.add_text(title, font_size=12, color="black")
        view(pl, pos)
    pl.add_text("black dot = one node on the mounting hole; orange = nodes it can receive information from",
                position="lower_left", font_size=9, color="black")
    pl.screenshot(OUT / "receptive_field.png")
    pl.close()


def slices(g, run, info):
    device = torch.device("cpu")
    model, _, cfg = load_model(ROOT / "results/runs" / run, device)
    maps = []
    hooks = [blk.attn.register_forward_hook(lambda mod, inp, out: maps.append(mod.slice_weights(inp[0])))
             for blk in model.blocks]
    with torch.no_grad():
        model(collate([g], *PAD))
    for h in hooks:
        h.remove()
    n = g.num_nodes
    w = [m[:n].numpy() for m in maps]
    info["slice_run"] = run
    info["slices"] = [dict(layer=i + 1, mean_max_weight=float(x.max(1).mean()),
                           n_slices_used_as_argmax=int(len(np.unique(x.argmax(1)))),
                           temperature=float(torch.exp(model.blocks[i].attn.log_tau)))
                      for i, x in enumerate(w)]
    grid = surface(g.meta["name"])
    layers = [0, len(w) // 2, len(w) - 1]
    pl = pv.Plotter(shape=(1, len(layers)), off_screen=True, window_size=(700 * len(layers), 700))
    rng = np.random.default_rng(0)
    perm = rng.permutation(w[0].shape[1])            # shuffle slice ids so neighbouring ids get different colours
    # one colour per slice (tab20 alone has only 20 colours, so 2 of the 32 slices would share one)
    SLICE_CMAP = ListedColormap((list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors))[:w[0].shape[1]])
    for i, li in enumerate(layers):
        pl.subplot(0, i)
        gr = grid.copy()
        gr.point_data["slice"] = perm[w[li].argmax(1)].astype(float)
        pl.add_mesh(gr.extract_surface(algorithm="dataset_surface"), scalars="slice", cmap=SLICE_CMAP,
                    clim=(0, w[li].shape[1] - 1), show_scalar_bar=False, interpolate_before_map=False)
        s = info["slices"][li]
        pl.add_text(f"layer {li + 1}: most likely slice of each node\n"
                    f"{s['n_slices_used_as_argmax']} of {w[li].shape[1]} slices used, "
                    f"mean top weight {s['mean_max_weight']:.2f}", font_size=11, color="black")
        view(pl, g.pos.numpy())
        pl.camera.zoom(0.85)                          # leave room for the text at the top
    pl.screenshot(OUT / "learned_slices.png")
    pl.close()


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "transformer_s0"
    graphs = load_dataset(ROOT / "data/coarse", ROOT / "data/graphs_coarse.pt")
    _, _, test, _ = split_dataset(graphs)
    g = test[0]
    info = dict(params=dict(zip(PARAM_NAMES, g.meta["params_raw"])))
    receptive_field(g, info)
    slices(g, run, info)
    (OUT / "explainer_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
