"""Figures for the README: parity plots, model comparison, field renders, training curves, screening."""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnnsurrogate.data import load_dataset, split_dataset
from gnnsurrogate.evaluate import load_model, predict
from gnnsurrogate.train import pick_device

pv.OFF_SCREEN = True
RUNS = ROOT / "results/runs"
FINAL = "gnn_stress" if (RUNS / "gnn_stress" / "summary.json").exists() else "gnn"
FIG = ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)
VARIANTS = {"gnn_stress": "GNN, stress-component targets", "gnn": "GNN (params + global node)",
            "gnn_noglobal": "GNN, no global node", "gnn_noparams": "GNN, mesh only (no params)",
            "gnn_noparams_noglobal": "GNN, mesh only, no global node",
            "pointmlp": "point-wise MLP (no edges)"}


def parity():
    g = pd.read_csv(RUNS / FINAL / "per_graph_metrics.csv")
    gp = pd.read_csv(RUNS / "gp/per_graph_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (df, name) in enumerate(((g, "GNN"), (gp, "GP on parameters"))):
        for col, (key, lab, log) in enumerate((("f1", "f1 (Hz)", False), ("vm_max", "peak von Mises (MPa)", True),
                                               ("life", "fatigue life (h)", True))):
            a = axes[row, col]
            for s, mk, c in (("test", "o", "C0"), ("ood", "^", "C3"), ("shift", "s", "C2")):
                d = df[df.set == s]
                if d.empty:
                    continue
                a.scatter(d[f"{key}_true"], d[f"{key}_pred"], marker=mk, s=18, c=c, alpha=0.7,
                          label=f"{s} (n={len(d)})")
            lo = min(df[f"{key}_true"].min(), df[f"{key}_pred"].min())
            hi = max(df[f"{key}_true"].max(), df[f"{key}_pred"].max())
            a.plot([lo, hi], [lo, hi], "k--", lw=1)
            if log:
                a.set_xscale("log")
                a.set_yscale("log")
                for ax_ in (a.xaxis, a.yaxis):
                    ax_.set_minor_formatter(matplotlib.ticker.NullFormatter())
            a.set_xlabel(f"FE {lab}")
            a.set_ylabel(f"{name} {lab}")
            a.grid(True, which="both", alpha=0.3)
            if col == 0:
                a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "parity_gnn_vs_gp.png", dpi=150)
    plt.close(fig)


def comparison():
    rows = []
    for k, lab in VARIANTS.items():
        f = RUNS / k / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        for split in ("test", "ood", "shift"):
            if split in s:
                rows.append(dict(model=lab, split=split, u=s[split]["u_rel_l2"]["mean"],
                                 vm=s[split]["vm_rel_l2"]["mean"]))
    df = pd.DataFrame(rows)
    df.to_csv(FIG / "model_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for a, (col, lab) in zip(axes, (("u", "displacement field"), ("vm", "von Mises field"))):
        piv = df.pivot(index="model", columns="split", values=col).reindex(
            [v for v in VARIANTS.values() if v in set(df.model)])
        y = np.arange(len(piv))
        cols = [c for c in ("test", "ood", "shift") if c in piv]
        w = 0.8 / len(cols)
        for i, c in enumerate(cols):
            a.barh(y - 0.4 + w * (i + 0.5), 100 * piv[c], w, label=c)
        a.set_yticks(y, piv.index)
        a.set_xlabel(f"mean relative L2 error, {lab} (%)")
        a.grid(True, axis="x", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=150)
    plt.close(fig)


def render_fields(n_examples=2):
    device = pick_device("auto")
    graphs = load_dataset(ROOT / "data/coarse", ROOT / "data/graphs_coarse.pt")
    _, _, test, _ = split_dataset(graphs)
    model, norm, cfg = load_model(RUNS / FINAL, device)
    m = pd.read_csv(RUNS / FINAL / "per_graph_metrics.csv")
    m = m[m.set == "test"].sort_values("vm_rel_l2").reset_index(drop=True)
    picks = {"median": m.iloc[len(m) // 2]["name"], "worst": m.iloc[-1]["name"]}
    by_name = {g.meta["name"]: g for g in test}
    for tag, name in picks.items():
        g = by_name[name]
        yp, _ = predict(model, norm, [g], device, cfg.get("targets", "vm"))[0]
        z = np.load(ROOT / "data/coarse" / f"{name}.npz")
        cells = np.hstack([np.full((len(z["tets"]), 1), 4), z["tets"]]).ravel()
        grid = pv.UnstructuredGrid(cells, np.full(len(z["tets"]), 10, dtype=np.uint8), z["pos"].astype(float))
        true, pred = g.y[:, 3].numpy(), yp[:, 3]
        clim = (0, float(np.percentile(true, 99.5)))
        pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 700))
        panels = ((true, "FE (CalculiX), von Mises", "turbo", clim, "FE von Mises (MPa)"),
                  (pred, "GNN prediction, von Mises", "turbo", clim, "GNN von Mises (MPa)"),
                  (np.abs(pred - true), "|GNN - FE|", "magma", (0, clim[1] / 4), "abs. error (MPa)"))
        for i, (vals, title, cmap, cl, bar) in enumerate(panels):
            pl.subplot(0, i)
            gr = grid.copy()
            gr.point_data[bar] = vals
            pl.set_background("white")
            pl.add_mesh(gr.extract_surface(algorithm="dataset_surface"), scalars=bar, cmap=cmap, clim=cl,
                        scalar_bar_args=dict(title=bar, color="black", fmt="%.0f", position_x=0.2,
                                             position_y=0.05, width=0.6, height=0.06))
            pl.add_text(title, font_size=12, color="black")
            pl.view_isometric()
            pl.camera.azimuth = 215
            pl.camera.elevation = 20
        row = m[m.name == name].iloc[0]
        pl.add_text(f"{name}: relative L2 error {100 * row.vm_rel_l2:.1f} %", position="lower_left",
                    font_size=10, color="black")
        pl.screenshot(FIG / f"field_{tag}.png")
        pl.close()


def curves():
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, lab in VARIANTS.items():
        f = RUNS / k / "history.csv"
        if f.exists():
            h = pd.read_csv(f)
            ax.plot(h.epoch, h.val_node, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss, node fields (normalised MSE)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=150)
    plt.close(fig)


def screening():
    f = ROOT / "results/screening/candidates.csv"
    if not f.exists():
        return
    c = pd.read_csv(f)
    p = pd.read_csv(ROOT / "results/screening/fe_checked_picks.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a, s, lab in ((axes[0], "gnn", "GNN"), (axes[1], "gp", "GP on parameters")):
        a.scatter(c.mass, c[f"{s}_life"], s=5, alpha=0.3, c="gray", label="candidates (predicted)")
        q = p[p.proposed_by == s]
        a.scatter(q.mass, q.fe_life, s=50, marker="o", facecolors="none",
                  edgecolors=np.where(q.fe_feasible, "green", "red"), linewidths=2,
                  label="proposed, FE-checked (green = feasible)")
        a.axhline(1500, color="k", ls="--", lw=1)
        a.set_yscale("log")
        a.set_xlabel("bracket mass (kg)")
        a.set_title(lab)
        a.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("fatigue life (h)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "screening.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (parity, comparison, curves, screening, render_fields):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:
            print("skipped", fn.__name__, type(e).__name__, e)
