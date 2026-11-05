"""Attention vs GNN vs GP: seed-averaged tables and figures for the README section.

Reads the summary.json / per_graph_metrics.csv files written by gnnsurrogate.evaluate and
gnnsurrogate.baselines; all metrics come from there. Writes to results/attention/.
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "results/runs"
OUT = ROOT / "results/attention"
OUT.mkdir(parents=True, exist_ok=True)

S = lambda name: [f"{name}_s{k}" for k in range(3)]
CHOICE = json.loads((OUT / "lr_choice.json").read_text()) if (OUT / "lr_choice.json").exists() else {}
GNN0 = CHOICE.get("gnn_seed0_run", "gnn_stress")       # GNN seed 0 at the learning rate chosen on val
GNN_RUNS = [GNN0, "gnn_stress_s1", "gnn_stress_s2"]
GROUPS = [
    ("gnn", f"GNN, stress targets (final), lr {CHOICE.get('gnn_lr', '1e-3')}", GNN_RUNS),
    ("gnn_readme", "GNN in the README (lr 1e-3, seed 0)", ["gnn_stress"]),
    ("a0", "Attention A0 (32 slices)", S("transformer")),
    ("a1_16", "A1: 16 slices", S("transformer_m16")),
    ("a1_64", "A1: 64 slices", S("transformer_m64")),
    ("a2", "A2: no Fourier features", S("transformer_nofourier")),
    ("a3", "A3: mesh only (no design variables)", S("transformer_noparams")),
    ("a4", "A4: + local mesh-neighbour step", S("transformer_local")),
    ("long", "A0, same wall-clock as the GNN", ["transformer_long_s0"]),
    ("gnn_mo", "GNN, mesh only", S("gnn_stress_noparams")),
    ("gnn_mo_ng", "GNN, mesh only, no global node", S("gnn_stress_noparams_noglobal")),
    ("pointmlp", "point-wise MLP, no edges (von Mises target)", ["pointmlp"]),
]
# (column in summary.json, statistic, label, scale)
METRICS = [("u_rel_l2", "mean", "u field", 100), ("vm_rel_l2", "mean", "von Mises field", 100),
           ("f1_err", "mean_abs", "f1", 100), ("vm_max_err", "mean_abs", "peak von Mises", 100),
           ("log10_life_err", "mean_abs", "log10 life", 1)]
SETS = ("test", "ood", "shift", "val")


def collect() -> pd.DataFrame:
    rows = []
    for key, label, runs in GROUPS:
        for run in runs:
            f = RUNS / run / "summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            for split in SETS:
                for col, stat, mlab, scale in METRICS:
                    if split in s and col in s[split]:
                        rows.append(dict(group=key, label=label, run=run, set=split, metric=mlab,
                                         value=scale * s[split][col][stat]))
    gp = json.loads((RUNS / "gp/summary.json").read_text())
    for split in ("test", "ood", "shift"):
        for col, mlab, scale in (("f1_err_mean_abs", "f1", 100), ("vm_max_err_mean_abs", "peak von Mises", 100),
                                 ("log10_life_err_mean_abs", "log10 life", 1)):
            rows.append(dict(group="gp", label="GP on the 6 design variables", run="gp", set=split,
                             metric=mlab, value=scale * gp[split][col]))
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["group", "label", "set", "metric"], sort=False)["value"]
    a = g.agg(mean="mean", std=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, n="count").reset_index()
    a["values"] = g.apply(lambda v: ";".join(f"{x:.4g}" for x in v)).values
    return a


def fmt(row, digits):
    if row is None:
        return "-"
    m = f"{row['mean']:.{digits}f}"
    if row["n"] > 1:
        return f"{m} ± {row['std']:.{digits}f}"
    return m + " (1 seed)" if row["group"] != "gp" else m


def table(a: pd.DataFrame, groups, split="test", with_shift=True) -> str:
    look = {(r.group, r.set, r.metric): r for r in a.itertuples()}
    digits = {"u field": 1, "von Mises field": 1, "f1": 2, "peak von Mises": 1, "log10 life": 3}
    unit = {"log10 life": ""}
    head = ["model", "seeds"] + [m[2] for m in METRICS] + (["von Mises field, shifted family"] if with_shift else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    labels = dict((k, l) for k, l, _ in GROUPS) | {"gp": "GP on the 6 design variables"}
    for gkey in groups:
        cells, n = [], "-"
        for _, _, mlab, _ in METRICS:
            r = look.get((gkey, split, mlab))
            if r is None:
                cells.append("-")
                continue
            r = r._asdict()
            n = str(r["n"]) if gkey != "gp" else "-"
            u = unit.get(mlab, " %")
            cells.append(fmt(r, digits[mlab]) + u)
        if with_shift:
            r = look.get((gkey, "shift", "von Mises field"))
            cells.append(fmt(r._asdict(), 1) + " %" if r is not None else "-")
        if all(c == "-" for c in cells):
            continue
        lines.append(f"| {labels[gkey]} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def extrapolation_table(a: pd.DataFrame) -> str:
    look = {(r.group, r.set, r.metric): r._asdict() for r in a.itertuples()}
    lines = ["| | GNN (final) | Attention A0 | Attention A2 (no Fourier) | GP |", "|---|---|---|---|---|"]
    for split, name in (("ood", "OOD designs (40)"), ("shift", "Shifted family (30)")):
        cells = []
        for gk in ("gnn", "a0", "a2", "gp"):
            parts = []
            for mlab, d, u in (("f1", 2, " %"), ("peak von Mises", 1, " %"), ("log10 life", 3, "")):
                r = look.get((gk, split, mlab))
                parts.append(fmt(r, d) + u if r else "-")
            cells.append(" / ".join(parts))
        lines.append(f"| {name}: f1 / peak stress / log10 life | " + " | ".join(cells) + " |")
    cells = []
    for gk in ("gnn", "a0", "a2", "gp"):
        parts = [fmt(look[(gk, "shift", m)], 1) + " %" if (gk, "shift", m) in look else "-"
                 for m in ("u field", "von Mises field")]
        cells.append(" / ".join(parts))
    lines.append("| Shifted family: displacement / von Mises field | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_graph(run):
    f = RUNS / run / "per_graph_metrics.csv"
    return pd.read_csv(f) if f.exists() else None


def parity():
    sets = (("test", "o", "C0"), ("ood", "^", "C3"), ("shift", "s", "C2"))
    models = [(GNN0, "GNN (seed 0)"), ("transformer_s0", "Attention A0 (seed 0)"), ("gp", "GP")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for row, (run, name) in enumerate(models):
        df = per_graph(run)
        for col, (key, lab, log) in enumerate((("f1", "f1 (Hz)", False), ("vm_max", "peak von Mises (MPa)", True),
                                               ("life", "fatigue life (h)", True))):
            a = axes[row, col]
            for s, mk, c in sets:
                d = df[df.set == s]
                a.scatter(d[f"{key}_true"], d[f"{key}_pred"], marker=mk, s=16, c=c, alpha=0.7,
                          label=f"{s} (n={len(d)})")
            d = df[df.set.isin([s for s, _, _ in sets])]
            lo = min(d[f"{key}_true"].min(), d[f"{key}_pred"].min())
            hi = max(d[f"{key}_true"].max(), d[f"{key}_pred"].max())
            a.plot([lo, hi], [lo, hi], "k--", lw=1)
            if log:
                a.set_xscale("log")
                a.set_yscale("log")
                for ax_ in (a.xaxis, a.yaxis):
                    ax_.set_minor_formatter(matplotlib.ticker.NullFormatter())
            a.set_xlabel(f"FE {lab}")
            a.set_ylabel(f"{name}: {lab}")
            a.grid(True, which="both", alpha=0.3)
            if col == 0:
                a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "parity_gnn_attention_gp.png", dpi=130)
    plt.close(fig)


def shift_distribution():
    """Per-design errors on the shifted family (each design averaged over the 3 seeds)."""
    models = [("GNN", GNN_RUNS), ("Attention A0", S("transformer")),
              ("A2: no Fourier", S("transformer_nofourier")), ("A4: + local step", S("transformer_local"))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rng = np.random.default_rng(0)
    for a, (col, lab) in zip(axes, (("u_rel_l2", "displacement field"), ("vm_rel_l2", "von Mises field"))):
        data, names = [], []
        for name, runs in models:
            dfs = [d for d in (per_graph(r) for r in runs) if d is not None]
            if not dfs:
                continue
            for split, tag in (("test", "test"), ("shift", "shifted")):
                v = pd.concat([d[d.set == split].set_index("name")[col] for d in dfs], axis=1).mean(axis=1)
                data.append(100 * v.values)
                names.append(f"{name}\n{tag}")
        a.boxplot(data, showfliers=False, widths=0.5)
        for i, v in enumerate(data):
            a.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(v)), v, s=8, alpha=0.6,
                      c="C2" if "shifted" in names[i] else "C0")
        a.set_xticks(range(1, len(names) + 1), names, fontsize=7)
        a.set_ylabel(f"relative L2 error, {lab} (%)")
        a.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-design error, test (blue) vs shifted family (green); each point = one design, mean of 3 seeds")
    fig.tight_layout()
    fig.savefig(OUT / "shift_error_distribution.png", dpi=130)
    plt.close(fig)


def ablation_bars(a: pd.DataFrame):
    keys = ["gnn", "a0", "a1_16", "a1_64", "a2", "a3", "a4", "gnn_mo", "gnn_mo_ng"]
    panels = [("test", "u field"), ("test", "von Mises field"), ("shift", "von Mises field")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    labels = dict((k, l) for k, l, _ in GROUPS)
    keys = [k for k in keys if ((a.group == k) & (a.set == "test")).any()]
    y = np.arange(len(keys))
    for ax, (split, metric) in zip(axes, panels):
        d = a[(a.set == split) & (a.metric == metric)].set_index("group").reindex(keys)
        ax.barh(y, d["mean"], xerr=d["std"].fillna(0), color=["C1" if k.startswith("gnn") else "C0" for k in keys],
                capsize=3)
        ax.set_title(f"{metric}, {split} set")
        ax.set_xlabel("relative L2 error (%), mean ± std over seeds")
        ax.set_xscale("log")
        ax.grid(True, axis="x", which="both", alpha=0.3)
    axes[0].set_yticks(y, [labels[k] for k in keys], fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png", dpi=130)
    plt.close(fig)


def curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run, lab, c in ((GNN0, "GNN (seed 0)", "C1"), ("transformer_s0", "Attention A0 (seed 0)", "C0"),
                        ("transformer_long_s0", "Attention, same wall-clock", "C2")):
        f = RUNS / run / "history.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        axes[0].plot(h.epoch, h.val, c=c, label=lab)
        axes[1].plot(60 * h.hours, h.val, c=c, label=lab)
    for ax, xl in zip(axes, ("epoch", "training time on the M3 GPU (min)")):
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("validation loss (normalised MSE)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=130)
    plt.close(fig)


def render_median():
    """FE vs GNN vs attention von Mises on the test design with the median attention error."""
    import pyvista as pv
    from gnnsurrogate.data import load_dataset, split_dataset
    from gnnsurrogate.evaluate import load_model, predict
    from gnnsurrogate.train import pick_device
    pv.OFF_SCREEN = True
    device = pick_device("auto")
    graphs = load_dataset(ROOT / "data/coarse", ROOT / "data/graphs_coarse.pt")
    _, _, test, _ = split_dataset(graphs)
    m = per_graph("transformer_s0")
    m = m[m.set == "test"].sort_values("vm_rel_l2").reset_index(drop=True)
    name = m.iloc[len(m) // 2]["name"]
    g = {x.meta["name"]: x for x in test}[name]
    preds, errs = {}, {}
    for run, lab in ((GNN0, "GNN"), ("transformer_s0", "Attention")):
        model, norm, cfg = load_model(RUNS / run, device)
        preds[lab] = predict(model, norm, [g], device, cfg.get("targets", "vm"))[0][0][:, 3]
        r = per_graph(run)
        errs[lab] = float(r[(r.set == "test") & (r.name == name)].vm_rel_l2.iloc[0])
    z = np.load(ROOT / "data/coarse" / f"{name}.npz")
    cells = np.hstack([np.full((len(z["tets"]), 1), 4), z["tets"]]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(z["tets"]), 10, dtype=np.uint8), z["pos"].astype(float))
    true = g.y[:, 3].numpy()
    clim = (0, float(np.percentile(true, 99.5)))
    pl = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(2100, 1400))
    for row, lab in enumerate(("GNN", "Attention")):
        pred = preds[lab]
        panels = ((true, "FE (CalculiX), von Mises", "turbo", clim, "FE von Mises (MPa)"),
                  (pred, f"{lab} prediction (seed 0)", "turbo", clim, f"{lab} von Mises (MPa)"),
                  (np.abs(pred - true), f"|{lab} - FE|, rel. L2 error {100 * errs[lab]:.1f} %", "magma",
                   (0, clim[1] / 4), f"{lab} abs. error (MPa)"))
        for col, (vals, title, cmap, cl, bar) in enumerate(panels):
            pl.subplot(row, col)
            gr = grid.copy()
            gr.point_data[bar] = vals
            pl.set_background("white")
            pl.add_mesh(gr.extract_surface(algorithm="dataset_surface"), scalars=bar, cmap=cmap, clim=cl,
                        scalar_bar_args=dict(title=bar, color="black", fmt="%.0f", position_x=0.2,
                                             position_y=0.05, width=0.6, height=0.06))
            pl.add_text(title, font_size=11, color="black")
            pl.view_isometric()
            pl.camera.azimuth = 215
            pl.camera.elevation = 20
    pl.screenshot(OUT / "field_median_gnn_vs_attention.png")
    pl.close()
    return name


def main():
    df = collect()
    a = aggregate(df)
    df.to_csv(OUT / "all_runs.csv", index=False)
    a.to_csv(OUT / "summary_by_model.csv", index=False)
    main_keys = ["gnn", "gnn_readme", "a0", "long", "gp", "pointmlp"]
    abl_keys = ["a0", "a1_16", "a1_64", "a2", "a3", "a4", "gnn", "gnn_mo", "gnn_mo_ng"]
    text = ["## Test set (60 designs)", table(a, main_keys), "",
            "## Ablations (test set)", table(a, abl_keys), "",
            "## Extrapolation", extrapolation_table(a), "",
            "## Validation set (used for model selection)", table(a, ["gnn", "a0", "a1_16", "a1_64", "a2", "a3", "a4"],
                                                              split="val", with_shift=False)]
    (OUT / "tables.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    for fn in (parity, shift_distribution, lambda: ablation_bars(a), curves, render_median):
        try:
            r = fn()
            print("ok", getattr(fn, "__name__", "fig"), r or "")
        except Exception as e:
            print("skipped", getattr(fn, "__name__", "fig"), type(e).__name__, e)


if __name__ == "__main__":
    main()
