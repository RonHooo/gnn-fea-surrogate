"""Surrogate-assisted design screening.

1. Sample new candidate designs (LHS, different seed from the training DOE, same ranges).
2. Mesh each candidate (coarse linear mesh only, no FE) and predict f1, fatigue life and peak stress
   with the trained GNN; the mass comes exactly from the CAD volume. The parameter-only Gaussian
   process makes the same predictions for comparison.
3. Each surrogate proposes its k lightest designs that it predicts to be feasible (with the same life
   margin as the DOE study). Those designs are checked with the full CalculiX analysis.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "bracket-fea"))
sys.path.insert(0, str(ROOT.parent / "bracket-fea" / "studies"))

from gnnsurrogate.data import load_dataset, split_dataset, graph_from_mesh, PARAM_NAMES
from gnnsurrogate.evaluate import load_model, predict
from gnnsurrogate.train import pick_device
from gnnsurrogate import baselines

COARSE = dict(h_max=4.0, h_min=2.5, h_hole=3.0, dist_max=6.0, order=1)
F1_MIN, LIFE_MIN, LIFE_MARGIN, STRESS_ALLOW = 100.0, 1000.0, 1.5, 140.0


def mesh_candidate(params: dict):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from bracketfea.geometry import BracketParams, MeshSettings, mesh_bracket
    from bracketfea.analysis import design_region
    t0 = time.perf_counter()
    p = BracketParams(**params)
    m = mesh_bracket(p, MeshSettings(**COARSE))
    flags = np.zeros((m.n_nodes, 4), dtype=np.uint8)
    flags[m.sets["FIX"], 0] = 1
    flags[m.sets["MOUNT"], 1] = 1
    flags[m.surface_nodes(), 2] = 1
    flags[design_region(m.nodes, p), 3] = 1
    mass = m.volume * 7.85e-9 * 1e3
    return m.nodes, m.tets, flags, mass, time.perf_counter() - t0


def fe_check(params: dict):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from bracketfea.analysis import run_design
    from bracketfea.geometry import BracketParams
    t0 = time.perf_counter()
    r = run_design(BracketParams(**params), threads=1)
    s = r.summary()
    return dict(fe_mass=s["mass_kg"], fe_f1=s["f1"], fe_life=s["life_h"], fe_vm_max=s["vm_max"],
                fe_time=time.perf_counter() - t0)


def predicted_feasible(df, prefix):
    return ((df[f"{prefix}_f1"] >= F1_MIN) & (df[f"{prefix}_life"] >= LIFE_MARGIN * LIFE_MIN)
            & (df[f"{prefix}_vm_max"] <= STRESS_ALLOW))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/runs/gnn")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/screening")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import doe as doe_mod

    designs = doe_mod.sample_designs(args.n, args.seed)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        meshes = list(ex.map(mesh_candidate, designs, chunksize=4))
    t_mesh_wall = time.perf_counter() - t0
    graphs = [graph_from_mesh(n, t, f, [d[k] for k in PARAM_NAMES], name=f"cand_{i:04d}")
              for i, (d, (n, t, f, _, _)) in enumerate(zip(designs, meshes))]

    device = pick_device("auto")
    model, norm, cfg = load_model(args.run, device)
    t0 = time.perf_counter()
    preds = predict(model, norm, graphs, device, cfg.get("targets", "vm"))
    if device.type == "mps":
        torch.mps.synchronize()
    t_gnn = time.perf_counter() - t0

    data = load_dataset(ROOT / "data/coarse", ROOT / "data/graphs_coarse.pt")
    train, val, test, ood = split_dataset(data)
    Xtr, Ytr = baselines.features(train + val), baselines.targets(train + val)
    Xc = baselines.features(graphs)
    gp_pred = np.stack([baselines.fit_gp(Xtr, Ytr[:, k]).predict(Xc) for k in range(3)], axis=1)

    df = pd.DataFrame(designs)
    df["mass"] = [m[3] for m in meshes]
    g = np.stack([p[1] for p in preds])
    df["gnn_f1"], df["gnn_life"], df["gnn_vm_max"] = np.exp(g[:, 0]), 10 ** g[:, 1], np.exp(g[:, 2])
    df["gp_f1"], df["gp_life"], df["gp_vm_max"] = np.exp(gp_pred[:, 0]), 10 ** gp_pred[:, 1], np.exp(gp_pred[:, 2])
    for s in ("gnn", "gp"):
        df[f"{s}_feasible"] = predicted_feasible(df, s)

    picks = []
    for s in ("gnn", "gp"):
        top = df[df[f"{s}_feasible"]].sort_values("mass").head(args.k).copy()
        top["proposed_by"] = s
        picks.append(top)
    picks = pd.concat(picks)
    uniq = picks.drop_duplicates(subset=PARAM_NAMES)
    with ProcessPoolExecutor(max_workers=min(args.workers, 5)) as ex:
        checks = list(ex.map(fe_check, [{k: float(r[k]) for k in PARAM_NAMES} for _, r in uniq.iterrows()]))
    chk = pd.concat([uniq[PARAM_NAMES].reset_index(drop=True), pd.DataFrame(checks)], axis=1)
    picks = picks.merge(chk, on=PARAM_NAMES, how="left")
    picks["fe_feasible"] = ((picks.fe_f1 >= F1_MIN) & (picks.fe_life >= LIFE_MARGIN * LIFE_MIN)
                            & (picks.fe_vm_max <= STRESS_ALLOW))
    df.to_csv(out / "candidates.csv", index=False)
    picks.to_csv(out / "fe_checked_picks.csv", index=False)

    summary = dict(n_candidates=args.n, mesh_wall_s=t_mesh_wall, mesh_cpu_s=float(sum(m[4] for m in meshes)),
                   gnn_wall_s=t_gnn, fe_time_per_design_s=float(np.mean([c["fe_time"] for c in checks])),
                   n_pred_feasible={s: int(df[f"{s}_feasible"].sum()) for s in ("gnn", "gp")})
    for s in ("gnn", "gp"):
        p = picks[picks.proposed_by == s]
        ok = p[p.fe_feasible]
        summary[s] = dict(proposed=len(p), truly_feasible=int(p.fe_feasible.sum()),
                          best_verified_mass=float(ok.mass.min()) if len(ok) else None,
                          best_verified=ok.sort_values("mass").iloc[0][PARAM_NAMES + ["mass", "fe_f1", "fe_life"]].to_dict()
                          if len(ok) else None)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
