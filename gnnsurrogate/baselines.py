"""Parameter-only surrogates for the scalar outputs (no mesh): Gaussian process regression.

Inputs are the 6 design variables (scaled to [0, 1] on the training ranges) plus two 0/1 flags
for "has gussets" and "has lightening hole", because L_g = 0 / d_h = 0 are on/off switches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from .data import load_dataset, split_dataset, GLOBAL_TARGETS
from .evaluate import r2


def features(graphs) -> np.ndarray:
    p = np.stack([g.params.numpy() for g in graphs])
    raw = np.array([g.meta["params_raw"] for g in graphs])
    flags = np.stack([(raw[:, 3] > 0), (raw[:, 5] > 0)], axis=1).astype(float)
    return np.concatenate([p, flags], axis=1)


def targets(graphs) -> np.ndarray:
    return np.stack([g.g.numpy() for g in graphs])


def fit_gp(X, y, seed=0):
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=3, random_state=seed)
    return gp.fit(X, y)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/coarse")
    ap.add_argument("--cache", default="data/graphs_coarse.pt")
    ap.add_argument("--out", default="results/runs/gp")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graphs = load_dataset(args.raw, args.cache)
    train, val, test, ood = split_dataset(graphs)
    shift = [g for g in graphs if g.meta["split"] == "shift"]
    sets = [("test", test), ("ood", ood)] + ([("shift", shift)] if shift else [])
    Xtr, Ytr = features(train + val), targets(train + val)
    rows, summary = [], {}
    preds = {}
    for k, name in enumerate(GLOBAL_TARGETS):
        gp = fit_gp(Xtr, Ytr[:, k])
        for set_name, gs in sets:
            p = gp.predict(features(gs))
            preds[(set_name, name)] = p
    for set_name, gs in sets:
        T = targets(gs)
        lf, ll, lv = (preds[(set_name, n)] for n in GLOBAL_TARGETS)
        df = pd.DataFrame(dict(name=[g.meta["name"] for g in gs], set=set_name,
                               f1_true=np.exp(T[:, 0]), f1_pred=np.exp(lf),
                               life_true=10 ** T[:, 1], life_pred=10 ** ll,
                               vm_max_true=np.exp(T[:, 2]), vm_max_pred=np.exp(lv)))
        df["f1_err"] = df.f1_pred / df.f1_true - 1
        df["vm_max_err"] = df.vm_max_pred / df.vm_max_true - 1
        df["log10_life_err"] = ll - T[:, 1]
        rows.append(df)
        summary[set_name] = dict(
            f1_err_mean_abs=df.f1_err.abs().mean(), vm_max_err_mean_abs=df.vm_max_err.abs().mean(),
            log10_life_err_mean_abs=df.log10_life_err.abs().mean(),
            r2_f1=r2(lf, T[:, 0]), r2_vm_max=r2(lv, T[:, 2]), r2_log10_life=r2(ll, T[:, 1]), n=len(df))
    pd.concat(rows).to_csv(out / "per_graph_metrics.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
