"""Evaluate a trained model on the test and OOD sets: field errors, peak stress, f1, fatigue life, speed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_dataset, split_dataset, Normalizer, collate, TARGET_COLS
from .train import build_model, pick_device, PAD


def load_model(run_dir: str | Path, device):
    ck = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"])
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    return model, Normalizer.from_state(ck["norm"]), ck["cfg"]


def _von_mises(s):
    sxx, syy, szz, sxy, syz, szx = (s[:, k] for k in range(6))
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


@torch.no_grad()
def predict(model, norm: Normalizer, graphs, device, targets: str = "vm"):
    """Returns per-graph (node predictions [u_x, u_y, u_z, von Mises] in physical units,
    global predictions or None). With targets="stress" von Mises comes from the predicted components."""
    normd = norm.to(device)
    cols = TARGET_COLS[targets]
    out = []
    for g in graphs:
        b = collate([g], *PAD).to(device)
        pn, pg = model(b)
        yn = (pn[:g.num_nodes] * normd.y_std[cols] + normd.y_mean[cols]).cpu().numpy()
        if targets == "stress":
            yn = np.concatenate([yn[:, :3], _von_mises(yn[:, 3:])[:, None]], axis=1)
        gg = None if pg is None else (pg * normd.g_std + normd.g_mean)[0].cpu().numpy()
        out.append((yn, gg))
    return out


@torch.no_grad()
def time_inference(model, graphs, device, n: int = 20) -> float:
    bs = [collate([g], *PAD).to(device) for g in graphs[:n]]
    model(bs[0])
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for b in bs:
        model(b)
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / len(bs)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def graph_metrics(g, yp, gp) -> dict:
    y = g.y.numpy()
    reg = g.region.numpy()
    pos = g.pos.numpy()
    vm_t, vm_p = y[:, 3], yp[:, 3]
    it = np.where(reg)[0][np.argmax(vm_t[reg])]
    ip = np.where(reg)[0][np.argmax(vm_p[reg])]
    m = dict(name=g.meta["name"], split=g.meta["split"],
             u_rel_l2=rel_l2(yp[:, :3], y[:, :3]),
             vm_rel_l2=rel_l2(vm_p, vm_t),
             vm_rel_l2_region=rel_l2(vm_p[reg], vm_t[reg]),
             vm_max_true_corner=float(vm_t[it]), vm_max_field_pred=float(vm_p[ip]),
             vm_max_field_err=float((vm_p[ip] - vm_t[it]) / vm_t[it]),
             hot_spot_dist_mm=float(np.linalg.norm(pos[ip] - pos[it])),
             u_max_true=float(np.linalg.norm(y[:, :3], axis=1).max()),
             u_max_err=float((np.linalg.norm(yp[:, :3], axis=1).max() - np.linalg.norm(y[:, :3], axis=1).max())
                             / np.linalg.norm(y[:, :3], axis=1).max()))
    gt = g.g.numpy()
    m.update(f1_true=float(np.exp(gt[0])), life_true=float(10 ** gt[1]), vm_max_true=float(np.exp(gt[2])))
    if gp is not None:
        m.update(f1_pred=float(np.exp(gp[0])), life_pred=float(10 ** gp[1]), vm_max_pred=float(np.exp(gp[2])),
                 f1_err=float(np.exp(gp[0] - gt[0]) - 1), vm_max_err=float(np.exp(gp[2] - gt[2]) - 1),
                 log10_life_err=float(gp[1] - gt[1]))
    return m


def r2(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    return float(1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))


def summarize(df: pd.DataFrame) -> dict:
    s = {}
    for col in ("u_rel_l2", "vm_rel_l2", "vm_rel_l2_region", "hot_spot_dist_mm"):
        s[col] = dict(mean=df[col].mean(), median=df[col].median(), p90=df[col].quantile(0.9))
    for col in ("vm_max_field_err", "u_max_err", "f1_err", "vm_max_err", "log10_life_err"):
        if col in df:
            s[col] = dict(mean_abs=df[col].abs().mean(), median_abs=df[col].abs().median(),
                          p90_abs=df[col].abs().quantile(0.9), bias=df[col].mean())
    if "f1_pred" in df:
        s["r2_f1"] = r2(np.log(df.f1_pred), np.log(df.f1_true))
        s["r2_vm_max"] = r2(np.log(df.vm_max_pred), np.log(df.vm_max_true))
        s["r2_log10_life"] = r2(np.log10(df.life_pred), np.log10(df.life_true))
    s["r2_vm_max_field"] = r2(np.log(df.vm_max_field_pred), np.log(df.vm_max_true_corner))
    s["n"] = len(df)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/runs/gnn")
    ap.add_argument("--raw", default="data/coarse")
    ap.add_argument("--cache", default="data/graphs_coarse.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    graphs = load_dataset(args.raw, args.cache)
    train, val, test, ood = split_dataset(graphs)
    shift = [g for g in graphs if g.meta["split"] == "shift"]
    model, norm, cfg = load_model(args.run, device)
    rows = []
    for split_name, gs in (("test", test), ("ood", ood), ("val", val), ("shift", shift)):
        if not gs:
            continue
        for g, (yp, gp) in zip(gs, predict(model, norm, gs, device, cfg.get("targets", "vm"))):
            m = graph_metrics(g, yp, gp)
            m["set"] = split_name
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.run) / "per_graph_metrics.csv", index=False)
    summary = {k: summarize(df[df.set == k]) for k in ("test", "ood", "val", "shift") if (df.set == k).any()}
    summary["inference_s_per_graph"] = time_inference(model, test, device)
    summary["device"] = str(device)
    summary["cfg"] = cfg
    (Path(args.run) / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    t = summary["test"]
    print(json.dumps({k: t[k] for k in t if k != "n"}, indent=1, default=float))
    print("OOD:", json.dumps({k: summary["ood"][k] for k in ("vm_rel_l2", "u_rel_l2")}, default=float))


if __name__ == "__main__":
    main()
