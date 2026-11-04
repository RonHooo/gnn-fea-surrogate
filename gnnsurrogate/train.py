"""Train the mesh GNN (or the point-wise MLP baseline) on the bracket dataset."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_dataset, split_dataset, Normalizer, collate, TARGET_COLS
from .model import MeshGNN, PointMLP

PAD = (1024, 8192)   # node / edge bucket sizes for the MPS backend


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: dict) -> torch.nn.Module:
    n_out = len(TARGET_COLS[cfg.get("targets", "vm")])
    if cfg["model"] == "pointmlp":
        return PointMLP(hidden=cfg.get("hidden", 256), layers=cfg.get("layers", 6), node_out=n_out)
    return MeshGNN(hidden=cfg["hidden"], layers=cfg["layers"], use_params=cfg["use_params"],
                   use_global=cfg["use_global"], node_out=n_out)


def losses(model, batch, norm: Normalizer, w_global: float, cols=TARGET_COLS["vm"]):
    pred_n, pred_g = model(batch)
    yn = (batch.y[:, cols] - norm.y_mean[cols]) / norm.y_std[cols]
    m = batch.mask
    l_node = torch.mean((pred_n[m] - yn[m]) ** 2)
    if pred_g is None:
        return l_node, l_node, torch.zeros((), device=l_node.device)
    gn = (batch.g - norm.g_mean) / norm.g_std
    l_glob = torch.mean((pred_g - gn) ** 2)
    return l_node + w_global * l_glob, l_node, l_glob


@torch.no_grad()
def evaluate_loss(model, graphs, norm, device, w_global, bs=4, cols=TARGET_COLS["vm"]):
    model.eval()
    tot = np.zeros(3)
    for i in range(0, len(graphs), bs):
        b = collate(graphs[i:i + bs], *PAD).to(device)
        l, ln, lg = losses(model, b, norm, w_global, cols)
        tot += np.array([l.item(), ln.item(), lg.item()]) * b.n_graphs
    return tot / len(graphs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/coarse")
    ap.add_argument("--cache", default="data/graphs_coarse.pt")
    ap.add_argument("--out", default="results/runs/gnn")
    ap.add_argument("--model", default="gnn", choices=["gnn", "pointmlp"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--no-params", action="store_true")
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument("--targets", default="vm", choices=list(TARGET_COLS))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-global", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-hours", type=float, default=10.0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    graphs = load_dataset(args.raw, args.cache)
    train, val, test, ood = split_dataset(graphs)
    norm = Normalizer.fit(train)
    normd = norm.to(device)
    cfg = dict(model=args.model, hidden=args.hidden, layers=args.layers,
               use_params=not args.no_params, use_global=not args.no_global, seed=args.seed,
               targets=args.targets,
               epochs=args.epochs, batch=args.batch, lr=args.lr, w_global=args.w_global,
               n_train=len(train), n_val=len(val), n_test=len(test), n_ood=len(ood))
    model = build_model(cfg).to(device)
    cols = TARGET_COLS[args.targets]
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_par} train={len(train)} val={len(val)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch * 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    hist, best, t0 = [], float("inf"), time.time()
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        run = np.zeros(3)
        for i in range(0, len(order), args.batch):
            b = collate([train[j] for j in order[i:i + args.batch]], *PAD).to(device)
            l, ln, lg = losses(model, b, normd, args.w_global, cols)
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += np.array([l.item(), ln.item(), lg.item()]) * b.n_graphs
        run /= len(train)
        v = evaluate_loss(model, val, normd, device, args.w_global, cols=cols)
        el = (time.time() - t0) / 3600
        hist.append(dict(epoch=epoch, train=run[0], train_node=run[1], train_glob=run[2],
                         val=v[0], val_node=v[1], val_glob=v[2], hours=el, lr=sched.get_last_lr()[0]))
        if v[0] < best:
            best = v[0]
            torch.save(dict(state=model.state_dict(), cfg=cfg, norm=norm.state(), epoch=epoch),
                       out / "model.pt")
        print(f"epoch {epoch:4d} train {run[0]:.4f} (node {run[1]:.4f} glob {run[2]:.4f}) "
              f"val {v[0]:.4f} (node {v[1]:.4f} glob {v[2]:.4f}) {el * 60:.1f} min", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if el > args.max_hours:
            print("time budget reached", flush=True)
            break
    (out / "config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
