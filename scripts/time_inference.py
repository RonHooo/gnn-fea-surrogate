"""Inference time per design for the GNN and the attention models, measured back to back in one process
(same machine, same batch of one design, same padding), so the numbers are comparable."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnnsurrogate.data import load_dataset, split_dataset
from gnnsurrogate.evaluate import load_model, time_inference
from gnnsurrogate.train import pick_device

RUNS = ["gnn_stress", "transformer_s0", "transformer_m16_s0", "transformer_m64_s0", "transformer_local_s0",
        "pointmlp"]


def main():
    device = pick_device("auto")
    graphs = load_dataset(ROOT / "data/coarse", ROOT / "data/graphs_coarse.pt")
    _, _, test, _ = split_dataset(graphs)
    out = {"device": str(device), "n_designs": len(test), "repeats": 5}
    for run in RUNS:
        if not (ROOT / "results/runs" / run / "model.pt").exists():
            continue
        model, _, cfg = load_model(ROOT / "results/runs" / run, device)
        t = [time_inference(model, test, device, n=len(test)) for _ in range(out["repeats"])]
        out[run] = dict(median_s=float(np.median(t)), min_s=float(np.min(t)), max_s=float(np.max(t)),
                        n_params=sum(p.numel() for p in model.parameters()))
        print(run, json.dumps(out[run]), flush=True)
    dest = ROOT / "results/attention"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "inference_time.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
