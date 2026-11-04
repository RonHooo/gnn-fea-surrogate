"""Print the model-comparison table (markdown) from the summary.json files."""

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "results/runs"
ROWS = [("gnn_stress", "GNN, stress-component targets (final)"),
        ("gnn", "GNN, von Mises target"),
        ("gnn_noglobal", "GNN, von Mises target, no global node"),
        ("gnn_noparams", "GNN, von Mises target, mesh only (no design variables)"),
        ("gnn_noparams_noglobal", "GNN, von Mises target, mesh only, no global node"),
        ("pointmlp", "point-wise MLP, no edges (von Mises target)")]


def pct(t, k, key="mean"):
    return f"{100 * t[k][key]:.1f} %" if k in t else "-"


def main():
    print("| model | u field | von Mises field | f1 | peak von Mises | log10 life | von Mises field, shifted family |")
    print("|---|---|---|---|---|---|---|")
    for run, label in ROWS:
        f = RUNS / run / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        t, sh = s["test"], s.get("shift", {})
        life = f"{t['log10_life_err']['mean_abs']:.3f}" if "log10_life_err" in t else "-"
        f1 = f"{100 * t['f1_err']['mean_abs']:.2f} %" if "f1_err" in t else "-"
        vmx = f"{100 * t['vm_max_err']['mean_abs']:.1f} %" if "vm_max_err" in t else "-"
        print(f"| {label} | {pct(t, 'u_rel_l2')} | {pct(t, 'vm_rel_l2')} | {f1} | {vmx} | {life} | "
              f"{pct(sh, 'vm_rel_l2') if sh else '-'} |")
    gp = json.loads((RUNS / "gp/summary.json").read_text())
    t = gp["test"]
    print(f"| GP on the 6 design variables | - | - | {100 * t['f1_err_mean_abs']:.2f} % | "
          f"{100 * t['vm_max_err_mean_abs']:.1f} % | {t['log10_life_err_mean_abs']:.3f} | - |")


if __name__ == "__main__":
    main()
