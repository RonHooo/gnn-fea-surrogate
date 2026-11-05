"""Schematic of one slice-attention layer (for the README / explanation): nodes -> slices -> attention -> nodes."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results/attention"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)

# an L-shaped cloud of "mesh nodes"
upright = np.c_[rng.uniform(0, 0.15, 45), rng.uniform(0, 1, 45)]
base = np.c_[rng.uniform(0.15, 1, 55), rng.uniform(0, 0.15, 55)]
pts = np.r_[upright, base]
centers = np.array([[0.07, 0.8], [0.07, 0.35], [0.35, 0.07], [0.8, 0.07]])
logits = -((pts[:, None, :] - centers[None]) ** 2).sum(-1) / 0.03
w = np.exp(logits - logits.max(1, keepdims=True))
w /= w.sum(1, keepdims=True)
cols = np.array(["#1b9e77", "#d95f02", "#7570b3", "#e7298a"])
tok = np.array([[0.5, 0.85], [0.5, 0.6], [0.5, 0.35], [0.5, 0.1]])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["1. each node is softly assigned to M slices\n(softmax over M scores, colour = most likely slice)",
          "2. one token per slice = weighted mean of its nodes;\nmulti-head self-attention among the M tokens",
          "3. tokens are sent back to the nodes with\nthe same weights, then residual + FFN"]
for a, t in zip(axes, titles):
    a.set_title(t, fontsize=10)
    a.set_xlim(-0.05, 1.05)
    a.set_ylim(-0.05, 1.05)
    a.set_aspect("equal")
    a.axis("off")
axes[0].scatter(*pts.T, c=cols[w.argmax(1)], s=25, alpha=0.9)
axes[0].text(0.3, 0.6, f"N nodes (here {len(pts)};\nin the bracket 3,000-5,000)", fontsize=9)
for i in range(4):
    axes[1].scatter(*tok[i], s=900, marker="s", c=cols[i])
    axes[1].text(tok[i, 0] + 0.08, tok[i, 1], f"slice token {i + 1}", va="center", fontsize=9)
    for j in range(4):
        if i < j:
            axes[1].annotate("", xy=tok[j] - [0.05, 0], xytext=tok[i] - [0.05, 0],
                             arrowprops=dict(arrowstyle="<->", color="gray", connectionstyle="arc3,rad=0.5"))
axes[1].text(0.0, -0.03, "M tokens (32 in the main model): M x M scores\ninstead of N x N", fontsize=9)
axes[2].scatter(*pts.T, c=cols[w.argmax(1)], s=25, alpha=0.9)
i = 3
axes[2].scatter(*pts[i], s=120, facecolors="none", edgecolors="k", linewidths=2)
axes[2].text(0.25, 0.6, "every node now carries information\nfrom the whole part (one layer)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "slice_attention_diagram.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "slice_attention_diagram.png")
