"""Build the coarse graphs used by the GNN.

For every FE sample in data/raw (corner nodes of the C3D10 solution), re-mesh the same geometry with
a coarse linear tet mesh, interpolate the FE displacement and stress components onto its nodes, and
recompute von Mises from the interpolated components. Scalars (f1, fatigue life, peak stress of the
full FE mesh, mass) are copied unchanged.

Why: the FE meshes have ~10k corner nodes and ~120k directed edges per design, which made one
training epoch take > 6 min on the laptop GPU. The coarse graphs have 3-5k nodes.
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

COARSE = dict(h_max=4.0, h_min=2.5, h_hole=3.0, dist_max=6.0, order=1)
NAMES = ["t_b", "t_w", "r_f", "L_g", "t_g", "d_h"]


def von_mises(s):
    sxx, syy, szz, sxy, syz, szx = (s[:, k] for k in range(6))
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def interpolate(fine_pos, fine_tets, fields: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Linear interpolation inside the fine tet mesh; nearest fine node where a point falls outside
    (coarse nodes on curved surfaces can sit just outside the faceted fine surface)."""
    import pyvista as pv
    from scipy.spatial import cKDTree
    cells = np.hstack([np.full((len(fine_tets), 1), 4), fine_tets]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(fine_tets), 10, dtype=np.uint8), fine_pos)
    grid.point_data["f"] = fields
    res = pv.PolyData(pts).sample(grid, tolerance=1e-6)
    out = np.asarray(res.point_data["f"]).reshape(len(pts), -1).copy()
    valid = np.asarray(res.point_data["vtkValidPointMask"]).astype(bool)
    if (~valid).any():
        _, idx = cKDTree(fine_pos).query(pts[~valid])
        out[~valid] = fields[idx]
    return out, float((~valid).mean())


def build_one(args):
    src, dst_dir = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from bracketfea.geometry import BracketParams, MeshSettings, mesh_bracket
    from bracketfea.analysis import design_region
    dst = Path(dst_dir) / Path(src).name
    if dst.exists():
        return str(dst), 0.0
    z = np.load(src)
    if "geom" in z.files:
        import json
        p = BracketParams(**json.loads(str(z["geom"])))
    else:
        p = BracketParams(**{k: float(v) for k, v in zip(NAMES, z["params"])})
    m = mesh_bracket(p, MeshSettings(**COARSE))
    fields = np.concatenate([z["u"], z["s"]], axis=1).astype(np.float64)
    vals, frac_out = interpolate(z["pos"].astype(np.float64), z["tets"].astype(np.int64), fields, m.nodes)
    flags = np.zeros((m.n_nodes, 4), dtype=np.uint8)
    flags[m.sets["FIX"], 0] = 1
    flags[m.sets["MOUNT"], 1] = 1
    flags[m.surface_nodes(), 2] = 1
    flags[design_region(m.nodes, p), 3] = 1
    keep = {k: z[k] for k in ("params", "freq", "mass", "vm_max", "life", "rms", "n_nodes_full")}
    np.savez_compressed(dst, pos=m.nodes.astype(np.float32), tets=m.tets.astype(np.int32), flags=flags,
                        u=vals[:, :3].astype(np.float32), s=vals[:, 3:].astype(np.float32),
                        vm=von_mises(vals[:, 3:]).astype(np.float32),
                        vm_max_fe_corner=np.float32(z["vm"][z["flags"][:, 3] > 0].max()),
                        frac_outside=np.float32(frac_out), **keep)
    return str(dst), frac_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/coarse")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(Path(args.raw) / "*.npz")))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(build_one, [(f, args.out) for f in files]))
    fr = np.array([r[1] for r in res])
    print(f"{len(res)} graphs, nearest-node fallback: mean {100 * fr.mean():.2f}% of nodes, max {100 * fr.max():.2f}%")


if __name__ == "__main__":
    main()
