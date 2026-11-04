import numpy as np
import torch

from gnnsurrogate.data import tet_edges, graph_from_mesh, collate
from gnnsurrogate.model import MeshGNN, scatter_mean, scatter_max


def two_tets():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    flags = np.zeros((5, 4), dtype=np.uint8)
    flags[:, 3] = 1
    return pos, tets, flags


def test_tet_edges_unique_and_symmetric():
    _, tets, _ = two_tets()
    ei = tet_edges(tets)
    pairs = set(map(tuple, ei.T))
    assert len(pairs) == ei.shape[1]                    # no duplicates
    assert all((b, a) in pairs for a, b in pairs)       # both directions
    assert ei.shape[1] == 2 * 9                         # 6 + 6 edges, 3 shared


def test_scatter_ops():
    src = torch.tensor([[1.0], [3.0], [5.0], [-2.0]])
    idx = torch.tensor([0, 0, 1, 1])
    assert torch.allclose(scatter_mean(src, idx, 2), torch.tensor([[2.0], [1.5]]))
    assert torch.allclose(scatter_max(src, idx, 2), torch.tensor([[3.0], [5.0]]))


def test_padding_does_not_change_predictions():
    pos, tets, flags = two_tets()
    g1 = graph_from_mesh(pos, tets, flags, [6, 6, 5, 0, 5, 0])
    g2 = graph_from_mesh(pos * 1.1, tets, flags, [8, 6, 5, 30, 5, 10])
    torch.manual_seed(0)
    model = MeshGNN(hidden=16, layers=3).eval()
    with torch.no_grad():
        n0, g0 = model(collate([g1, g2]))
        n1, g1p = model(collate([g1, g2], 16, 64))
    assert g1p.shape == (2, 3)
    assert torch.allclose(n0, n1[:10], atol=1e-5)
    assert torch.allclose(g0, g1p, atol=1e-5)


def test_padding_mask_and_shapes():
    pos, tets, flags = two_tets()
    g = graph_from_mesh(pos, tets, flags, [6, 6, 5, 0, 5, 0])
    b = collate([g, g], 16, 64)
    assert b.x.shape[0] == 16 and b.edge_index.shape[1] == 64
    assert int(b.mask.sum()) == 10
    assert b.n_slots == 3 and b.params.shape[0] == 3
