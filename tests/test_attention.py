import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gnnsurrogate.attention import MultiHeadSelfAttention, scaled_dot_product_attention, softmax
from gnnsurrogate.data import collate, graph_from_mesh
from gnnsurrogate.transformer import SliceTransformer

TOL = 1e-5


@pytest.mark.parametrize("shape", [(1, 1, 5, 8), (2, 4, 7, 16), (3, 8, 32, 12)])
def test_attention_matches_torch(shape):
    g = torch.Generator().manual_seed(0)
    b, h, l, d = shape
    q, k, v = (torch.randn(b, h, l, d, generator=g) for _ in range(3))
    ours = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert ours.dtype == torch.float32
    assert (ours - ref).abs().max().item() < TOL


def test_attention_with_mask_and_different_lengths():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(2, 3, 6, 8, generator=g)
    k = torch.randn(2, 3, 9, 8, generator=g)
    v = torch.randn(2, 3, 9, 5, generator=g)
    mask = torch.rand(2, 3, 6, 9, generator=g) > 0.4
    mask[..., 0] = True                                     # every query sees at least one key
    ours = scaled_dot_product_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert (ours - ref).abs().max().item() < TOL


def test_softmax_large_inputs():
    x = torch.tensor([[1000.0, 1001.0, 999.0], [-5.0, 0.0, 5.0]])
    assert torch.isfinite(softmax(x)).all()
    assert (softmax(x) - torch.softmax(x, -1)).abs().max().item() < TOL


def test_multihead_matches_nn_multiheadattention():
    """Same weights in torch's module -> same output, so the head split / merge (reshape) is right."""
    torch.manual_seed(0)
    dim, heads = 32, 4
    ours = MultiHeadSelfAttention(dim, heads)
    ref = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(ours.qkv.weight)
        ref.in_proj_bias.copy_(ours.qkv.bias)
        ref.out_proj.weight.copy_(ours.out.weight)
        ref.out_proj.bias.copy_(ours.out.bias)
    x = torch.randn(3, 11, dim)
    with torch.no_grad():
        a = ours(x)
        r, _ = ref(x, x, x, need_weights=False)
    assert (a - r).abs().max().item() < TOL


def random_graph(n=40, seed=0, params=(6, 6, 5, 20, 5, 10)):
    """A random point cloud with random tetrahedra; enough to exercise the model."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 100, size=(n, 3))
    tets = np.stack([rng.choice(n, 4, replace=False) for _ in range(2 * n)])
    flags = rng.integers(0, 2, size=(n, 4)).astype(np.uint8)
    return graph_from_mesh(pos, tets, flags, params)


def permuted(g, perm):
    """The same graph with its nodes renumbered: new node i is old node perm[i]."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    return type(g)(x=g.x[perm], edge_index=inv[g.edge_index], edge_attr=g.edge_attr, y=g.y[perm], g=g.g,
                   params=g.params, region=g.region[perm], pos=g.pos[perm], meta=g.meta)


@pytest.mark.parametrize("local", [False, True])
def test_permutation_equivariance(local):
    torch.manual_seed(0)
    model = SliceTransformer(hidden=32, layers=2, heads=4, n_slices=8, local=local).eval()
    g = random_graph()
    perm = torch.randperm(g.num_nodes, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        n0, g0 = model(collate([g]))
        n1, g1 = model(collate([permuted(g, perm)]))
    assert (n1 - n0[perm]).abs().max().item() < TOL     # node outputs are shuffled the same way
    assert (g1 - g0).abs().max().item() < TOL           # graph outputs do not change


def test_graphs_in_a_batch_do_not_interact_and_padding_is_ignored():
    torch.manual_seed(0)
    model = SliceTransformer(hidden=32, layers=2, heads=4, n_slices=8).eval()
    a, b = random_graph(30, seed=1), random_graph(50, seed=2, params=(10, 5, 8, 0, 4, 0))
    with torch.no_grad():
        na, ga = model(collate([a]))
        nab, gab = model(collate([a, b], 64, 512))
    assert gab.shape == (2, 3)
    assert (nab[:30] - na).abs().max().item() < TOL
    assert (gab[0] - ga[0]).abs().max().item() < TOL


def test_no_params_variant_ignores_params():
    torch.manual_seed(0)
    model = SliceTransformer(hidden=32, layers=2, heads=4, n_slices=8, use_params=False).eval()
    a = random_graph(seed=4)
    b = random_graph(seed=4, params=(11, 9, 3, 40, 7, 18))
    with torch.no_grad():
        assert torch.equal(model(collate([a]))[0], model(collate([b]))[0])
