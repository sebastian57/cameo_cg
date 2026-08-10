import numpy as np
import pytest

from config.manager import ConfigManager
from data.loader import build_tiled_dataset
from data.static_neighbors import (
    StaticGraphIncompatibleError,
    assert_static_graph_compatible,
    build_static_graphs,
    build_tile_graph,
    segment_pairs_chunked,
    segment_pairs_kdtree,
)

CUTOFF = 5.5
DR_THRESHOLD = 1.0
R_LIST = CUTOFF + DR_THRESHOLD


def _config(data=None, model=None, training=None):
    cfg = ConfigManager.__new__(ConfigManager)
    cfg._config = {
        "data": {"batch_mode": "tiled", **(data or {})},
        "model": {"cutoff": CUTOFF, "dr_threshold": DR_THRESHOLD, **(model or {})},
        "training": training or {},
    }
    return cfg


def _edge_set(idx, n_atoms):
    """Directed edge set with padding sentinel entries removed."""
    receivers, senders = np.asarray(idx)
    keep = (receivers < n_atoms) & (senders < n_atoms)
    return set(zip(senders[keep].tolist(), receivers[keep].tolist()))


def _reference_pairs(X, r_list, dtype=np.float32):
    X = np.asarray(X, dtype=dtype)
    d = X[:, None, :] - X[None, :, :]
    d2 = d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2
    hit = (d2 < dtype(r_list) ** 2) & np.triu(np.ones(d2.shape, dtype=bool), k=1)
    return set(map(tuple, np.argwhere(hit)))


def _toy_structures(n_structures=24, n_atoms=8, seed=0):
    rng = np.random.RandomState(seed)
    R = rng.uniform(0.0, 6.0, size=(n_structures, n_atoms, 3)).astype(np.float32)
    F = np.zeros_like(R)
    mask = np.ones(R.shape[:2], dtype=np.float32)
    species = np.ones(R.shape[:2], dtype=np.int32)
    return R, F, mask, species


# --------------------------------------------------------------------------- #
#  Radius search backends
# --------------------------------------------------------------------------- #


def test_chain_has_analytic_edges():
    n = 8
    X = np.zeros((n, 3), dtype=np.float32)
    X[:, 0] = np.arange(n)

    assert set(map(tuple, segment_pairs_chunked(X, 1.5))) == {
        (i, i + 1) for i in range(n - 1)
    }
    assert set(map(tuple, segment_pairs_chunked(X, 2.5))) == {
        (i, j) for i in range(n) for j in range(i + 1, n) if j - i <= 2
    }


def test_cutoff_comparator_is_strict():
    """A pair exactly at the radius is excluded, matching JAX-MD's `d2 < r2`."""
    X = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    assert segment_pairs_chunked(X, 2.0).shape[0] == 0
    assert segment_pairs_kdtree(X, 2.0).shape[0] == 0
    assert segment_pairs_chunked(X, 2.0 + 1e-3).shape[0] == 1
    assert segment_pairs_kdtree(X, 2.0 + 1e-3).shape[0] == 1


@pytest.mark.parametrize("block_size", [1, 3, 17, 64, 4096])
def test_chunked_blocking_is_exact(block_size):
    """Every block pair is visited, so results are block-size independent."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0.0, 12.0, size=(200, 3)).astype(np.float32)
    assert set(map(tuple, segment_pairs_chunked(X, CUTOFF, block_size=block_size))) == (
        _reference_pairs(X, CUTOFF)
    )


def test_backends_agree():
    rng = np.random.RandomState(1)
    X = rng.uniform(0.0, 20.0, size=(500, 3)).astype(np.float32)
    assert (
        set(map(tuple, segment_pairs_chunked(X, R_LIST)))
        == set(map(tuple, segment_pairs_kdtree(X, R_LIST)))
        == _reference_pairs(X, R_LIST)
    )


# --------------------------------------------------------------------------- #
#  Tile assembly
# --------------------------------------------------------------------------- #


def test_tile_graph_respects_segments_and_mask():
    """No cross-segment, self, or padded-node edges even when segments overlap."""
    rng = np.random.RandomState(2)
    n_valid = 40
    R = np.zeros((n_valid + 2, 3), dtype=np.float32)
    R[:n_valid] = rng.uniform(0.0, 6.0, size=(n_valid, 3))
    mask = np.zeros((n_valid + 2,), dtype=np.float32)
    mask[:n_valid] = 1.0
    segment_id = np.full((n_valid + 2,), -1, dtype=np.int32)
    segment_id[: n_valid // 2] = 0
    segment_id[n_valid // 2 : n_valid] = 1

    idx, n_edges = build_tile_graph(R, mask, segment_id, CUTOFF, backend="chunked")
    edges = _edge_set(idx, R.shape[0])

    assert n_edges == idx.shape[1]
    assert edges, "overlapping segments should still produce intra-segment edges"
    for s, r in edges:
        assert s != r, "self edge"
        assert mask[s] > 0 and mask[r] > 0, "padded-node edge"
        assert segment_id[s] == segment_id[r], "cross-segment edge"


def test_tile_graph_is_bidirectional_and_deterministic():
    rng = np.random.RandomState(3)
    R = rng.uniform(0.0, 8.0, size=(60, 3)).astype(np.float32)
    mask = np.ones((60,), dtype=np.float32)
    segment_id = np.zeros((60,), dtype=np.int32)

    idx_a, _ = build_tile_graph(R, mask, segment_id, CUTOFF, backend="chunked")
    idx_b, _ = build_tile_graph(R, mask, segment_id, CUTOFF, backend="kdtree")

    edges = _edge_set(idx_a, 60)
    assert all((r, s) in edges for s, r in edges), "missing reverse orientation"
    assert np.array_equal(idx_a, idx_b), "backends must agree down to edge order"


def test_build_static_graphs_pads_with_sentinel():
    rng = np.random.RandomState(4)
    n_tiles, n_atoms = 3, 50
    R = rng.uniform(0.0, 9.0, size=(n_tiles, n_atoms, 3)).astype(np.float32)
    mask = np.ones((n_tiles, n_atoms), dtype=np.float32)
    segment_id = np.zeros((n_tiles, n_atoms), dtype=np.int32)

    out = build_static_graphs(R, mask, segment_id, CUTOFF, capacity_multiplier=1.25)
    capacity = int(out["neighbor_capacity"][0])

    assert out["neighbor_idx"].shape == (n_tiles, 2, capacity)
    assert out["neighbor_capacity"].shape == (n_tiles,), "must shard by tile"
    for t in range(n_tiles):
        n = int(out["neighbor_n_edges"][t])
        assert np.all(out["neighbor_idx"][t, :, n:] == n_atoms), "padding sentinel"
        assert np.all(out["neighbor_idx"][t, :, :n] < n_atoms), "valid edge range"


# --------------------------------------------------------------------------- #
#  build_tiled_dataset integration
# --------------------------------------------------------------------------- #


def test_tiled_dataset_attaches_graphs():
    R, F, mask, species = _toy_structures()
    cfg = _config(data={"static_neighbors": {"enabled": True}})
    tiled = build_tiled_dataset(
        R,
        F,
        mask,
        species,
        target_beads=64,
        spatial_separation=True,
        spatial_layout="grid_3d",
        structure_gap=30.0,
        static_neighbors=cfg.get_static_neighbors_config(),
    )

    n_tiles, n_atoms = tiled["R"].shape[0], tiled["R"].shape[1]
    assert tiled["neighbor_idx"].shape[:2] == (n_tiles, 2)
    assert tiled["neighbor_idx"].dtype == np.int32

    for t in range(n_tiles):
        edges = _edge_set(tiled["neighbor_idx"][t], n_atoms)
        seg = tiled["segment_id"][t]
        for s, r in edges:
            assert seg[s] == seg[r] >= 0, "cross-segment or padded edge in tile"


def test_tiled_dataset_without_config_has_no_graphs():
    R, F, mask, species = _toy_structures()
    tiled = build_tiled_dataset(R, F, mask, species, target_beads=64)
    assert "neighbor_idx" not in tiled


def test_graph_matches_dynamic_jaxmd_path():
    """Edge-set equality against the neighbor list the dynamic path would build."""
    jnp = pytest.importorskip("jax.numpy")
    jax = pytest.importorskip("jax")
    partition = pytest.importorskip("jax_md.partition")
    space = pytest.importorskip("jax_md.space")
    custom_partition = pytest.importorskip("jax_md_mod.custom_partition")

    R, F, mask, species = _toy_structures(n_structures=12, n_atoms=10)
    cfg = _config(data={"static_neighbors": {"enabled": True}})
    tiled = build_tiled_dataset(
        R,
        F,
        mask,
        species,
        target_beads=60,
        spatial_separation=True,
        spatial_layout="grid_3d",
        structure_gap=30.0,
        static_neighbors=cfg.get_static_neighbors_config(),
    )

    tile_R = np.asarray(tiled["R"][0])
    tile_mask = np.asarray(tiled["mask"][0])
    tile_seg = np.asarray(tiled["segment_id"][0])
    n_atoms = tile_R.shape[0]
    box = float(np.max(np.abs(tile_R)) + R_LIST + 1.0)

    displacement, _ = space.free()
    neighbor_fn = custom_partition.masked_neighbor_list(
        displacement,
        box=jnp.asarray(box, dtype=jnp.float32),
        r_cutoff=CUTOFF,
        dr_threshold=DR_THRESHOLD,
        capacity_multiplier=1.25,
        fractional_coordinates=False,
        disable_cell_list=True,
        format=partition.Sparse,
    )
    valid = jnp.asarray(tile_mask > 0, dtype=jnp.bool_)
    nbrs = neighbor_fn.allocate(
        jnp.asarray(tile_R, dtype=jnp.float32), extra_capacity=10, mask=valid
    )
    nbrs = custom_partition.mask_neighbor_list(
        nbrs, mask=valid, segment_id=jnp.asarray(tile_seg, dtype=jnp.int32)
    )

    dynamic = _edge_set(np.asarray(jax.device_get(nbrs.idx)), n_atoms)
    static = _edge_set(tiled["neighbor_idx"][0], n_atoms)
    assert static == dynamic


def test_static_shell_is_used_verbatim_by_models():
    """`error=None` is the signal the Allegro paths use to skip neighbor_update."""
    jax = pytest.importorskip("jax")
    partition = pytest.importorskip("jax_md.partition")
    custom_partition = pytest.importorskip("jax_md_mod.custom_partition")

    idx = np.array([[1, 2, 3], [0, 0, 1]], dtype=np.int32)
    pos = np.zeros((4, 3), dtype=np.float32)
    nbrs = custom_partition.static_neighbor_list(idx, pos)

    assert nbrs.error is None
    assert nbrs.format is partition.Sparse
    assert np.array_equal(np.asarray(jax.device_get(nbrs.idx)), idx)
    assert nbrs.max_occupancy == 3


# --------------------------------------------------------------------------- #
#  Compatibility guard
# --------------------------------------------------------------------------- #


def test_plain_tiled_force_matching_passes():
    assert assert_static_graph_compatible(_config()) is None


def test_standard_batch_mode_rejected():
    with pytest.raises(StaticGraphIncompatibleError, match="batch_mode='tiled'"):
        assert_static_graph_compatible(_config(data={"batch_mode": "standard"}))


@pytest.mark.parametrize(
    "training, expected",
    [
        ({"dsm": {"enabled": True}}, "training.dsm.enabled"),
        (
            {"noised_residual_training": {"enabled": True}},
            "training.noised_residual_training.enabled",
        ),
        ({"relative_entropy": {"enabled": True}}, "training.relative_entropy.enabled"),
    ],
)
def test_graph_invalidating_mechanisms_rejected(training, expected):
    with pytest.raises(StaticGraphIncompatibleError, match=expected):
        assert_static_graph_compatible(_config(training=training))


def test_pbc_rejected():
    with pytest.raises(StaticGraphIncompatibleError, match="model.pbc"):
        assert_static_graph_compatible(_config(model={"pbc": True}))


def test_all_incompatibilities_reported_together():
    """One run surfaces every blocking mechanism, not just the first."""
    config = _config(
        model={"pbc": True},
        training={"dsm": {"enabled": True}, "relative_entropy": {"enabled": True}},
    )
    with pytest.raises(StaticGraphIncompatibleError) as excinfo:
        assert_static_graph_compatible(config)
    for key in ("training.dsm.enabled", "training.relative_entropy.enabled", "model.pbc"):
        assert key in str(excinfo.value)


@pytest.mark.parametrize(
    "training",
    [
        {"hvp": {"enabled": True}},
        {"msam": {"enabled": True}},
        {"swa": {"enabled": True}},
        {"support_gate": {"enabled": True}},
        {"dsm": {"enabled": False}},
    ],
)
def test_compatible_mechanisms_not_rejected(training):
    """HVP differentiates at R and the rest never move coordinates."""
    assert assert_static_graph_compatible(_config(training=training)) is None


def test_noise_decoys_not_rejected():
    """Decoys are appended before tiling, so graphs come from their stored R."""
    config = _config(data={"noise_decoys": {"every_n": 4, "sigma": 0.5}})
    assert assert_static_graph_compatible(config) is None


# --------------------------------------------------------------------------- #
#  Config contract
# --------------------------------------------------------------------------- #


def test_r_list_defaults_to_cutoff_plus_skin():
    cfg = _config().get_static_neighbors_config()
    assert cfg["r_list"] == pytest.approx(CUTOFF + DR_THRESHOLD)
    assert cfg["backend"] == "kdtree"
    assert cfg["enabled"] is False


def test_r_list_below_cutoff_rejected():
    config = _config(data={"static_neighbors": {"r_list": CUTOFF - 1.0}})
    with pytest.raises(ValueError, match="below.*model.cutoff"):
        config.get_static_neighbors_config()


def test_unknown_backend_rejected():
    config = _config(data={"static_neighbors": {"backend": "octree"}})
    with pytest.raises(ValueError, match="backend"):
        config.get_static_neighbors_config()
