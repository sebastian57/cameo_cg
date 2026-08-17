import numpy as np
import pytest


def test_loader_preserves_frame_boxes_and_split(tmp_path):
    from data.loader import DatasetLoader

    path = tmp_path / "dynamic.npz"
    boxes = np.asarray([[10.0, 11.0, 12.0], [10.5, 11.5, 12.5], [11.0, 12.0, 13.0], [11.5, 12.5, 13.5]], dtype=np.float32)
    R = np.zeros((4, 2, 3), dtype=np.float32)
    F = np.zeros_like(R)
    np.savez(path, R=R, F=F, box=boxes)

    loader = DatasetLoader(path, dynamic_box=True)
    np.testing.assert_allclose(loader.box, boxes[0])
    np.testing.assert_allclose(loader.get_all()["box"], boxes)
    train, val = loader.split_train_val(val_fraction=1.0 / 3.0)
    np.testing.assert_allclose(train.box_per_frame, boxes[:3])
    np.testing.assert_allclose(val.box_per_frame, boxes[3:])


def test_dynamic_neighbor_list_changes_with_box_and_is_jittable():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax_md import space
    from models.dynamic_neighborlist import dynamic_neighbor_list

    R = jnp.asarray([[0.0, 0.0, 0.0], [7.2, 0.0, 0.0], [3.0, 3.0, 3.0], [7.0, 7.0, 7.0]], dtype=jnp.float32)
    small_box = jnp.asarray([10.0, 10.0, 10.0], dtype=jnp.float32)
    large_box = jnp.asarray([14.0, 14.0, 14.0], dtype=jnp.float32)
    displacement, _ = space.periodic_general(large_box, fractional_coordinates=False)
    neighbor_fn = dynamic_neighbor_list(
        displacement, large_box, r_cutoff=3.0, capacity_multiplier=2.0, box_min=small_box
    )

    nbrs = neighbor_fn.allocate(R, box=small_box)
    updated = neighbor_fn.update(R, nbrs, box=large_box)
    updated_jit = jax.jit(lambda box: neighbor_fn.update(R, nbrs, box=box))(large_box)

    assert int(jnp.sum(nbrs.idx < R.shape[0])) > int(jnp.sum(updated.idx < R.shape[0]))
    np.testing.assert_array_equal(updated.idx, updated_jit.idx)
    assert not bool(jnp.any(updated.error.code))


def test_dynamic_box_config_requires_pbc(tmp_path):
    from config.manager import ConfigManager

    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  dynamic_box: true\nmodel:\n  pbc: false\ntraining: {}\noptimizer: {}\n"
    )
    with pytest.raises(ValueError, match="requires model.pbc"):
        ConfigManager(path)

    unsupported = tmp_path / "unsupported.yaml"
    unsupported.write_text(
        "data:\n  dynamic_box: true\nmodel:\n  pbc: true\n  ml_model: mace\ntraining: {}\noptimizer: {}\n"
    )
    with pytest.raises(ValueError, match="Allegro backends only"):
        ConfigManager(unsupported)
