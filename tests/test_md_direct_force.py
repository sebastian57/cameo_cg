import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax_md import space

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from md.runner import MDRunner


class _NoNeighborUpdate:
    @staticmethod
    def update(position, neighbor, mask=None):
        del position, mask
        return neighbor


class _DirectML:
    def __init__(self):
        _, self.shift = space.free()
        self.nbrs_init = None
        self.nneigh_fn = _NoNeighborUpdate()


class _DirectModel:
    direct_force_enabled = True
    use_priors = False

    def __init__(self):
        self.ml_model = _DirectML()

    @staticmethod
    def compute_direct_force(
        params, R, mask, species, neighbor=None, segment_id=None
    ):
        del species, neighbor, segment_id
        return -params["ml"]["k"] * R * mask[:, None]


@pytest.mark.parametrize("scan_chunk_size", [0, 2])
def test_direct_force_langevin_md_has_forces_without_fake_energy(scan_chunk_size):
    runner = MDRunner(
        _DirectModel(),
        {"ml": {"k": jnp.asarray(0.5, dtype=jnp.float32)}},
        {
            "integrator": "nvt_langevin",
            "n_steps": 4,
            "dt": 0.001,
            "kT": 0.2,
            "gamma": 1.0,
            "mass": 12.0,
            "output_every": 1,
            "observables_every": 1,
            "scan_chunk_size": scan_chunk_size,
            "zero_com_velocity": True,
            "observables": ["step", "T", "KE", "PE", "E_total"],
        },
    )
    trajectory = runner.run(
        jnp.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.int32),
        random.PRNGKey(7),
    )

    assert trajectory["complete"]
    assert trajectory["model_output_mode"].item() == "direct_force"
    assert not trajectory["energy_available"]
    assert np.isfinite(trajectory["R"]).all()
    assert np.isfinite(trajectory["F"]).all()
    assert np.isfinite(trajectory["T"]).all()
    assert np.isnan(trajectory["PE"]).all()
    assert np.isnan(trajectory["obs_PE"]).all()
    assert np.isnan(trajectory["obs_E_total"]).all()


def test_direct_force_md_rejects_energy_decomposition():
    with pytest.raises(ValueError, match="force_decomp is unavailable"):
        MDRunner(
            _DirectModel(),
            {"ml": {"k": jnp.asarray(0.5, dtype=jnp.float32)}},
            {"force_decomp": True},
        )
