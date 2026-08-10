"""MLCG teacher bias: a frozen learned CG potential applied as a NEGATIVE bias.

    U_samp(x) = U_AA(x) - alpha * U_ML(xi(x))

so the atomistic system is pushed OUT of regions the CG teacher already scores as
favourable, into teacher-adversarial territory. alpha is an exploration knob, not a
temperature. Teacher parameters stay frozen.

Force convention: with F_ML = -grad U_ML, the bias contributes F_bias = -alpha * F_ML
(i.e. +alpha * grad U_ML) on the mapped beads.

IMPORTANT: training labels must NOT be the biased forces. Configurations are saved and
re-run through GROMACS without PLUMED to recover f_AA. See
KB DESIGN/MLCG_TEACHER_BIAS_SAMPLING.md "Bias-Free Training Labels".
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import BiasTerm, register_bias

__all__ = ["MLCGTeacherBias", "alpha_at_step"]


def alpha_at_step(step: int, target: float, equilibrate_steps: int, ramp_steps: int) -> float:
    """Cosine ramp: 0 during equilibration, then smoothly up to `target`."""
    step = int(step)
    if step < equilibrate_steps:
        return 0.0
    if ramp_steps <= 0:
        return float(target)
    q = (step - equilibrate_steps) / float(ramp_steps)
    if q >= 1.0:
        return float(target)
    return float(target) * 0.5 * (1.0 - np.cos(np.pi * q))


@register_bias("mlcg_teacher")
class MLCGTeacherBias(BiasTerm):
    """Frozen CG model coupled as a negative bias on the atomistic Hamiltonian.

    Config keys: training_config_path, params_path, alpha, equilibrate_steps,
    ramp_steps, dataset_path (for box construction), name, enabled.
    """

    def __init__(
        self,
        training_config_path: str,
        params_path: str,
        dataset_path: str,
        alpha: float = 1.0,
        equilibrate_steps: int = 0,
        ramp_steps: int = 0,
        name: str | None = None,
        enabled: bool = True,
    ):
        super().__init__(name=name or "mlcg_teacher", enabled=enabled)
        self.training_config_path = str(training_config_path)
        self.params_path = str(params_path)
        self.dataset_path = str(dataset_path)
        self.alpha_target = float(alpha)
        self.equilibrate_steps = int(equilibrate_steps)
        self.ramp_steps = int(ramp_steps)
        self._last_alpha = 0.0
        self._last_energy = 0.0
        self._build()

    # -- model construction mirrors scripts/run_md.py -----------------------------
    def _build(self) -> None:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")  # teacher runs on CPU beside GROMACS
        import sys

        repo = Path(__file__).resolve().parents[2]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from utils.jax_setup import apply_jax_compat_shims

        apply_jax_compat_shims()
        import jax
        import jax.numpy as jnp
        import yaml
        from config.manager import ConfigManager
        from data.preprocessor import CoordinatePreprocessor
        from models.combined_model import CombinedModel

        self._jax = jax
        self._jnp = jnp

        # Dense neighbor list: independent frames overflow a cell list built from one
        # reference frame, silently dropping edges. See KB BUGS/2026-07-31_neighbor-list-*.
        cfg_dict = yaml.safe_load(open(self.training_config_path))
        cfg_dict.setdefault("model", {})["neighbor_disable_cell_list"] = True
        import tempfile

        tf = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(cfg_dict, tf)
        tf.close()
        cfg = ConfigManager(tf.name)

        with np.load(self.dataset_path, allow_pickle=False) as d:
            R = np.asarray(d["R"], np.float32)
            species = np.asarray(d["species"], np.int32)
            mask = (
                np.asarray(d["mask"], np.float32)
                if "mask" in d
                else np.ones(R.shape[:2], np.float32)
            )
        pre = CoordinatePreprocessor(
            cutoff=cfg.get_cutoff(),
            buffer_multiplier=cfg.get_buffer_multiplier(),
            park_multiplier=cfg.get_park_multiplier(),
        )
        box, shift = pre.compute_box_extent(R, mask)
        self._box, self._shift, self._pre = box, shift, pre
        self._n_beads = int(R.shape[1])
        self._mask0 = jnp.asarray(mask[0])
        self._species0 = jnp.asarray(species[0])

        R0 = pre.center_and_park(R[:1], mask[:1], box, shift)[0]
        n_species = max(
            int(species.max()) + 1,
            int(cfg.get("model", "allegro", "num_types", default=0) or 0),
        )
        model = CombinedModel(
            config=cfg,
            R0=jnp.asarray(R0),
            box=box,
            species=self._species0,
            N_max=self._n_beads,
            prior_only=cfg.prior_only_enabled(),
            n_species_override=n_species,
        )
        params = pickle.load(open(self.params_path, "rb"))
        if isinstance(params, dict) and isinstance(params.get("params"), dict):
            params = params["params"]
        self._model, self._params = model, params

        def energy(r):
            return model.compute_energy(params, r, self._mask0, self._species0)

        self._energy_and_grad = jax.jit(jax.value_and_grad(energy))

    # -- BiasTerm -----------------------------------------------------------------
    def n_beads_expected(self) -> int:
        return self._n_beads

    def evaluate(self, positions_A: np.ndarray, step: int):
        jnp = self._jnp
        alpha = alpha_at_step(step, self.alpha_target, self.equilibrate_steps, self.ramp_steps)
        self._last_alpha = alpha
        if alpha == 0.0:
            self._last_energy = 0.0
            return 0.0, np.zeros_like(positions_A)

        # Park into the same frame the model was constructed with, so the neighbor
        # list and box are the ones it expects. Translation does not change forces.
        p = np.asarray(positions_A, dtype=np.float32)[None]
        m = np.asarray(self._mask0, dtype=np.float32)[None]
        parked = self._pre.center_and_park(p, m, self._box, self._shift)[0]

        u_ml, grad_u = self._energy_and_grad(jnp.asarray(parked))
        u_ml = float(u_ml)
        grad_u = np.asarray(grad_u, dtype=np.float64)

        # U_bias = -alpha * U_ML  =>  F_bias = -grad U_bias = +alpha * grad U_ML
        energy = -alpha * u_ml
        forces = alpha * grad_u
        self._last_energy = energy
        return energy, forces

    def diagnostics(self) -> Dict[str, Any]:
        return {
            f"{self.name}_alpha": self._last_alpha,
            f"{self.name}_energy_kcal": self._last_energy,
        }

    def describe(self) -> str:
        return (
            f"{self.name} (mlcg_teacher): {self._n_beads} beads, alpha_target="
            f"{self.alpha_target}, equilibrate={self.equilibrate_steps}, "
            f"ramp={self.ramp_steps}, params={Path(self.params_path).name}"
        )
