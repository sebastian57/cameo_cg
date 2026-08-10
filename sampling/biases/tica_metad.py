"""Well-tempered metadynamics in the frozen TICA space.

    V(z, t) = sum_k h_k * exp(-|z - z_k|^2 / (2 sigma^2))
    h_k     = height * exp(-V(z_k, t_k) / (kB * dT)),   dT = (bias_factor - 1) * T

History-dependent, unlike every other term in this registry: `mlcg_teacher` is a fixed
negative model energy after its ramp and `tica_regional` is a fixed density-ratio
potential. Neither remembers where it has been, so a replica that finds a comfortable
basin can sit in it for the whole run. MetaD fills basins progressively and keeps moving.

WHAT THIS DOES TO THE ENSEMBLE -- read before building a dataset from it
-----------------------------------------------------------------------
Well-tempered MetaD drives the biased-CV distribution toward flatness. TIC 1 separates
the ala2 basins, so this bias WILL equalise basin populations. That is not a defect to
tune away; it is what the method does.

It is therefore only safe under the acquisition/assembly split: use MetaD to propose
frames, and never inherit its distribution. The dataset assembler must restore reference
basin weights (importance weights or subsampling) while keeping transition frames.
Training directly on a MetaD ensemble reproduces the 2026-08-05 failure, where a 5.6x
alphaL over-representation became a ~27x over-population in CG MD.

Well-tempered rather than plain MetaD specifically so the deposited bias converges
instead of growing without bound: hill heights decay where bias has accumulated, and
V -> -(1 - 1/gamma) * F, so a finite bias factor limits how far the ensemble is flattened.

Sigma guidance
--------------
Hill width should be a fraction of the reference spread in each TIC, not a guess. For
ala2 bb6 on the current artifact the reference gives TIC1 std 0.763 and TIC2 std 0.473,
so sigma ~ [0.15, 0.09] (about std/5) resolves basins without hills that leak across
the whole map.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from .base import BiasTerm, register_bias
from .tica_regional import SmoothTICABias

KB_KCAL = 0.0019872042586


@register_bias("tica_metad")
class TICAWellTemperedMetaD(BiasTerm):
    def __init__(self, bias_npz: str, height: float = 0.15,
                 sigma: Sequence[float] = (0.15, 0.09), pace: int = 500,
                 bias_factor: float = 8.0, temperature: float = 298.0,
                 equilibrate_steps: int = 0, hills_path: str | None = None,
                 max_hills: int = 200000, name: str = "tica_metad",
                 enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        # Reuse the persisted projection so MetaD and `tica_regional` are guaranteed to
        # live in the SAME coordinates; refitting here would silently decouple them.
        self.projection = SmoothTICABias.load(bias_npz).projection
        self.height = float(height)
        self.sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        self.pace = int(pace)
        self.bias_factor = float(bias_factor)
        self.temperature = float(temperature)
        self.kT = KB_KCAL * self.temperature
        self.equilibrate_steps = int(equilibrate_steps)
        self.hills_path = Path(hills_path) if hills_path else None
        self.max_hills = int(max_hills)

        self._validate()
        self._centers = np.zeros((0, len(self.sigma)), dtype=np.float64)
        self._heights = np.zeros((0,), dtype=np.float64)
        self._next_deposit = self.equilibrate_steps
        self.last_z = np.full(len(self.sigma), np.nan)
        self.last_bias = 0.0
        if self.hills_path and self.hills_path.exists():
            self._load_hills()

    def _validate(self) -> None:
        if self.sigma.size != 2:
            raise ValueError(f"{self.name}: sigma must have 2 entries (TIC1, TIC2), "
                             f"got {self.sigma.tolist()}")
        if not np.all(np.isfinite(self.sigma)) or np.any(self.sigma <= 0):
            raise ValueError(f"{self.name}: sigma must be positive and finite")
        if not np.isfinite(self.height) or self.height <= 0:
            raise ValueError(f"{self.name}: height must be positive, got {self.height}")
        if self.pace < 1:
            raise ValueError(f"{self.name}: pace must be >= 1, got {self.pace}")
        if self.bias_factor <= 1.0:
            raise ValueError(
                f"{self.name}: bias_factor must be > 1 for well-tempered MetaD "
                f"(got {self.bias_factor}); 1.0 would mean no tempering at all")
        if self.equilibrate_steps < 0:
            raise ValueError(f"{self.name}: equilibrate_steps must be >= 0")

    # ---------------- hills ----------------
    def _bias_and_grad(self, z: np.ndarray) -> Tuple[float, np.ndarray]:
        """V(z) and dV/dz from the deposited hills."""
        if len(self._centers) == 0:
            return 0.0, np.zeros_like(z)
        d = (z[None, :] - self._centers) / self.sigma[None, :]
        g = self._heights * np.exp(-0.5 * np.sum(d * d, axis=1))
        V = float(g.sum())
        # dV/dz = sum_k g_k * -(z - z_k)/sigma^2
        dV = -np.sum(g[:, None] * d / self.sigma[None, :], axis=0)
        return V, dV

    def _deposit(self, z: np.ndarray) -> None:
        if len(self._centers) >= self.max_hills:
            return
        V, _ = self._bias_and_grad(z)
        dT = (self.bias_factor - 1.0) * self.temperature
        h = self.height * np.exp(-V / (KB_KCAL * dT))
        self._centers = np.vstack([self._centers, z[None, :]])
        self._heights = np.concatenate([self._heights, [h]])

    def _load_hills(self) -> None:
        with np.load(self.hills_path) as d:
            self._centers = np.asarray(d["centers"], dtype=np.float64)
            self._heights = np.asarray(d["heights"], dtype=np.float64)
        print(f"[{self.name}] restored {len(self._heights)} hills from {self.hills_path}")

    def save_hills(self) -> None:
        if not self.hills_path:
            return
        self.hills_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.hills_path, centers=self._centers, heights=self._heights,
                 sigma=self.sigma, height=self.height, bias_factor=self.bias_factor,
                 temperature=self.temperature, pace=self.pace)

    # ---------------- interface ----------------
    def evaluate(self, positions_A: np.ndarray, step: int) -> Tuple[float, np.ndarray]:
        z_all, jac = self.projection.value_and_jacobian(positions_A)
        z = np.asarray(z_all, dtype=np.float64)[: len(self.sigma)]

        # Deposit on schedule. `step` advances by RECOMPUTE_STRIDE, so catch up rather
        # than testing step % pace == 0, which would silently skip every deposition when
        # pace is not a multiple of the stride.
        if step >= self._next_deposit:
            self._deposit(z)
            self._next_deposit = max(step + 1, self._next_deposit + self.pace)

        V, dVdz = self._bias_and_grad(z)
        # chain rule: dV/dR = sum_k dV/dz_k * dz_k/dR   (jac is (n_tics, n_beads, 3))
        forces = -np.einsum("k,kij->ij", dVdz, np.asarray(jac)[: len(self.sigma)])
        self.last_z, self.last_bias = z, V
        return float(V), forces

    def n_beads_expected(self) -> int | None:
        return self.projection.n_beads

    def diagnostics(self) -> dict:
        eff = (float(self._heights[-1] / self.height) if len(self._heights) else 1.0)
        return {"tic1": float(self.last_z[0]) if np.isfinite(self.last_z[0]) else None,
                "tic2": float(self.last_z[1]) if np.isfinite(self.last_z[1]) else None,
                "bias_kcal": self.last_bias, "n_hills": int(len(self._heights)),
                "last_hill_height_frac": eff}
