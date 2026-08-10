"""Local inversion umbrella bias -- probes the parity direction TICA cannot see.

    U_inv(z) = k/2 * [chi(z) - chi_target]^2

with chi the normalised signed volume of a 3-branch center (see
`mapping.normalized_signed_volume`). One fixed `chi_target` per replica gives a
standard umbrella ladder from the physical value toward the planar region chi = 0.

Why this bias exists
--------------------
The pair-distance TICA representation is reflection-invariant: a structure and its
mirror occupy identical TICA coordinates, so the TICA bias has no direction in which
to push to induce inversion. chi is parity-ODD and supplies exactly that direction.

What it can and cannot teach
----------------------------
Allegro's readout keeps only even-parity scalars, so U(mirror R) == U(R) exactly and
**no data can make one enantiomer lower in energy than the other**. What a
parity-invariant model *can* represent is the barrier, since |chi| and the barrier
height are parity-EVEN. This bias generates labels in precisely that region -- the
transition data the training set lacks -- and nothing else.

Scope caveat: in amber99sb-ildn there is **no improper dihedral on CA**. Chirality
there is held up by angle terms and sterics alone, so driving chi -> 0 is umbrella
inversion (CA pushed into the plane of N/CB/C) against bending terms. The barrier
these labels encode is therefore a force-field construct; real racemization proceeds
by deprotonation, a mechanism no classical FF contains. Reproducing the AA model is
the goal here, so that is correct -- but do not read these labels as racemization
physics.

Target ramp
-----------
A window far from equilibrium would otherwise start with a huge force (target 0 is
~13.5 sigma from the reference chi of -0.711; at k=2000 that is ~950 kcal/mol/A on
the first step). The target therefore moves from the measured initial chi to
`chi_target` over `ramp_steps`, then is held FIXED -- so sampling is stationary
afterwards and the window is a genuine umbrella, not a steered pull.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ..mapping import get_mapping
from .base import BiasTerm, register_bias


@register_bias("local_inversion_umbrella")
class LocalInversionUmbrella(BiasTerm):
    def __init__(self, mapping: str, chi_target: float, force_constant: float = 2000.0,
                 center: int | None = None, neighbors: Sequence[int] | None = None,
                 equilibrate_steps: int = 0, ramp_steps: int = 0,
                 name: str = "local_inversion_umbrella", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        m = get_mapping(mapping)

        if center is None:
            cands = m.inversion_centers()
            if len(cands) != 1:
                raise ValueError(
                    f"{mapping}: expected exactly one inversion center for auto "
                    f"selection, found {sorted(cands)}; set `center` explicitly"
                )
            center, auto_nb = next(iter(cands.items()))
            neighbors = neighbors if neighbors is not None else auto_nb
        if neighbors is None or len(neighbors) != 3:
            raise ValueError("need exactly three neighbour beads")

        self.center = int(center)
        self.neighbors = tuple(int(n) for n in neighbors)
        self.k = float(force_constant)
        self.chi_target = float(chi_target)
        self.equilibrate_steps = int(equilibrate_steps)
        self.ramp_steps = int(ramp_steps)
        self._n_beads = m.n_beads
        self._validate(m)
        self._chi_start: float | None = None
        self.last_chi = float("nan")
        self.last_target = float("nan")

    def _validate(self, m) -> None:
        """Reject bad configurations at construction, i.e. at server startup.

        Every one of these would otherwise surface only after GROMACS has launched
        and the socket handshake has completed -- a queue slot spent to learn that a
        bead index was out of range.
        """
        n = m.n_beads
        idx = (self.center,) + self.neighbors
        if len(set(idx)) != 4:
            raise ValueError(
                f"{self.name}: center and neighbours must be four distinct beads, got "
                f"center={self.center} neighbours={self.neighbors}"
            )
        for b in idx:
            if not (0 <= b < n):
                raise ValueError(
                    f"{self.name}: bead index {b} out of range for mapping "
                    f"{m.name} with {n} beads"
                )
        if m.bonds:
            bonded = set(m.neighbors(self.center))
            missing = [b for b in self.neighbors if b not in bonded]
            if missing:
                raise ValueError(
                    f"{self.name}: bead(s) {missing} are not bonded to center "
                    f"{self.center} in mapping {m.name} (bonded: {sorted(bonded)}). "
                    f"chi over non-bonded branches is not an inversion coordinate."
                )
        if not np.isfinite(self.k) or self.k <= 0:
            raise ValueError(f"{self.name}: force_constant must be positive and finite, got {self.k}")
        if not np.isfinite(self.chi_target) or not (-1.0 <= self.chi_target <= 1.0):
            raise ValueError(
                f"{self.name}: chi_target must be finite and within [-1, 1] (chi is a "
                f"normalised signed volume), got {self.chi_target}"
            )
        if self.equilibrate_steps < 0 or self.ramp_steps < 0:
            raise ValueError(
                f"{self.name}: equilibrate_steps and ramp_steps must be >= 0, got "
                f"{self.equilibrate_steps} and {self.ramp_steps}"
            )

    # -- chi and its Cartesian gradient ------------------------------------
    def _chi_and_grad(self, R: np.ndarray) -> Tuple[float, np.ndarray]:
        c, (i, j, k) = self.center, self.neighbors
        a = R[i] - R[c]
        b = R[j] - R[c]
        d = R[k] - R[c]
        na, nb, nd = (float(np.linalg.norm(v)) for v in (a, b, d))
        if min(na, nb, nd) < 1e-6:
            # chi is undefined at zero branch length and its gradient diverges; a
            # silent nan here would propagate into the forces PLUMED applies.
            raise FloatingPointError(
                f"{self.name}: degenerate branch length ({na:.2e}, {nb:.2e}, {nd:.2e} A) "
                f"at center {self.center} -- structure has collapsed"
            )
        cross_ab = np.cross(a, b)
        num = float(np.dot(cross_ab, d))
        den = na * nb * nd
        chi = num / den

        # d(num) wrt each branch vector
        dnum_da = np.cross(b, d)
        dnum_db = np.cross(d, a)
        dnum_dd = cross_ab
        # chi = num/den ; d(chi)/da = dnum_da/den - num/den^2 * d(den)/da
        # d(den)/da = (a/na) * nb * nd
        g_a = dnum_da / den - num / den**2 * (a / na) * nb * nd
        g_b = dnum_db / den - num / den**2 * (b / nb) * na * nd
        g_d = dnum_dd / den - num / den**2 * (d / nd) * na * nb

        grad = np.zeros_like(R)
        grad[i] += g_a
        grad[j] += g_b
        grad[k] += g_d
        grad[c] -= g_a + g_b + g_d      # center appears in all three branches
        return chi, grad

    def _target(self, step: int) -> float:
        if self._chi_start is None:
            return self.chi_target
        if step < self.equilibrate_steps:
            return self._chi_start
        t = step - self.equilibrate_steps
        if self.ramp_steps > 0 and t < self.ramp_steps:
            f = t / self.ramp_steps
            return (1.0 - f) * self._chi_start + f * self.chi_target
        return self.chi_target

    def evaluate(self, positions_A: np.ndarray, step: int) -> Tuple[float, np.ndarray]:
        chi, grad = self._chi_and_grad(positions_A)
        if self._chi_start is None:
            self._chi_start = float(chi)
        target = self._target(step)
        self.last_chi, self.last_target = float(chi), float(target)

        delta = chi - target
        energy = 0.5 * self.k * delta**2
        forces = -self.k * delta * grad      # F = -dU/dz
        return float(energy), forces

    def n_beads_expected(self) -> int | None:
        # The mapping is known at construction, so report its bead count rather than
        # None -- returning None silently disables the server's bead-count check and
        # a mapping/plugin mismatch would surface as misread positions, not an error.
        return self._n_beads

    def diagnostics(self) -> dict:
        """chi vs its target every report interval -- a window pinned at the edge of
        its target is the signature of an unreachable window, and is only visible here."""
        return {"chi": self.last_chi, "chi_target": self.last_target,
                "chi_error": self.last_chi - self.last_target}
