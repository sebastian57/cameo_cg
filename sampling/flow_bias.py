"""Phase 2: the acquisition bias with the flow standing in for the KDE reference density.

THE FORM (settled in FLOW_ACQUISITION_FORM_DECISION.md)

    q_acq(z) = (1 - lambda) * p(z) + lambda * r(z)
    V(z)     = -kT log[ q_acq(z) / p(z) ] = -kT log[ (1 - lambda) + lambda * r(z)/p(z) ]

`p` is the reference density: the KDE today, the flow here. The algebra above is exact for BOTH,
because the artifact stores `target_weights = (1-lambda)*p_k + lambda*r_k` exactly, and a KDE is
linear in its weights:

    pi(z) = sum_k [(1-l) p_k + l r_k] K(z-c_k) = (1-l) p_KDE(z) + l r_KDE(z)

so `V_KDE = -kT log(pi/p_KDE)` is already `-kT log[(1-l) + l r/p]`. Swapping `p_KDE -> p_theta`
therefore changes the density representation and NOTHING ELSE, which is the whole point of the
comparison. Same lambda, same r, same kT.

`r` IS TRANSITION RELEVANCE ONLY
    The stored `reference_exploration` is `(sparsity + transition)/2`. Sparsity is excluded here:
    it is monotone in -p_ref, so combining it with the 1/p denominator counts low density twice,
    which is the mechanism behind the 2026-08-05 anti-density failure (an 85%-transition rebuild
    still correlated +0.9907 with the original map and still pulled hardest into alphaL).
    Transition relevance is safe because enriching a transition region does not distort basin
    ratios -- the mean force there points outward into the basins.

NORMALISATION MATTERS HERE IN A WAY IT DID NOT BEFORE
    `SmoothTICABias._log_density_and_gradient` uses an UNNORMALISED Gaussian kernel. That is
    harmless for `pi/p_KDE`, where the constant cancels. It is NOT harmless for `r/p_theta`: the
    flow is a normalised density, so `r` must be normalised too or the ratio is wrong by
    (2 pi)^(d/2) |h| -- a factor of ~10 here, i.e. ~1.4 kcal/mol of spurious bias depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["transition_weights", "kde_density_and_grad", "AcquisitionBias"]


def transition_weights(npz_path) -> np.ndarray:
    """Per-centre weights for `r`, from the TRANSITION component alone, summing to 1.

    `reference_transition_component` lives on the full 30x30 cell grid; its support turns out to
    be entirely inside the occupied cells (230 nonzero, all occupied), so restricting to
    `bias.centers` loses nothing and introduces no grid holes.
    """
    # read the artifact directly: SmoothTICABias.load keeps only the KDE-level fields, not the
    # 900-cell component arrays this needs
    d = np.load(npz_path, allow_pickle=False)
    tr = np.asarray(d["reference_transition_component"], dtype=np.float64)
    counts = np.asarray(d["reference_counts"], dtype=np.float64)
    occ = np.where(counts > 0)[0]
    w = tr[occ]
    if w.sum() <= 0:
        raise ValueError("transition component is empty on occupied cells")
    return w / w.sum()


def kde_density_and_grad(z: np.ndarray, centers: np.ndarray, weights: np.ndarray,
                         bandwidth: np.ndarray):
    """NORMALISED Gaussian KDE and its gradient, vectorised over `z` (n, d).

    Returns `(density, grad_density)`. Normalised so `\\int rho dz = 1` when `weights` sums to 1
    -- see the module docstring for why that matters here and not in the incumbent code.
    """
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    h = np.asarray(bandwidth, dtype=np.float64)
    norm = 1.0 / ((2.0 * np.pi) ** (0.5 * z.shape[1]) * np.prod(h))
    out_rho = np.empty(len(z))
    out_grad = np.empty((len(z), z.shape[1]))
    # chunked: (n_z, n_centres, d) is 470x the grid otherwise
    for s in range(0, len(z), 4096):
        d = z[s:s + 4096, None, :] - centers[None, :, :]          # (m, k, d)
        k = np.exp(-0.5 * np.sum((d / h) ** 2, axis=2)) * weights  # (m, k)
        out_rho[s:s + 4096] = k.sum(1) * norm
        out_grad[s:s + 4096] = np.einsum("mk,mkd->md", k, -d / h ** 2) * norm
    return out_rho, out_grad


@dataclass
class AcquisitionBias:
    """`V(z) = -kT log[(1-lambda) + lambda * r(z)/p_eff(z)]` with a DIFFERENTIABLE density floor.

    THE PROBLEM THE FLOOR SOLVES
        With `u = lambda*r/p`, the exact gradient is

            grad V = -kT * D(u) * (grad log r - grad log p),   D(u) = u / ((1-lambda) + u)

        `D` is the whole protection: it tends to 0 in the bulk (u << 1) and to 1 as u grows.
        So the `(1-lambda)` term stops protecting exactly where `p` collapses, and in that
        limit `V -> +kT log p + const`, i.e. the bias becomes MINUS the free energy -- a well
        wherever density is low. Measured on the ala2 artifact: V reached -6.31 kcal/mol at
        (-0.02, 0.56), a cell with ZERO reference frames, where p_flow = 5.5e-07 against
        p_KDE = 9.3e-02, a factor 168,000.

        Lowering `lambda` does NOT help: in that limit lambda enters only as the constant
        `kT log(1/lambda)`, so grad V is lambda-INDEPENDENT. Measured: a 25x cut in lambda
        moved V_min by 1.91 kcal/mol against 1.906 predicted, and left the force nearly intact.

    WHY THE KDE NEVER HAD THIS
        `r` and `p_KDE` share a kernel and centres, so far from the data both are dominated by
        the same nearest centre and `r/p_KDE -> w_r/w_p`, a FINITE constant (max 76.4 here,
        i.e. a structural depth bound of -1.77 kcal/mol). A flow's tails are unrelated to the
        KDE bandwidth, so the ratio is unbounded. The root cause is a mismatch of smoothing
        scales between numerator and denominator, not the flow being wrong.

    THE FLOOR
        Bound `u` by bounding `1/p`. ADDING rather than clipping keeps everything smooth:

            p_eff = p_theta + p_floor    =>    u <= lambda*max(r)/p_floor

        so `D` is bounded away from 1 and `V >= -kT log[(1-lambda) + lambda*max(r)/p_floor]`.
        An earlier version clipped `p` at a hard threshold; that is NOT differentiable and is
        replaced by this.

    MODES
        "constant" p_eff = p_theta + p_c, `p_c` calibrated from a target depth. KDE-FREE.
                   THE DEFAULT. A floor must be INERT where there is data and DOMINANT in the
                   tail, and only a constant has that shape here:

                       p_c/p_theta   = 1.4e-02 on supported cells,  3.5e+04 in the tail
                       p_KDE/p_theta = 9.0e-01 on supported cells,  1.2e-10 in the tail

                   so `p_c` perturbs the bias by a median 0.0005 kcal/mol on support and
                   NOTHING above 0.05, while the KDE floor perturbs it by 0.019 and exceeds
                   0.05 over 24.4% of the supported region.
        "kde"      p_eff = p_theta + p_KDE. Rejected as the default: `p_KDE` is the same SIZE
                   as `p_theta` where data exists -- both estimate the same density there -- so
                   it roughly halves `u` and DILUTES the bias exactly where the bias is meant
                   to act, then collapses (1.2e-10) exactly where a floor is needed. It looks
                   better on aggregate deep-cell counts only because the dilution suppresses
                   everything, wanted or not. Kept for comparison and because its adaptive tail
                   does give a lower |F|max (13.1 vs 20.7 inside bounds).
        "none"     unprotected, for comparison only. Do not sample with it.
    """

    density: object                 # callable(z) -> (rho, grad_rho); the flow
    centers: np.ndarray
    r_weights: np.ndarray
    bandwidth: np.ndarray
    lam: float
    kbt: float
    floor_mode: str = "constant"    # "constant" (default, KDE-free) | "kde" | "none"
    floor_density: object = None    # callable, required for floor_mode="kde"
    floor_constant: float = 0.0     # used by floor_mode="constant"
    target_depth_kcal: float = 1.77  # calibrates floor_constant; default = the KDE's own bound

    def calibrate(self, z_grid: np.ndarray) -> float:
        """Set `floor_constant` from the target depth. Returns it (0 for the other modes).

        `V >= -kT log[(1-l) + l*max(r)/p_c)]`, so
            p_c = l*max(r) / (exp(D/kT) - (1-l)).
        `max(r)` is taken over the evaluation grid, which is where the bias will be tabulated.
        """
        if self.floor_mode != "constant":
            return 0.0
        r, _ = kde_density_and_grad(z_grid, self.centers, self.r_weights, self.bandwidth)
        denom = np.exp(self.target_depth_kcal / self.kbt) - (1.0 - self.lam)
        self.floor_constant = float(self.lam * r.max() / denom)
        return self.floor_constant

    def _floor(self, z, n):
        if self.floor_mode == "kde":
            if self.floor_density is None:
                raise ValueError("floor_mode='kde' needs floor_density")
            return self.floor_density(z)
        if self.floor_mode == "constant":
            return np.full(n, self.floor_constant), np.zeros((n, z.shape[1]))
        if self.floor_mode == "none":
            return np.zeros(n), np.zeros((n, z.shape[1]))
        raise ValueError(f"unknown floor_mode {self.floor_mode!r}")

    def depth_bound(self, z_grid: np.ndarray) -> float:
        """Guaranteed lower bound on V, in kcal/mol. `-inf` for floor_mode='none'."""
        if self.floor_mode == "none":
            return -np.inf
        r, _ = kde_density_and_grad(z_grid, self.centers, self.r_weights, self.bandwidth)
        if self.floor_mode == "constant":
            u_max = self.lam * r.max() / self.floor_constant
        else:
            pf, _ = self.floor_density(z_grid)
            u_max = float((self.lam * r / np.maximum(pf, 1e-300)).max())
        return float(-self.kbt * np.log((1.0 - self.lam) + u_max))

    def energy_gradient(self, z: np.ndarray):
        """Returns `(V, dV/dz)` in kcal/mol and kcal/mol per TICA unit. Smooth everywhere."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        p, gp = self.density(z)
        pf, gpf = self._floor(z, len(z))
        p_eff, gp_eff = p + pf, gp + gpf

        r, gr = kde_density_and_grad(z, self.centers, self.r_weights, self.bandwidth)
        u = self.lam * r / p_eff
        s = (1.0 - self.lam) + u
        V = -self.kbt * np.log(s)
        # d/dz[(1-l) + l r/p] = l (grad_r/p - r grad_p/p^2)
        ds = self.lam * (gr / p_eff[:, None] - (r / p_eff ** 2)[:, None] * gp_eff)
        dV = -self.kbt * ds / s[:, None]
        return V, dV


def flow_density_fn(params, cfg):
    """Adapter: flow -> `(rho, grad_rho)` in the same convention as `kde_density_and_grad`."""
    import jax.numpy as jnp
    from sampling.flow_density import log_prob_and_grad

    def fn(z):
        zz = jnp.asarray(np.atleast_2d(np.asarray(z, np.float32)))
        lp, glp = log_prob_and_grad(params, cfg, zz)
        lp = np.asarray(lp, np.float64)
        rho = np.exp(lp)
        # grad(rho) = rho * grad(log rho)
        return rho, rho[:, None] * np.asarray(glp, np.float64)

    return fn


def kde_density_fn(bias):
    """Adapter for the incumbent reference density, NORMALISED so it is comparable to the flow."""
    centers = np.asarray(bias.centers, np.float64)
    w = np.asarray(bias.reference_weights, np.float64)
    h = np.asarray(bias.bandwidth, np.float64)
    return lambda z: kde_density_and_grad(z, centers, w / w.sum(), h)
