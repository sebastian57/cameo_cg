"""Rational-quadratic spline normalizing flow over frozen TICA coordinates.

WHY THIS EXISTS
    The acquisition bias is `V(z) = -kT log(q_acq(z)/p_ref(z))`, and `p_ref` is currently a
    KDE over occupied grid-cell centres (`biases/tica_regional.py`). That representation is
    tied to a grid resolution, a bandwidth and a set of explicit kernel centres, and the grid
    dies combinatorially past two TICs: at 30 cells/axis, d=2 is 900 cells but d=4 is 810,000.

    This module replaces `p_ref` with a trained continuous density `p_theta(z)`: one globally
    smooth function `z -> log p_theta(z)` with exact likelihoods and analytic gradients, and
    no grid.

DIRECTION CONVENTION
    The flow is written in the NORMALIZING direction, `u = f(z)`, not the generative direction.
    Density evaluation then needs only the forward pass:

        log p_theta(z) = log N(f(z); 0, I) + log|det df/dz|

    so no numerical inversion is ever required for the bias. `sample()` is not implemented
    because nothing in the acquisition pipeline needs to draw from the flow -- it needs the
    density and its gradient. (Inverting an RQ spline is closed-form if that changes.)

WHAT IT IS DELIBERATELY NOT
    Not a molecular generative model. It never sees Cartesian coordinates, only the 2-6
    dimensional frozen TICA projection. Structures continue to come from the atomistic
    enhanced-sampling stack.

CAPACITY
    Keep it small. A flow expressive enough to interpolate finite-sample density fluctuations
    turns those fluctuations into artificial energy corrugations, and the derivative -- not the
    density plot -- is what the MD integrator feels. Prefer a smooth, slightly underfit density.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "FlowConfig",
    "init_flow",
    "log_prob",
    "log_prob_and_grad",
    "train_flow",
    "save_flow",
    "load_flow",
    "FlowDensity",
]

# Numerical floors from Durkan et al. 2019 (Neural Spline Flows). Without the width/height
# floors a softmax can collapse a bin to zero measure and the spline derivative blows up;
# without the derivative floor the transform can become non-monotonic under float32.
MIN_BIN_WIDTH = 1e-3
MIN_BIN_HEIGHT = 1e-3
MIN_DERIVATIVE = 1e-3


@dataclass(frozen=True)
class FlowConfig:
    """Small by default -- see the CAPACITY note in the module docstring."""

    n_dims: int = 2
    n_layers: int = 6            # coupling layers; each transforms half the dims
    n_bins: int = 8              # spline bins per transformed dimension
    hidden: int = 64             # conditioner MLP width
    tail_bound: float = 5.0      # spline acts on [-B, B] in STANDARDISED z; identity outside
    seed: int = 0

    def as_dict(self) -> dict:
        return dict(n_dims=self.n_dims, n_layers=self.n_layers, n_bins=self.n_bins,
                    hidden=self.hidden, tail_bound=self.tail_bound, seed=self.seed)

    @staticmethod
    def from_dict(d: dict) -> "FlowConfig":
        return FlowConfig(n_dims=int(d["n_dims"]), n_layers=int(d["n_layers"]),
                          n_bins=int(d["n_bins"]), hidden=int(d["hidden"]),
                          tail_bound=float(d["tail_bound"]), seed=int(d["seed"]))


def layer_indices(n_dims: int, layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """(conditioning dims, transformed dims) for one coupling layer.

    Derived from the config rather than stored in the parameter tree: a boolean mask living
    inside `params` would be handed to the optimiser, which would try to apply float updates
    to a bool array. Masks are structure, not parameters.
    """
    n_pass = n_dims // 2
    keep_first = (layer % 2 == 0)
    mask = np.zeros(n_dims, dtype=bool)
    if keep_first:
        mask[:n_pass] = True
    else:
        mask[n_pass:] = True
    return np.where(mask)[0], np.where(~mask)[0]


# --------------------------------------------------------------------------------------
# Rational-quadratic spline transform
# --------------------------------------------------------------------------------------

def _rq_spline(x, unnorm_w, unnorm_h, unnorm_d, bound: float):
    """Monotonic rational-quadratic spline on [-bound, bound]; identity outside.

    Returns `(y, logabsdet)` elementwise. `x` is (...,) and the parameter arrays are
    (..., n_bins), (..., n_bins), (..., n_bins + 1).

    Linear tails: the boundary derivatives are pinned to 1 so the spline meets the identity
    outside the interval with a CONTINUOUS derivative. A derivative jump at +/-bound would be
    a force discontinuity in MD, so `bound` must cover the data with margin -- the caller
    standardises z, so +/-5 sigma is generous.
    """
    inside = jnp.abs(x) <= bound
    n_bins = unnorm_w.shape[-1]

    widths = jax.nn.softmax(unnorm_w, axis=-1)
    widths = MIN_BIN_WIDTH + (1.0 - MIN_BIN_WIDTH * n_bins) * widths
    heights = jax.nn.softmax(unnorm_h, axis=-1)
    heights = MIN_BIN_HEIGHT + (1.0 - MIN_BIN_HEIGHT * n_bins) * heights
    derivs = MIN_DERIVATIVE + jax.nn.softplus(unnorm_d)

    cw = jnp.cumsum(widths, axis=-1)
    x_knots = 2.0 * bound * jnp.concatenate([jnp.zeros_like(cw[..., :1]), cw], -1) - bound
    ch = jnp.cumsum(heights, axis=-1)
    y_knots = 2.0 * bound * jnp.concatenate([jnp.zeros_like(ch[..., :1]), ch], -1) - bound

    ones = jnp.ones_like(derivs[..., :1])
    derivs = jnp.concatenate([ones, derivs[..., 1:-1], ones], axis=-1)

    xc = jnp.clip(x, -bound, bound)
    idx = jnp.sum((xc[..., None] >= x_knots).astype(jnp.int32), axis=-1) - 1
    idx = jnp.clip(idx, 0, n_bins - 1)

    take = lambda arr, i: jnp.take_along_axis(arr, i[..., None], axis=-1)[..., 0]
    xk, xk1 = take(x_knots, idx), take(x_knots, idx + 1)
    yk, yk1 = take(y_knots, idx), take(y_knots, idx + 1)
    dk, dk1 = take(derivs, idx), take(derivs, idx + 1)

    w, h = xk1 - xk, yk1 - yk
    s = h / w
    xi = jnp.clip((xc - xk) / w, 0.0, 1.0)
    xi1 = 1.0 - xi

    denom = s + (dk1 + dk - 2.0 * s) * xi * xi1
    y = yk + h * (s * xi * xi + dk * xi * xi1) / denom
    # dy/dx (Durkan et al. eq. 5)
    dnum = s * s * (dk1 * xi * xi + 2.0 * s * xi * xi1 + dk * xi1 * xi1)
    logdet = jnp.log(dnum) - 2.0 * jnp.log(denom)

    return jnp.where(inside, y, x), jnp.where(inside, logdet, 0.0)


# --------------------------------------------------------------------------------------
# Coupling layers
# --------------------------------------------------------------------------------------

def _mlp_apply(mlp, x):
    h = x
    for W, b in mlp[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = mlp[-1]
    return h @ W + b


#: Unnormalised derivative that maps to exactly 1 through `MIN_DERIVATIVE + softplus(.)`.
#: Zero-initialising the whole final layer is NOT enough to make the spline the identity:
#: softmax(0) does give uniform (hence identity) knots, but softplus(0) = 0.693, so the knot
#: derivatives are wrong and the transform bends. Measured: max|u - z| = 0.12 with a plain
#: zero init, 0.0 with this one.
_IDENTITY_DERIV = float(np.log(np.exp(1.0 - MIN_DERIVATIVE) - 1.0))


def _mlp_init(key, n_in: int, hidden: int, n_transform: int, n_bins: int):
    k1, k2 = jax.random.split(key, 2)
    per_dim = 3 * n_bins + 1
    # Final layer weights zero + a structured bias, so every coupling layer starts as the
    # identity and the untrained flow is exactly the standardising Gaussian. A randomly
    # initialised spline stack can start with a wildly wrong Jacobian and never recover.
    final_b = np.zeros((n_transform, per_dim), dtype=np.float32)
    final_b[:, 2 * n_bins:] = _IDENTITY_DERIV          # widths/heights stay 0 -> uniform knots
    return [
        (jax.random.normal(k1, (n_in, hidden)) / np.sqrt(n_in), jnp.zeros(hidden)),
        (jax.random.normal(k2, (hidden, hidden)) / np.sqrt(hidden), jnp.zeros(hidden)),
        (jnp.zeros((hidden, n_transform * per_dim)), jnp.asarray(final_b.ravel())),
    ]


def init_flow(cfg: FlowConfig, z_train: np.ndarray) -> dict:
    """Trainable parameters. Standardisation is FIXED from the training data, not learned.

    `z` comes out of a TICA projection with arbitrary per-axis scale, while the spline lives on
    [-tail_bound, tail_bound]. Standardising first is what makes one `tail_bound` sensible for
    every axis and every system.
    """
    z = np.asarray(z_train, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != cfg.n_dims:
        raise ValueError(f"expected (n, {cfg.n_dims}) training data, got {z.shape}")

    key = jax.random.PRNGKey(cfg.seed)
    mlps = []
    for i in range(cfg.n_layers):
        key, sub = jax.random.split(key)
        pi, ti = layer_indices(cfg.n_dims, i)
        n_in = max(len(pi), 1)                      # d == 1 -> unconditional spline
        mlps.append(_mlp_init(sub, n_in, cfg.hidden, len(ti), cfg.n_bins))

    return dict(mlps=mlps,
                shift=jnp.asarray(z.mean(axis=0)),
                scale=jnp.asarray(z.std(axis=0) + 1e-12))


def _forward(params, cfg: FlowConfig, z):
    """z -> u, the normalizing direction. Returns (u, total logabsdet)."""
    x = (z - params["shift"]) / params["scale"]
    logdet = jnp.broadcast_to(-jnp.sum(jnp.log(params["scale"])), x.shape[:-1])

    for i, mlp in enumerate(params["mlps"]):
        pi, ti = layer_indices(cfg.n_dims, i)
        cond = x[..., pi] if len(pi) else jnp.ones(x.shape[:-1] + (1,))
        raw = _mlp_apply(mlp, cond).reshape(x.shape[:-1] + (len(ti), 3 * cfg.n_bins + 1))
        y, ld = _rq_spline(x[..., ti], raw[..., :cfg.n_bins],
                           raw[..., cfg.n_bins:2 * cfg.n_bins],
                           raw[..., 2 * cfg.n_bins:], cfg.tail_bound)
        x = x.at[..., ti].set(y)
        logdet = logdet + jnp.sum(ld, axis=-1)

    return x, logdet


def log_prob(params, cfg: FlowConfig, z):
    """log p_theta(z). `z` is (..., d)."""
    u, logdet = _forward(params, cfg, z)
    d = u.shape[-1]
    return -0.5 * jnp.sum(u * u, -1) - 0.5 * d * jnp.log(2.0 * jnp.pi) + logdet


def log_prob_and_grad(params, cfg: FlowConfig, z):
    """(log p_theta(z), d log p_theta / dz) -- the two objects the bias needs."""
    g = jax.grad(lambda zz: log_prob(params, cfg, zz).sum())(z)
    return log_prob(params, cfg, z), g


def to_latent(params, cfg: FlowConfig, z):
    """u = f(z). The latent ensemble, for analysis."""
    return _forward(params, cfg, z)[0]


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def train_flow(z: np.ndarray, cfg: FlowConfig, *, steps: int = 4000, batch: int = 4096,
               lr: float = 1e-3, val_fraction: float = 0.1, weight_decay: float = 1e-4,
               report_every: int = 250, swa_frac: float = 0.5, swa_every: int = 10,
               select: str = "swa", log: Callable[[str], None] = print):
    """Maximum-likelihood fit. Returns (params, history).

    SELECTION IS SWA BY DEFAULT, NOT ARGMIN-OF-VALIDATION.
        This flow converges in a few hundred steps and then oscillates on a plateau whose
        WIDTH exceeds the difference between independently seeded runs. Picking the minimum of
        that noisy sequence is therefore close to picking a random point on the plateau, and it
        propagates straight into the score field: two identical invocations of the Phase 1
        driver produced 0.785 and 0.619 for the same gradient-reproducibility statistic, i.e.
        a ~20% swing from checkpoint choice alone.

        Averaging the parameters over the plateau (SWA) removes that degree of freedom, which
        is what makes a capacity sweep interpretable -- otherwise the sweep measures selection
        noise as much as capacity. `select="best"` restores the old behaviour for comparison.

    An overfit flow is still the failure mode to fear (artificial energy corrugations), so the
    held-out NLL of BOTH the SWA and the best-checkpoint parameters is reported.
    """
    import optax

    z = np.asarray(z, dtype=np.float32)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(z))
    n_val = max(1, int(val_fraction * len(z)))
    val, train = z[perm[:n_val]], z[perm[n_val:]]
    log(f"  flow: {len(train)} train / {len(val)} val, d={cfg.n_dims}, "
        f"{cfg.n_layers} layers x {cfg.n_bins} bins, hidden {cfg.hidden}, seed {cfg.seed}")

    params = init_flow(cfg, train)
    nll = lambda p, b: -jnp.mean(log_prob(p, cfg, b))

    opt = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, o, b):
        loss, g = jax.value_and_grad(nll)(p, b)
        updates, o = opt.update(g, o, p)
        return optax.apply_updates(p, updates), o, loss

    val_j = jax.jit(lambda p: nll(p, jnp.asarray(val)))

    history, best, best_val, best_step = [], None, np.inf, -1
    swa_sum, swa_n = None, 0
    swa_start = int(swa_frac * steps)
    draw = np.random.default_rng(cfg.seed + 1)
    for i in range(steps):
        idx = draw.integers(0, len(train), size=min(batch, len(train)))
        params, opt_state, loss = step(params, opt_state, jnp.asarray(train[idx]))
        # accumulate the plateau average
        if i >= swa_start and (i - swa_start) % swa_every == 0:
            swa_sum = (jax.tree_util.tree_map(lambda a: np.asarray(a, np.float64), params)
                       if swa_sum is None else
                       jax.tree_util.tree_map(lambda s, a: s + np.asarray(a, np.float64),
                                              swa_sum, params))
            swa_n += 1
        if (i + 1) % report_every == 0 or i == 0:
            v = float(val_j(params))
            history.append(dict(step=i + 1, train_nll=float(loss), val_nll=v))
            if v < best_val:
                best_val, best_step = v, i + 1
                best = jax.tree_util.tree_map(lambda a: np.asarray(a).copy(), params)
            log(f"    step {i+1:6d}  train {float(loss):8.4f}  val {v:8.4f}"
                f"{'  *' if best_step == i + 1 else ''}")

    swa = jax.tree_util.tree_map(lambda s: jnp.asarray(s / swa_n, jnp.float32), swa_sum) \
        if swa_n else None
    swa_val = float(val_j(swa)) if swa is not None else float("nan")
    best = jax.tree_util.tree_map(jnp.asarray, best)
    log(f"  best-checkpoint val NLL {best_val:.4f} (step {best_step}/{steps}) | "
        f"SWA over {swa_n} snapshots from step {swa_start}: {swa_val:.4f}")

    if select == "swa" and swa is not None:
        out = swa
    elif select == "best":
        out = best
    else:
        raise ValueError(f"select must be 'swa' or 'best', got {select!r}")
    return out, dict(history=history, best_val_nll=best_val, best_step=best_step,
                     swa_val_nll=swa_val, swa_n=swa_n, selected=select)


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------

def save_flow(params: dict, cfg: FlowConfig, path: Path, **extra) -> None:
    """Flat NPZ so the artifact is inspectable without this module."""
    flat = {"shift": np.asarray(params["shift"]), "scale": np.asarray(params["scale"])}
    for li, mlp in enumerate(params["mlps"]):
        for wi, (W, b) in enumerate(mlp):
            flat[f"mlp{li}_W{wi}"] = np.asarray(W)
            flat[f"mlp{li}_b{wi}"] = np.asarray(b)
    for k, v in cfg.as_dict().items():
        flat[f"cfg_{k}"] = np.asarray(v)
    flat.update({k: np.asarray(v) for k, v in extra.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **flat)


def load_flow(path: Path) -> Tuple[dict, FlowConfig]:
    d = np.load(path, allow_pickle=False)
    cfg = FlowConfig.from_dict({k[4:]: d[k].item() for k in d.files if k.startswith("cfg_")})
    mlps = []
    for li in range(cfg.n_layers):
        mlp, wi = [], 0
        while f"mlp{li}_W{wi}" in d.files:
            mlp.append((jnp.asarray(d[f"mlp{li}_W{wi}"]), jnp.asarray(d[f"mlp{li}_b{wi}"])))
            wi += 1
        mlps.append(mlp)
    params = dict(mlps=mlps, shift=jnp.asarray(d["shift"]), scale=jnp.asarray(d["scale"]))
    return params, cfg


@dataclass(frozen=True)
class FlowDensity:
    """Numpy-facing wrapper matching what the bias needs from a density model.

    Mirrors `SmoothTICABias._log_density_and_gradient`'s signature so the KDE reference
    density can be swapped for this one at a single call site (`tica_regional.py:126`).
    """

    params: dict
    cfg: FlowConfig

    @classmethod
    def load(cls, path: Path) -> "FlowDensity":
        p, c = load_flow(path)
        return cls(params=p, cfg=c)

    def log_density_and_gradient(self, z: np.ndarray):
        scalar = np.ndim(z) == 1
        z_in = jnp.asarray(np.atleast_2d(np.asarray(z, dtype=np.float32)))
        lp, g = log_prob_and_grad(self.params, self.cfg, z_in)
        lp, g = np.asarray(lp, np.float64), np.asarray(g, np.float64)
        return (float(lp[0]), g[0]) if scalar else (lp, g)
