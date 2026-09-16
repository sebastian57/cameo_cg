"""Build a small outer-radius and negative-mode control state file.

The output is intentionally separate from production v4 states. It preserves the v4 reference
anchor provenance so ``build_stencil_campaign_v4.py`` can use it for a small optional pilot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from analysis.physics.v3_v4.common import write_json


def build_control_metadata(
    states: dict[str, np.ndarray],
    anchor_slots: np.ndarray,
    path_fractions: list[float],
    fixed_eps: list[float],
) -> dict[str, np.ndarray]:
    """Create path-control rows from existing v4 direction geometry.

    This pure function is also used by the schema test. Model-derived fixed negative modes are
    appended by the CLI after this deterministic path-control block.
    """
    R = np.asarray(states["R"], dtype=np.float64)
    slot = np.asarray(states["anchor"], dtype=np.int64)
    direction = np.asarray(states["direction"], dtype=np.int8)
    multiplier = np.asarray(states.get("multiplier", np.zeros(len(R))), dtype=np.float64)
    source = np.asarray(states.get("anchor_source", np.zeros(len(R), dtype=np.int8)), dtype=np.int8)
    index = np.asarray(states.get("anchor_index", np.arange(len(R))), dtype=np.int64)
    def provenance(values: np.ndarray, row: int, anchor: int, default: int) -> int:
        # Production v4 writes these arrays once per anchor; synthetic/control tables may
        # repeat them once per state. Support both layouts without changing the production file.
        if len(values) == len(R):
            return int(values[row])
        if 0 <= anchor < len(values):
            return int(values[anchor])
        return int(default)

    anchor_rows = {int(s): i for i, s in enumerate(slot) if direction[i] < 0}
    selected = [int(s) for s in np.asarray(anchor_slots, dtype=np.int64)]
    rows: list[np.ndarray] = []
    meta: dict[str, list[Any]] = {
        "anchor": [], "direction": [], "multiplier": [], "anchor_source": [],
        "anchor_index": [], "control_kind": [], "mode_sign": [], "path_fraction": [],
        "fixed_eps_A": [], "eps_state": [], "lam_state": [], "kind": [],
    }

    for s in selected:
        if s not in anchor_rows:
            continue
        ai = anchor_rows[s]
        anchor_R = R[ai]
        rows.append(anchor_R)
        for key, value in (
            ("anchor", s), ("direction", -1), ("multiplier", 0.0),
            ("anchor_source", provenance(source, ai, s, 0)),
            ("anchor_index", provenance(index, ai, s, s)),
            ("control_kind", "anchor"), ("mode_sign", 0), ("path_fraction", 0.0),
            ("fixed_eps_A", np.nan), ("eps_state", 0.0), ("lam_state", 0.0), ("kind", 0),
        ):
            meta[key].append(value)

        direction_rows: dict[int, int] = {}
        for i in np.flatnonzero((slot == s) & (direction >= 0) & (multiplier > 0)):
            direction_rows.setdefault(int(direction[i]), int(i))
        for d, pos_i in sorted(direction_rows.items()):
            delta = R[pos_i] - anchor_R
            base_eps = abs(float(states.get("eps_state", np.zeros(len(R)))[pos_i]))
            lam = float(states.get("lam_state", np.zeros(len(R)))[pos_i])
            kind = int(states.get("kind", np.zeros(len(R), dtype=np.int8))[pos_i])
            for fraction in path_fractions:
                for sign in (1.0, -1.0):
                    rows.append(anchor_R + sign * fraction * delta)
                    values = {
                        "anchor": s, "direction": d, "multiplier": sign * fraction,
                        "anchor_source": provenance(source, ai, s, 0),
                        "anchor_index": provenance(index, ai, s, s),
                        "control_kind": "radius_path", "mode_sign": int(sign),
                        "path_fraction": fraction, "fixed_eps_A": np.nan,
                        "eps_state": base_eps, "lam_state": lam, "kind": kind,
                    }
                    for key in meta:
                        meta[key].append(values[key])

    if not rows:
        raise ValueError("no selected anchors with direction==-1 rows")
    out = {"R": np.asarray(rows, dtype=np.float32)}
    out.update({key: np.asarray(value) for key, value in meta.items()})
    return out


def _rigid_basis(R: np.ndarray) -> np.ndarray:
    centered = R - R.mean(axis=0)
    basis = np.zeros((6, R.size), dtype=np.float64)
    for k in range(3):
        basis[k, k::3] = 1.0
    basis[3, 1::3] = -centered[:, 2]
    basis[3, 2::3] = centered[:, 1]
    basis[4, 0::3] = centered[:, 2]
    basis[4, 2::3] = -centered[:, 0]
    basis[5, 0::3] = -centered[:, 1]
    basis[5, 1::3] = centered[:, 0]
    return np.linalg.qr(basis.T)[0]


def _append_fixed_modes(
    out: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    selected: np.ndarray,
    hessian_fn,
    fixed_eps: list[float],
    lam_floor: float,
    n_modes: int,
) -> dict[str, np.ndarray]:
    rows = [np.asarray(value) for value in out["R"]]
    meta = {key: [value for value in np.asarray(out[key])] for key in out if key != "R"}
    anchor_rows = {int(s): i for i, s in enumerate(states["anchor"]) if states["direction"][i] < 0}
    source = np.asarray(states.get("anchor_source", np.zeros(len(states["R"]), dtype=np.int8)))
    index = np.asarray(states.get("anchor_index", np.arange(len(states["R"]), dtype=np.int64)))

    def provenance(values: np.ndarray, row: int, anchor: int, default: int) -> int:
        if len(values) == len(states["R"]):
            return int(values[row])
        if 0 <= anchor < len(values):
            return int(values[anchor])
        return int(default)

    for s in selected:
        if int(s) not in anchor_rows:
            continue
        ai = anchor_rows[int(s)]
        R0 = np.asarray(states["R"][ai], dtype=np.float64)
        H = np.asarray(hessian_fn(R0), dtype=np.float64).reshape(R0.size, R0.size)
        Q = _rigid_basis(R0)
        P = np.eye(R0.size) - Q @ Q.T
        Hi = 0.5 * (P @ H @ P + (P @ H @ P).T)
        eig, vec = np.linalg.eigh(Hi)
        candidates = np.flatnonzero(eig <= lam_floor)
        candidates = candidates[np.argsort(eig[candidates])][:n_modes]
        for mode in candidates:
            v = vec[:, mode].reshape(R0.shape)
            v /= np.linalg.norm(v)
            for eps in fixed_eps:
                for sign in (1.0, -1.0):
                    rows.append(R0 + sign * eps * v)
                    values = {
                        "anchor": int(s), "direction": int(mode), "multiplier": sign,
                        "anchor_source": provenance(source, ai, int(s), 0),
                        "anchor_index": provenance(index, ai, int(s), int(s)),
                        "control_kind": "fixed_low_or_negative_mode", "mode_sign": int(sign),
                        "path_fraction": np.nan, "fixed_eps_A": eps, "eps_state": eps,
                        "lam_state": float(eig[mode]), "kind": 3,
                    }
                    for key in meta:
                        meta[key].append(values[key])
    if len(rows) == len(out["R"]):
        return out
    result = {"R": np.asarray(rows, dtype=np.float32)}
    result.update({key: np.asarray(value) for key, value in meta.items()})
    return result


def _geometry_ok(R: np.ndarray, max_bond: float, min_pair: float) -> bool:
    bond = np.linalg.norm(R[1:] - R[:-1], axis=-1)
    pair = np.linalg.norm(R[:, None] - R[None, :], axis=-1)
    pair = pair[np.triu_indices(len(R), 1)]
    return bool(bond.max() <= max_bond and pair.min() >= min_pair)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--model-params", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-anchors", type=int, default=64)
    parser.add_argument("--path-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--fixed-eps", type=float, nargs="+", default=[0.04, 0.08, 0.16])
    parser.add_argument("--label-se", type=float, default=3.615)
    parser.add_argument("--snr-target", type=float, default=3.0)
    parser.add_argument("--n-modes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-bond", type=float, default=3.0)
    parser.add_argument("--min-pair", type=float, default=1.0)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from analysis.md.analyze_model_residuals_by_region import _load_model
    from sampling.mapping import get_mapping
    from analysis.physics.v3_v4.common import load_state_table

    states = load_state_table(args.states)
    anchor_slots = np.unique(states["anchor"][states["direction"] < 0])
    rng = np.random.default_rng(args.seed)
    if len(anchor_slots) > args.n_anchors:
        anchor_slots = np.sort(rng.choice(anchor_slots, args.n_anchors, replace=False))
    out = build_control_metadata(states, anchor_slots, args.path_fractions, args.fixed_eps)

    mapping = get_mapping("ala2_backbone_cb_6")
    model, params, mask0, species0 = _load_model(
        args.model_config, args.model_params, str(args.reference), mapping.n_beads
    )
    energy = lambda r: model.compute_energy(params, r, mask0, species0)
    hessian = jax.jit(jax.jacfwd(jax.grad(energy)))
    lam_floor = args.snr_target * np.sqrt(2.0) * args.label_se / (2.0 * max(args.fixed_eps))
    out = _append_fixed_modes(out, states, anchor_slots, lambda r: np.asarray(hessian(jnp.asarray(r, jnp.float32))), args.fixed_eps, lam_floor, args.n_modes)

    valid = np.asarray([_geometry_ok(R, args.max_bond, args.min_pair) for R in out["R"]])
    rejected = int((~valid).sum())
    out = {key: np.asarray(value)[valid] for key, value in out.items()}
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.outdir / "control_states.npz", **out)
    manifest = {
        "states_input": str(args.states), "model_config": str(args.model_config),
        "model_params": str(args.model_params), "reference": str(args.reference),
        "n_anchors": len(anchor_slots), "n_states": len(out["R"]),
        "n_geometry_rejected": rejected, "anchor_slots": anchor_slots,
        "path_fractions": args.path_fractions, "fixed_eps_A": args.fixed_eps,
        "lambda_floor": lam_floor, "constant_cost_claim": False,
        "note": "fixed low/negative modes are bounded-step controls, not equal-cost probes",
    }
    write_json(args.outdir / "control_manifest.json", manifest)
    (args.outdir / "control_plan.md").write_text(
        "# Non-production v3/v4 control pilot\n\n"
        "This state file is a physics control arm. `radius_path` rows test inner/outer "
        "linearity and endpoint-cost asymmetry. `fixed_low_or_negative_mode` rows use "
        "explicit fixed Cartesian steps and must not be interpreted as constant-cost v4 rows.\n"
    )
    print(f"wrote {len(out['R']):,} control states from {len(anchor_slots):,} anchors")
    print(f"geometry rejected: {rejected:,}; lambda floor: {lam_floor:.2f}")


if __name__ == "__main__":
    main()
