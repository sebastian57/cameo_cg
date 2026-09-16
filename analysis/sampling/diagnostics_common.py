"""Shared configuration and diagnostics for the MD-less analysis suite.

All scientific artifacts are supplied by the caller. This module intentionally has no
machine-specific reference, bias, flow, or project paths.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.common.paths import repo_root, resolve_input


@dataclass(frozen=True)
class DiagnosticConfig:
    reference: Path
    bias_npz: Path
    flow_dir: Path
    flow_primary: str = "small_seed0"
    mapping_name: str = "ala2_backbone_cb_6"
    temperature_K: float = 298.0

    @property
    def kT(self) -> float:
        return 0.0019872042586 * self.temperature_K

    def flow_path(self) -> Path:
        return self.flow_dir / f"flow_{self.flow_primary}.npz"


PROJECT_ROOT = str(repo_root())
KT = 0.0019872042586 * 298.0


def add_latent_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the required shared reference/TICA/flow inputs to an analysis CLI."""
    parser.add_argument("--frames", type=Path, required=False,
                        help="reference NPZ containing coordinates (and, where needed, forces)")
    parser.add_argument("--bias-npz", type=Path, required=True,
                        help="frozen TICA bias/projection artifact")
    parser.add_argument("--flow-dir", type=Path, required=True,
                        help="directory containing flow_<flow-primary>.npz")
    parser.add_argument("--flow-primary", default="small_seed0",
                        help="flow artifact stem suffix")
    parser.add_argument("--mapping", default="ala2_backbone_cb_6",
                        help="registered mapping name")
    parser.add_argument("--temperature-K", type=float, default=298.0,
                        help="temperature used for kT-scaled diagnostics")


def config_from_args(args: argparse.Namespace) -> DiagnosticConfig:
    if args.frames is None or args.bias_npz is None or args.flow_dir is None:
        raise SystemExit("--frames, --bias-npz, and --flow-dir are required for this analysis")
    return DiagnosticConfig(
        reference=resolve_input(args.frames, label="reference frames"),
        bias_npz=resolve_input(args.bias_npz, label="TICA bias artifact"),
        flow_dir=resolve_input(args.flow_dir, label="flow directory"),
        flow_primary=str(args.flow_primary),
        mapping_name=str(args.mapping),
        temperature_K=float(args.temperature_K),
    )


def parse_spec(spec):
    """'config.yaml:params.pkl' -> (config, params)."""
    cfg, _, par = str(spec).partition(":")
    if not par:
        raise SystemExit(f"expected config:params, got {spec!r}")
    return cfg, par


def pair_distance_features(R):
    """Standardized unique bead-pair distances, (n, beads*(beads-1)/2)."""
    n_beads = R.shape[1]
    iu, ju = np.triu_indices(n_beads, k=1)
    D = np.linalg.norm(R[:, iu, :] - R[:, ju, :], axis=-1)
    return (D - D.mean(0)) / D.std(0)


def assign_regions(R, mapping):
    """Assign the documented beta/alphaR/alphaL basin labels."""
    from sampling.mapping import dihedral_deg, wrap_deg
    cv = lambda n: wrap_deg(
        dihedral_deg(np.asarray(R, np.float64), mapping.cvs[n].bead_indices)
        + mapping.cvs[n].shift_deg
    )
    phi, psi = cv("phi"), cv("psi")
    reg = np.full(len(R), "other", dtype=object)
    reg[(phi > -180) & (phi < -20) & ((psi > 90) | (psi < -150))] = "beta"
    reg[(phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)] = "alphaR"
    reg[(phi > 20) & (phi < 100) & (psi > -20) & (psi < 100)] = "alphaL"
    return reg, phi, psi


def load_latent(
    frames_path,
    n_frames,
    grid_lim=4.5,
    bins=80,
    *,
    config: DiagnosticConfig,
):
    """Project reference frames through the explicitly supplied bias and flow."""
    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import load_flow, to_latent
    from sampling.mapping import get_mapping

    mapping = get_mapping(config.mapping_name)
    bias = SmoothTICABias.load(config.bias_npz)
    params_f, cfg_f = load_flow(config.flow_path())

    R_all = np.asarray(np.load(frames_path, allow_pickle=False)["R"], np.float64)
    idx = np.linspace(0, len(R_all) - 1, min(n_frames, len(R_all))).astype(int)
    R = R_all[idx]
    z = np.asarray(bias.projection.transform(R), np.float64)[:, :cfg_f.n_dims]
    u = np.asarray(to_latent(params_f, cfg_f, z), np.float64)

    reg, phi, psi = assign_regions(R, mapping)
    edges = np.linspace(-grid_lim, grid_lim, bins + 1)
    inside = (np.abs(u) < grid_lim).all(axis=1)
    ix = np.clip(np.searchsorted(edges, u[:, 0], "right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(edges, u[:, 1], "right") - 1, 0, bins - 1)
    flat = np.where(inside, ix * bins + iy, -1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return dict(
        R=R, u=u, region=reg, phi=phi, psi=psi, flat=flat, inside=inside,
        edges=edges, centers=centers, bins=bins, mapping=mapping,
    )


def cell_stats(dU, flat, n_cells, min_count, *, kT=KT):
    """Per-cell mean and exact free-energy shift."""
    beta = 1.0 / kT
    n_c = np.bincount(flat, minlength=n_cells).astype(np.float64)
    m_c = np.full(n_cells, -np.inf)
    np.maximum.at(m_c, flat, -beta * dU)
    s_c = np.bincount(flat, weights=np.exp(-beta * dU - m_c[flat]), minlength=n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnmean = m_c + np.log(s_c / n_c)
        dU_mean = np.bincount(flat, weights=dU, minlength=n_cells) / n_c
    out = -kT * lnmean
    bad = n_c < min_count
    out[bad], dU_mean[bad] = np.nan, np.nan
    return dU_mean, out, n_c


def selfcheck_cell_stats():
    rng = np.random.default_rng(0)
    dU = rng.normal(0, 2, 5000)
    flat = rng.integers(0, 12, 5000)
    dU_mean, dF, _ = cell_stats(dU, flat, 12, min_count=5)
    for c in range(12):
        m = flat == c
        if m.sum() < 5:
            assert np.isnan(dF[c]) and np.isnan(dU_mean[c])
            continue
        assert abs(dU_mean[c] - dU[m].mean()) < 1e-12
        assert abs(dF[c] + KT * np.log(np.exp(-dU[m] / KT).mean())) < 1e-9
    _, dF0, _ = cell_stats(np.zeros(50), np.zeros(50, int), 1, 5)
    assert abs(dF0[0]) < 1e-12
    print("[selfcheck] cell_stats matches brute force; zero-shift -> dF=0. OK")
