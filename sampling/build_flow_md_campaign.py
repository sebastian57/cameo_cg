#!/usr/bin/env python3
"""Phase 3: matched MD comparing control / KDE bias / flow bias.

    python -m sampling.build_flow_md_campaign --bias-npz <b.npz> --reference <ref.npz> \
        --flow local_work/flow_sweep_final/flow_small_seed0.npz \
        --mdp <production.mdp> --topology <topol.top> --structures <dir> \
        --outdir local_work/flow_phase3 --n-replicas 8

WHAT PHASE 3 IS FOR
    NOT "is the flow bias better at sampling" -- one short campaign cannot answer that. It is
    to establish that swapping the reference density does not BREAK physical acquisition
    behaviour: same molecules, same stability, same geometric validity, comparable coverage.

    Three arms, identical in every respect except the bias:
      control  no bias at all -- the baseline the other two are enrichments over
      kde      the incumbent `V = -kT log(pi/p_KDE)`
      flow     the same form with `p_theta + p_c` (floor B, KDE-free)

    Same start structures, same mdp, same seeds, same walls, same TICA CVs. Every arm also
    PRINTs its bias value, so what the trajectory actually experienced is recorded rather
    than inferred from the tabulated field.

WHY THE BIAS IS TABULATED
    At d=2 the flow never runs during MD: it is evaluated once here onto an EXTERNAL grid, so
    the MD loop is pure GROMACS+PLUMED. (At d>=3 tabulation dies combinatorially and the flow
    would have to go through `CG_BIAS MODEL=` instead.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PLUMED_HEAD = """\
UNITS LENGTH=A ENERGY=kcal/mol
WHOLEMOLECULES ENTITY0={atoms}
{cvs}"""

PLUMED_TAIL = """\
{cv_lines}
PRINT ARG={printed} FILE=colvar.dat STRIDE={stride}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bias-npz", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--flow", type=Path, required=True)
    ap.add_argument("--mdp", type=Path, required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--structures", type=Path, required=True,
                    help="dir of start_*.gro; cycled if fewer than --n-replicas")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-replicas", type=int, default=8)
    ap.add_argument("--grid-points", type=int, default=401)
    ap.add_argument("--print-stride", type=int, default=100)
    ap.add_argument("--floor", choices=("constant", "kde"), default="constant")
    ap.add_argument("--target-depth", type=float, default=1.77)
    ap.add_argument("--wall-hours", type=float, default=1.0)
    ap.add_argument("--gen-seed-base", type=int, default=20260813)
    ap.add_argument("--arms", nargs="+", default=["control", "kde", "flow"],
                    choices=["control", "kde", "flow"],
                    help="a DISCOVER run wants just `flow`; the three-arm default is the "
                         "Phase-3 substitution test")
    ap.add_argument("--metad", action="store_true",
                    help="add well-tempered MetaD on the TICA CVs to every BIASED arm. The "
                         "flow bias is a static prior; MetaD is the adaptive term that fills "
                         "what the prior points at. They compose because PLUMED sums bias "
                         "forces.")
    ap.add_argument("--metad-walkers", action="store_true",
                    help="MULTIPLE WALKERS (WALKERS_MPI): all replicas share ONE growing bias. "
                         "Without it N replicas each fill from zero -- Nx redundant work and "
                         "no replica gets far. Required for wide-and-short discovery.")
    ap.add_argument("--metad-height", type=float, default=0.15)
    ap.add_argument("--metad-sigma", type=float, nargs=2, default=[0.15, 0.09])
    ap.add_argument("--metad-pace", type=int, default=500)
    ap.add_argument("--metad-biasfactor", type=float, default=8.0)
    ap.add_argument("--metad-equilibrate-steps", type=int, default=10000)
    a = ap.parse_args()

    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_bias import (AcquisitionBias, flow_density_fn, kde_density_fn,
                                    transition_weights)
    from sampling.flow_density import load_flow
    from sampling.launch import cpus_per_rank, group_ranges, multidir_group_script, submit_script
    from sampling.mapping import get_mapping
    from sampling.plumed_native import (external_block, metad_block, padded_bounds,
                                        tica_cv_block, walls_block, write_grid_from_fn)

    bias = SmoothTICABias.load(a.bias_npz)
    mapping = get_mapping("ala2_backbone_cb_6")
    kbt, lam = float(bias.kbt_kcal_mol), float(np.load(a.bias_npz)["lambda_value"])
    lo, hi = padded_bounds(bias)
    C = np.asarray(bias.centers); band = np.asarray(bias.bandwidth)
    r_w = transition_weights(a.bias_npz)
    gx = np.linspace(lo[0], hi[0], a.grid_points)
    gy = np.linspace(lo[1], hi[1], a.grid_points)
    cal = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)

    a.outdir.mkdir(parents=True, exist_ok=True)
    # RESOLVED: the group script runs `cd replica_NN` before grompp, so a relative path here
    # fails with "File ... does not exist" (measured, jobs 1350203-1350205). Same reason
    # launch.py resolves the campaign dir.
    starts = sorted(q.resolve() for q in a.structures.glob("start_*.gro"))
    if not starts:
        raise SystemExit(f"no start_*.gro under {a.structures.resolve()}")
    a.topology = str(Path(a.topology).resolve())

    def make_bias(density, mode, floor_density=None):
        b = AcquisitionBias(density=density, centers=C, r_weights=r_w, bandwidth=band,
                            lam=lam, kbt=kbt, floor_mode=mode, floor_density=floor_density,
                            target_depth_kcal=a.target_depth)
        b.calibrate(cal)
        return b

    dens_kde = kde_density_fn(bias)
    flow_params, flow_cfg = load_flow(a.flow)
    all_arms = {
        "control": None,
        "kde": make_bias(dens_kde, "none"),
        "flow": make_bias(flow_density_fn(flow_params, flow_cfg), a.floor,
                          floor_density=dens_kde if a.floor == "kde" else None),
    }
    arms = {k: all_arms[k] for k in a.arms}

    # MetaD grid comes from the SAME padded bounds as the EXTERNAL grid, so the two gridded
    # actions cannot disagree about where the sampled region ends. Gridless METAD costs
    # O(N_hills) per step and grows without bound -- invisible in a 20 ps test, serious in a
    # discover run, which is exactly this campaign.
    metad_txt = ""
    if a.metad:
        metad_txt = metad_block(
            height=a.metad_height, sigma=tuple(a.metad_sigma), pace=a.metad_pace,
            bias_factor=a.metad_biasfactor, temperature=298.0,
            equilibrate_steps=a.metad_equilibrate_steps, dt_ps=_mdp_dt_ps(a.mdp),
            grid_min=tuple(lo), grid_max=tuple(hi), walkers_mpi=a.metad_walkers)

    cv_lines, printed = [], ["tic1", "tic2"]
    for name in ("phi", "psi", "chirality"):
        idx = ",".join(str(i) for i in mapping.cvs[name].atom_indices_1based(mapping))
        cv_lines.append(f"{name}: TORSION ATOMS={idx}")
        printed.append(name)

    manifest = {"lambda": lam, "kbt": kbt, "floor": a.floor,
                "metad": dict(enabled=bool(a.metad), walkers_mpi=bool(a.metad_walkers),
                              height=a.metad_height, sigma=list(a.metad_sigma),
                              pace=a.metad_pace, bias_factor=a.metad_biasfactor,
                              equilibrate_steps=a.metad_equilibrate_steps) if a.metad else None,
                "target_depth": a.target_depth, "grid_points": a.grid_points,
                "bounds": bias.bounds.tolist(), "padded": [lo.tolist(), hi.tolist()],
                "n_replicas": a.n_replicas, "arms": {}}

    for arm, ab in arms.items():
        root = a.outdir / arm
        root.mkdir(parents=True, exist_ok=True)
        head = PLUMED_HEAD.format(atoms=mapping.plumed_atom_selection(),
                                  cvs=tica_cv_block(bias, mapping))
        body, printed_arm = "", list(printed)
        if ab is not None:
            grid = root / "acq_grid.dat"
            write_grid_from_fn(lambda Z: ab.energy_gradient(Z), grid, lo, hi,
                               n_points=(a.grid_points, a.grid_points), label="acq")
            body = external_block(Path("../acq_grid.dat"), label="acq") + walls_block(bias)
            printed_arm += ["acq.bias"]
            if metad_txt:
                body += metad_txt
                printed_arm += ["metad.bias"]
            manifest["arms"][arm] = dict(depth_bound=ab.depth_bound(cal),
                                         floor_constant=ab.floor_constant,
                                         grid=str(grid), grid_mb=round(grid.stat().st_size / 1e6, 1))
        else:
            # walls still applied, so the control is the SAME confined system minus the bias
            body = walls_block(bias)
            manifest["arms"][arm] = dict(depth_bound=0.0)

        text = head + body + PLUMED_TAIL.format(cv_lines="\n".join(cv_lines),
                                                printed=",".join(printed_arm),
                                                stride=a.print_stride)
        for k in range(a.n_replicas):
            d = root / f"replica_{k:02d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "plumed.dat").write_text(text)
            _write_mdp(a.mdp, d / "production.mdp", a.gen_seed_base + k)

        ntomp = cpus_per_rank(a.n_replicas)
        groups = group_ranges(a.n_replicas, a.n_replicas)
        (root / "run_group_000.sh").write_text(multidir_group_script(
            case_dirs=[f"replica_{k:02d}" for k in range(a.n_replicas)],
            structure_for=[str(starts[k % len(starts)]) for k in range(a.n_replicas)],
            topology=a.topology, ntomp=ntomp, n_gpus=4, use_server=False, mps=True))
        (root / "run_group_000.sh").chmod(0o755)
        (root / "submit.slurm").write_text(submit_script(
            campaign_dir=root, groups=groups, job_name=f"p3_{arm}", hours=a.wall_hours))
        print(f"{arm:8s}: {a.n_replicas} replicas"
              + (f", grid {manifest['arms'][arm].get('grid_mb')} MB, "
                 f"depth bound {manifest['arms'][arm]['depth_bound']:.2f} kcal/mol"
                 if ab is not None else ", no bias (walls only)"))

    (a.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {a.outdir}/manifest.json")
    for arm in arms:
        print(f"  sbatch {a.outdir/arm}/submit.slurm")


def _mdp_dt_ps(mdp: Path) -> float:
    for line in Path(mdp).read_text().splitlines():
        if "=" in line and line.split("=")[0].strip().lower() == "dt":
            return float(line.split("=")[1].split(";")[0].strip())
    raise ValueError(f"{mdp}: no dt found")


def _write_mdp(src: Path, dst: Path, seed: int) -> None:
    """Per-replica velocity seed. Without it every replica is the SAME trajectory."""
    over = {"gen_vel": "yes", "gen_temp": "298", "gen_seed": str(seed), "continuation": "no"}
    seen, out = set(), []
    for line in Path(src).read_text().splitlines():
        k = line.split("=")[0].strip().replace("-", "_").lower() if "=" in line else ""
        if k in over:
            out.append(f"{k:<24}= {over[k]}   ; per-replica"); seen.add(k)
        else:
            out.append(line)
    out += [""] + [f"{k:<24}= {v}   ; per-replica" for k, v in over.items() if k not in seen]
    dst.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
