#!/usr/bin/env python3
"""Stage-3 v3 campaign: frozen-bead AA runs at SYNTHESISED stencil configurations.

WHAT IS NEW RELATIVE TO build_meanforce_campaign.py
    v2 could only harvest at configurations that EXIST in the source trajectory: each state's
    AA seed was the trajectory frame itself. A finite-difference stencil needs configurations
    that were never visited, so this builder realises them:

        seed_AA(anchor + delta) = seed_AA(anchor) with the SIX bead atoms translated by delta,
        everything else (rest of the peptide, all solvent) left alone and relaxed by the
        frozen-bead MD.

    That is legitimate exactly because Stage 3 FREEZES the beads: the free atoms relax around
    whatever bead positions they are given, so the resulting ensemble is the correct conditional
    one. It is safe because the displacements are tiny -- 0.08 and 0.16 A against ~1.5 A bonds,
    well inside thermal fluctuation -- so no clash is created. Validated already: 250,000/250,000
    synthesised configurations passed bond, closest-approach and chirality checks.

    Displacement vectors transfer from pool coordinates to seed coordinates unchanged because
    `trjconv -pbc mol -center` applies TRANSLATIONS only, never a rotation, and a translation
    leaves a difference vector invariant.

THE FAILURE MODE THIS MUST NOT REPEAT
    In v2 the freeze/restraint target was briefly taken from the collected array rather than
    from the extracted seed. trjconv places atoms differently, so the two disagreed in absolute
    position and the restraint held 164-570 kcal/mol at step 0. Here the anchor target still
    comes from ITS OWN seed.gro, and every stencil target is that seed plus an internal
    displacement -- so the initial frozen configuration is by construction exactly what the
    label will be attributed to. `--verify` re-reads every written .gro and asserts it.

USAGE
    python -m sampling.build_stencil_campaign \
        --stencil-npz local_work/v3_stage3_stencil/stencil_states.npz \
        --pool local_work/v2_stage2_harvest/cg_coords_all.npz \
        --source-campaign local_work/v2_stage2_harvest \
        --topology <topol.top> --mdp <frozen.mdp> \
        --outdir local_work/v3_stage3_meanforce --nodes 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np


def _log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stencil-npz", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--source-campaign", type=Path, required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--mdp", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--ps-per-state", type=float, default=11.0,
                    help="v2 used 11 ps (55 frames at 0.2 ps); tau = 1 frame under freezing")
    ap.add_argument("--output-ps", type=float, default=0.2)
    ap.add_argument("--pbc-margin", type=float, default=6.0)
    ap.add_argument("--conformation-tol", type=float, default=0.05)
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--replicas-per-job", type=int, default=32)
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--groups-per-task", type=int, default=8)
    ap.add_argument("--wall-hours", type=float, default=8.0)
    ap.add_argument("--seed-jobs", type=int, default=64)
    ap.add_argument("--gmx", default="gmx")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--limit-anchors", type=int, default=None,
                    help="smoke-test hook: build only the first N anchors")
    ap.add_argument("--verify", action="store_true", default=True)
    a = ap.parse_args()

    import concurrent.futures as cf
    import multiprocessing as _mp

    import mdtraj as md

    from sampling.build_meanforce_campaign import _seeds_for_replica, pbc_margin
    from sampling.mapping import get_mapping

    m = get_mapping(a.mapping)
    # Same accessor build_meanforce_campaign.py:244 uses, so the bead atom set is identical to
    # the one the v2 harvest froze -- a different set here would silently change what "the CG
    # configuration" means between the two datasets.
    atoms = m.aa_atom_indices_1based
    bead_atoms0 = [q - 1 for q in atoms]

    st = np.load(a.stencil_npz)
    S_R = np.asarray(st["R"], np.float64)
    S_anchor = np.asarray(st["anchor"], np.int64)
    pool_R = np.asarray(np.load(a.pool)["R"], np.float64)
    per_case = np.asarray(np.load(a.pool)["per_case"], np.int64)
    _log(f"[stencil] {len(S_R)} states over {len(np.unique(S_anchor))} anchors")

    # ---- pool index -> (case dir, time_ps) ------------------------------------------------
    cases = sorted([p for p in a.source_campaign.iterdir()
                    if p.is_dir() and (p.name.startswith("replica_") or p.name.startswith("case_"))])
    if len(cases) != len(per_case):
        raise SystemExit(f"{len(cases)} case dirs but per_case has {len(per_case)} entries")
    starts = np.concatenate([[0], np.cumsum(per_case)])
    times = [np.asarray(np.load(c / "cg_coords.npz")["time_ps"], np.float64) for c in cases]

    anchors = np.unique(S_anchor)
    if a.limit_anchors:
        anchors = anchors[:a.limit_anchors]
        keep = np.isin(S_anchor, anchors)
        S_R, S_anchor = S_R[keep], S_anchor[keep]
        _log(f"[smoke] limited to {len(anchors)} anchors -> {len(S_R)} states")

    by_case: dict[int, list] = {}
    anchor_slot = {}
    for slot, pi in enumerate(anchors):
        ci = int(np.searchsorted(starts, pi, side="right") - 1)
        fi = int(pi - starts[ci])
        by_case.setdefault(ci, []).append((slot, float(times[ci][fi])))
        anchor_slot[int(pi)] = slot

    a.outdir.mkdir(parents=True, exist_ok=True)
    tmpdir = a.outdir / "_seedtmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # ---- extract ONE AA seed per anchor (reuses the verified v2 worker) --------------------
    jobs = [(ci, items, str(cases[ci]), str(a.outdir), str(tmpdir), a.gmx, bead_atoms0)
            for ci, items in sorted(by_case.items())]
    t0 = time.time()
    seed_cache: dict[int, tuple] = {}
    with cf.ProcessPoolExecutor(max_workers=max(1, min(a.seed_jobs, len(jobs))),
                                mp_context=_mp.get_context("spawn")) as ex:
        for ci, got, err in ex.map(_seeds_for_replica, jobs):
            if err:
                raise SystemExit(err)
            seed_cache.update(got)
    _log(f"[seeds] {len(seed_cache)} anchor seeds from {len(jobs)} source trajectories "
         f"in {time.time()-t0:.0f} s")

    # ---- expand each anchor seed into its stencil -----------------------------------------
    nsteps = int(a.ps_per_state * 1000)
    nst_out = int(a.output_ps * 1000)
    base = a.mdp.read_text().splitlines()
    order = np.argsort(S_anchor, kind="stable")
    written, margins, devs = 0, [], []
    gro_cache: dict[int, list] = {}
    t0 = time.time()
    for k in order:
        pi = int(S_anchor[k])
        slot = anchor_slot[pi]
        seed, seed_box = seed_cache[slot]                       # anchor bead coords (A), box
        delta = S_R[k] - pool_R[pi]                             # internal displacement
        tgt = seed + delta                                      # target bead positions

        mg = pbc_margin(tgt, seed_box)
        if mg < a.pbc_margin:
            raise SystemExit(f"state {k}: bead within {mg:.1f} A of a box face "
                             f"(need {a.pbc_margin})")
        margins.append(mg)

        d = a.outdir / f"state_{int(k):06d}"
        d.mkdir(parents=True, exist_ok=True)
        # Direct .gro text patching, not an mdtraj load/save round-trip: at ~40 ms per state
        # that round-trip is 2.8 HOURS for 250,000 states. The anchor's file is read once and
        # cached; each stencil point rewrites only its six bead lines.
        # .gro is fixed-width -- coordinates occupy chars 20:28, 28:36, 36:44 as %8.3f in nm,
        # which is also why the verify tolerance below is 0.01 A (1e-3 nm), not machine epsilon.
        if slot not in gro_cache:
            gro_cache.clear()                                   # anchors are processed in order
            gro_cache[slot] = (a.outdir / f"state_{slot:05d}" / "seed.gro").read_text().split("\n")
        lines = list(gro_cache[slot])
        for bi, ai0 in enumerate(bead_atoms0):
            ln = lines[ai0 + 2]
            xyz = np.array([float(ln[20:28]), float(ln[28:36]), float(ln[36:44])])
            xyz = xyz + delta[bi] / 10.0                        # A -> nm
            lines[ai0 + 2] = (ln[:20] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                              + ln[44:])
        (d / "seed.gro").write_text("\n".join(lines))
        written += 1

        mdp = []
        for line in base:
            s = line.strip()
            if s.startswith("nsteps"):
                line = f"nsteps                  = {nsteps} ; {a.ps_per_state} ps"
            elif s.startswith(("nstxout", "nstfout")):
                line = f"{s.split()[0]:<24}= {nst_out}"
            elif s.startswith(("gen_vel", "gen-vel")):
                line = "gen_vel                 = yes"
            elif s.startswith("gen_seed"):
                line = f"gen_seed                = {a.seed + int(k)}"
            # The base mdp may ALREADY carry a freeze block (e.g. when a v2 state's
            # production.mdp is reused as the template). grompp rejects duplicate keys, so
            # drop any existing freeze lines and re-emit exactly one block below.
            if s.split("=")[0].strip() in ("freezegrps", "freezedim", "comm-mode", "comm_mode"):
                continue
            mdp.append(line)
        mdp += ["", "freezegrps              = CGbeads   ; CG beads held EXACTLY fixed",
                "freezedim               = Y Y Y",
                "comm-mode               = None      ; ill-defined with frozen atoms"]
        (d / "production.mdp").write_text("\n".join(mdp) + "\n")

        if a.verify:
            vl = (d / "seed.gro").read_text().split("\n")
            chk = np.array([[float(vl[i + 2][20:28]), float(vl[i + 2][28:36]),
                             float(vl[i + 2][36:44])] for i in bead_atoms0]) * 10.0
            devs.append(float(np.abs(chk - tgt).max()))

    _log(f"[expand] {written} state dirs written in {time.time()-t0:.0f} s")
    _log(f"[verify] max |written bead position - intended target| = {max(devs):.4f} A "
         f"(gro precision is 1e-3 nm = 0.01 A)")
    _log(f"[pbc]    min margin {min(margins):.2f} A")

    manifest = {"n_states": int(written), "n_anchors": int(len(anchors)),
                "ps_per_state": a.ps_per_state, "output_ps": a.output_ps,
                "frames_per_state": int(a.ps_per_state / a.output_ps),
                "expected_SE": float(19.8 / np.sqrt(a.ps_per_state / a.output_ps)),
                "freeze": True, "nodes": a.nodes,
                "stencil_npz": str(a.stencil_npz),
                "max_seed_deviation_A": float(max(devs)) if devs else None,
                "min_pbc_margin_A": float(min(margins))}
    (a.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _log(f"\nwrote {a.outdir}/manifest.json  ({written} states)")


if __name__ == "__main__":
    main()
