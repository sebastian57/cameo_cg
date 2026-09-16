"""Build a frozen-bead campaign from v4 stencil states whose anchors are REFERENCE frames.

v3's `build_stencil_campaign.py` sources AA seeds from a Stage-2 harvest campaign: each anchor
is a pool frame, and the seed is extracted from that case's trajectory with gmx. v4 anchors are
frames of the AA reference trajectory instead, so the seed is simply that frame -- no harvest
campaign is involved.

Deliberately a SEPARATE file rather than a branch inside the v3 builder: that code is validated
and was broken once already by an in-place refactor (BUGS/2026-08-18). The state-expansion,
mdp-writing and cleanup logic below is a faithful copy of it; only seed acquisition differs.

PBC: raw GROMACS frames store molecules broken across the periodic boundary (a bead-bead
distance of 31.8 A in a 33.2 A box at frame 1000). `mdtraj.image_molecules` restores them,
verified against the CG reference npz to 2.6e-6 A.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/e/project1/cameo/schmidt36/cameo_cg")


def _log(msg: str) -> None:
    print(msg, flush=True)


def format_seed_gro(template_lines, xyz_nm, box_nm) -> list[str]:
    """Replace coordinates in a reference GRO while preserving atom-name columns.

    ``mdtraj.save_gro`` canonicalises names such as ``HH31`` to ``H1`` and ``OW`` to
    ``O``.  The topology used by the reference trajectory retains the original names,
    so write the coordinates into the reference GRO records instead.  Velocities are
    intentionally dropped because the campaign MDP generates fresh velocities.
    """
    lines = list(template_lines)
    xyz = np.asarray(xyz_nm, np.float64)
    box_array = np.asarray(box_nm, np.float64)
    if box_array.shape == (3, 3):
        a, b, c = box_array
        box = np.asarray([a[0], b[1], c[2], a[1], a[2], b[0],
                          b[2], c[0], c[1]], np.float64)
    else:
        box = box_array.reshape(-1)
    if len(lines) < 3 or int(lines[1].strip()) != len(xyz):
        raise ValueError("GRO template atom count does not match coordinates")
    for i, (x, y, z) in enumerate(xyz):
        lines[i + 2] = lines[i + 2][:20] + f"{x:8.3f}{y:8.3f}{z:8.3f}"
    if len(box) not in (3, 9):
        raise ValueError("GRO box must contain 3 or 9 values")
    lines[len(xyz) + 2] = " ".join(f"{value:10.5f}" for value in box)
    return lines


def _mdp_key(line: str) -> str:
    if "=" not in line:
        return ""
    return line.split("=", 1)[0].strip().lower().replace("_", "-")


def rewrite_production_mdp(base_lines, *, ps_per_state: float, output_ps: float,
                           gen_seed: int) -> list[str]:
    """Adapt the reference MDP for one independent frozen-state production run.

    GROMACS ``dt`` is in ps.  Deriving step counts from it preserves the reference
    trajectory's timestep instead of silently assuming 1 fs.  A seed.gro has no
    velocities or checkpoint, so each state starts an independent NPT trajectory.
    """
    dt_ps = None
    for line in base_lines:
        if _mdp_key(line) == "dt":
            dt_ps = float(line.split("=", 1)[1].split(";", 1)[0].strip())
            break
    if dt_ps is None or dt_ps <= 0:
        raise ValueError("reference MDP must contain a positive dt")
    nsteps = int(round(ps_per_state / dt_ps))
    nst_out = int(round(output_ps / dt_ps))
    if nsteps < 1 or nst_out < 1:
        raise ValueError("production and output intervals must contain at least one step")
    if not np.isclose(nsteps * dt_ps, ps_per_state, rtol=0, atol=1e-9):
        raise ValueError(f"ps_per_state={ps_per_state} is not divisible by dt={dt_ps}")
    if not np.isclose(nst_out * dt_ps, output_ps, rtol=0, atol=1e-9):
        raise ValueError(f"output_ps={output_ps} is not divisible by dt={dt_ps}")

    replacements = {
        "nsteps": f"nsteps                  = {nsteps} ; {ps_per_state:g} ps",
        "nstxout": f"nstxout                 = {nst_out}",
        "nstvout": f"nstvout                 = {nst_out}",
        "nstfout": f"nstfout                 = {nst_out}",
        "continuation": "continuation            = no",
        "gen-vel": "gen_vel                 = yes",
        "gen-temp": "gen_temp                = 298",
        "gen-seed": f"gen_seed                = {int(gen_seed)}",
    }
    removed = {"freezegrps", "freezedim", "comm-mode"}
    seen = set()
    out = []
    for line in base_lines:
        key = _mdp_key(line)
        if key in removed:
            continue
        if key in replacements:
            out.append(replacements[key])
            seen.add(key)
        else:
            out.append(line)
    for key in ("continuation", "gen-vel", "gen-temp", "gen-seed"):
        if key not in seen:
            out.append(replacements[key])
    out += ["", "freezegrps              = CGbeads   ; CG beads held EXACTLY fixed",
            "freezedim               = Y Y Y",
            "comm-mode               = None      ; ill-defined with frozen atoms"]
    return out


def pool_case_path(root: Path, template: str, index: int) -> Path:
    """Resolve a pool case directory while retaining the legacy default template."""
    if index < 0:
        raise ValueError("pool case index must be non-negative")
    try:
        name = template.format(index=index, case=index)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError("pool case template must format {index} or {case}") from exc
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("pool case template must resolve below the campaign directory")
    return Path(root) / path


def pool_source_frame(local_frame: int, offset: int) -> int:
    """Translate a post-discard pool frame to its original trajectory frame."""
    if local_frame < 0 or offset < 0:
        raise ValueError("pool frame and offset must be non-negative")
    return int(local_frame + offset)


def state_manifest_entry(state, anchor, direction, target, anchor_source=None,
                         anchor_index=None, multiplier=None, kind=None):
    """Return the collector-compatible record for one realized frozen state."""
    entry = {
        "state": int(state),
        "anchor": int(anchor),
        "direction": int(direction),
        "target": np.asarray(target, np.float64).round(3).tolist(),
    }
    for key, value in (("anchor_source", anchor_source), ("anchor_index", anchor_index),
                       ("multiplier", multiplier), ("kind", kind)):
        if value is not None:
            entry[key] = int(value) if key in ("anchor_source", "anchor_index", "kind") else float(value)
    return entry


def pool_case_frame_groups(pool_slots, anchor_index, pool_starts):
    """Map global pool slots to case-local frame numbers in first-seen case order."""
    groups = {}
    for slot in sorted(pool_slots, key=lambda s: int(anchor_index[s])):
        global_frame = int(anchor_index[slot])
        case = int(np.searchsorted(pool_starts, global_frame, side="right") - 1)
        local_frame = global_frame - int(pool_starts[case])
        groups.setdefault(case, []).append((local_frame, int(slot)))
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stencil-npz", type=Path, required=True)
    ap.add_argument("--aa-traj", required=True, help="solvated AA trajectory (frame i <-> CG row i)")
    ap.add_argument("--aa-top", required=True, help="topology for the solvated system (.gro)")
    ap.add_argument("--mdp", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--ps-per-state", type=float, default=3.8)
    ap.add_argument("--output-ps", type=float, default=0.032)
    ap.add_argument("--discard-ps", type=float, default=1.0)
    ap.add_argument("--pbc-margin", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--limit-anchors", type=int, default=None, help="pilot mode")
    ap.add_argument("--pool-npz", default=None,
                    help="cg_coords_all.npz that SRC_POOL anchor_index values index into")
    ap.add_argument("--pool-source-campaign", default=None,
                    help="campaign dir with pool-case-template/biased.{trr,xtc} supplying seeds")
    ap.add_argument("--pool-case-template", default="case_{index:03d}",
                    help="pool case directory template (default: case_{index:03d})")
    ap.add_argument("--pool-traj-name", default=None,
                    help="pool trajectory file to seek in (default: biased.trr, else biased.xtc). "
                         "Set explicitly when the pool frame indices refer to one of the two: a "
                         "campaign that writes trr and xtc at different strides has different "
                         "frame numbering in each.")
    ap.add_argument("--pool-frame-offset", type=int, default=0,
                    help="frames to add when pool indices refer to a discarded trajectory prefix")
    ap.add_argument("--verify", action="store_true", default=True)
    a = ap.parse_args()

    import mdtraj as md
    from sampling.build_meanforce_campaign import pbc_margin
    from sampling.mapping import get_mapping

    m = get_mapping(a.mapping)
    bead_atoms0 = [q - 1 for q in m.aa_atom_indices_1based]

    st = np.load(a.stencil_npz)
    S_R = np.asarray(st["R"], np.float64)
    S_anchor = np.asarray(st["anchor"], np.int64)
    S_dir = np.asarray(st["direction"], np.int8)
    S_mult = np.asarray(st["multiplier"], np.float64) if "multiplier" in st else None
    S_kind = np.asarray(st["kind"], np.int8) if "kind" in st else None
    anchor_index = np.asarray(st["anchor_index"], np.int64)   # -> AA/CG frame number
    _log(f"[stencil] {len(S_R)} states over {len(np.unique(S_anchor))} anchors")

    anchors = np.unique(S_anchor)
    if a.limit_anchors:
        anchors = anchors[:a.limit_anchors]
        keep = np.isin(S_anchor, anchors)
        S_R, S_anchor, S_dir = S_R[keep], S_anchor[keep], S_dir[keep]
        _log(f"[pilot] limited to {len(anchors)} anchors -> {len(S_R)} states")

    # anchor CG coordinates come from the state file itself (the direction == -1 rows)
    anchor_R = {}
    for k in np.flatnonzero(S_dir == -1):
        anchor_R[int(S_anchor[k])] = S_R[k]
    missing = [int(s) for s in anchors if int(s) not in anchor_R]
    if missing:
        raise SystemExit(f"{len(missing)} anchors have no direction==-1 row (first: {missing[:3]})")

    a.outdir.mkdir(parents=True, exist_ok=True)

    # ---- per-anchor source: 0 = reference traj, 1 = harvest pool -------------------------
    src_states = np.asarray(st["anchor_source"], np.int64) if "anchor_source" in st else None
    first_of_anchor = {}
    if src_states is not None:
        # NB: anchor_source/anchor_index are PER-ANCHOR arrays indexed by anchor ID
        # (build_stencil_states_v4 appends extras after the reference block), NOT
        # per-state arrays — index them with S_anchor[k], never k.
        for k in np.flatnonzero(S_dir == -1):
            first_of_anchor.setdefault(int(S_anchor[k]), int(src_states[int(S_anchor[k])]))
        n_pool_anchors = sum(1 for s in anchors if first_of_anchor.get(int(s)) == 1)
        _log(f"[seeds] {n_pool_anchors} pool-sourced anchors of {len(anchors)}")
        if n_pool_anchors and (a.pool_npz is None or a.pool_source_campaign is None):
            raise SystemExit(
                "stencil has pool anchors but --pool-npz/--pool-source-campaign missing"
            )
    pool_per_case = None
    if src_states is not None and n_pool_anchors:
        pool_per_case = np.asarray(np.load(a.pool_npz)["per_case"], np.int64)
        pool_starts = np.concatenate([[0], np.cumsum(pool_per_case)])

    # ---- one AA seed per anchor: reference trajectory or DHH case frames ---------------
    top = md.load(a.aa_top).topology
    gro_template = Path(a.aa_top).read_text().splitlines()
    if len(gro_template) < 3 or int(gro_template[1].strip()) != top.n_atoms:
        raise SystemExit("AA GRO template atom count does not match its MDTraj topology")
    _log(f"[seeds] topology {top.n_atoms} atoms, {top.n_bonds} bonds")
    seed_cache: dict[int, tuple] = {}
    devs_seed, offs, margins_seed = [], [], []
    t0 = time.time()

    def place_seed(xyz_nm, box_nm, slot):
        """Image, recentre, verify against the stencil anchor, write seed.gro."""
        t = md.Trajectory(np.asarray(xyz_nm), top)
        t.unitcell_vectors = np.asarray(box_nm)          # settable property, not a ctor arg
        t = t.image_molecules(inplace=False)
        xyz = np.asarray(t.xyz[0], np.float64) * 10.0            # nm -> A
        box = np.asarray(t.unitcell_vectors[0], np.float64) * 10.0

        # Reference frames are not centred: the molecule can sit against a box face, which
        # fails the >=6 A margin the displaced beads need. Shift the WHOLE system so the
        # bead centroid is at the box centre -- a rigid translation, so it changes no physics
        # and GROMACS is indifferent to absolute position.
        bd = xyz[bead_atoms0]
        shift = 0.5 * box.sum(0) - bd.mean(0)
        xyz = xyz + shift
        bd = xyz[bead_atoms0]

        A = anchor_R[slot]
        dev = float(np.abs((bd - bd.mean(0)) - (A - A.mean(0))).max())
        if dev > 0.02:
            raise SystemExit(f"anchor {slot}: seed bead GEOMETRY deviates "
                             f"{dev:.4f} A from the stencil anchor (centroid removed) -- "
                             f"frame indexing, PBC, or orientation is wrong")
        devs_seed.append(dev); offs.append(float(np.linalg.norm(shift)))
        margins_seed.append(pbc_margin(bd, box))

        d = a.outdir / f"state_{slot:05d}"
        d.mkdir(parents=True, exist_ok=True)
        seed_lines = format_seed_gro(gro_template, xyz / 10.0, box / 10.0)
        (d / "seed.gro").write_text("\n".join(seed_lines) + "\n")
        seed_cache[slot] = (bd, box)

    # Reference-trajectory anchors: open the trajectory ONCE and seek. md.load_frame reopens
    # the 19 GB file per call, which measured 2.4 s/anchor -- 10 hours for 15,000. Sorted
    # seeks on a single handle is ~30x faster.
    ref_slots = [int(s_) for s_ in anchors if first_of_anchor.get(int(s_), 0) == 0]
    order_anch = sorted((int(anchor_index[s_]), s_) for s_ in ref_slots)
    with md.open(a.aa_traj) as fh:
        for n, (fr, slot) in enumerate(order_anch):
            fh.seek(fr)
            fr_ = fh.read(1)          # TRR yields 5 fields, XTC 4; xyz is 0, box is 3
            xyz_nm, box_nm = fr_[0], fr_[3]
            place_seed(xyz_nm, box_nm, slot)
            if (n + 1) % 1000 == 0:
                _log(f"  ...{n+1}/{len(order_anch)} ref seeds ({time.time()-t0:.0f} s)")

    # Pool anchors: global pool index -> (case dir, local frame) via per_case boundaries;
    # AA seed comes from that case's own biased.trr/xtc at the matching output frame.
    pool_slots = [int(s_) for s_ in anchors if first_of_anchor.get(int(s_), 0) == 1]
    by_case = (pool_case_frame_groups(pool_slots, anchor_index, pool_starts)
               if pool_slots else {})
    for c, frames in sorted(by_case.items()):
        case_dir = pool_case_path(Path(a.pool_source_campaign), a.pool_case_template, c)
        names = ((a.pool_traj_name,) if a.pool_traj_name else ("biased.trr", "biased.xtc"))
        traj_path = next((case_dir / nm for nm in names if (case_dir / nm).exists()), None)
        if traj_path is None:
            raise SystemExit(f"pool case {c}: no biased.trr/biased.xtc in {case_dir}")
        with md.open(str(traj_path)) as fh:
            n_avail = 0
            for f_local, slot in frames:
                source_frame = pool_source_frame(f_local, a.pool_frame_offset)
                fh.seek(source_frame)
                fr_ = fh.read(1)      # TRR yields 5 fields, XTC 4; xyz is 0, box is 3
                xyz_nm, box_nm = fr_[0], fr_[3]
                place_seed(xyz_nm, box_nm, slot)
                n_avail += 1
            _log(f"  ...{case_dir.name}: {n_avail} pool seeds "
                 f"(frame offset {a.pool_frame_offset})")
    _log(f"[seeds] {len(seed_cache)} anchor seeds in {time.time()-t0:.0f} s")
    _log(f"[seeds] max centroid-removed geometry deviation {max(devs_seed):.2e} A "
         f"(tolerance 0.02) -- pure translation, rotation-free")
    _log(f"[seeds] recentring shift {np.mean(offs):.2f} +/- {np.std(offs):.2f} A; "
         f"anchor box margin after centring: min {min(margins_seed):.2f} A")

    # ---- expand each anchor seed into its stencil (logic copied from the v3 builder) -------
    base = a.mdp.read_text().splitlines()
    order = np.argsort(S_anchor, kind="stable")
    state_records = []
    written, margins, devs = 0, [], []
    gro_cache: dict[int, list] = {}
    t0 = time.time()
    for k in order:
        slot = int(S_anchor[k])
        seed, seed_box = seed_cache[slot]
        delta = S_R[k] - anchor_R[slot]
        tgt = seed + delta
        mg = pbc_margin(tgt, seed_box)
        if mg < a.pbc_margin:
            raise SystemExit(f"state {k}: bead within {mg:.1f} A of a box face "
                             f"(need {a.pbc_margin})")
        margins.append(mg)

        d = a.outdir / f"state_{int(k):06d}"
        d.mkdir(parents=True, exist_ok=True)
        if slot not in gro_cache:
            gro_cache.clear()
            gro_cache[slot] = (a.outdir / f"state_{slot:05d}" / "seed.gro").read_text().split("\n")
        lines = list(gro_cache[slot])
        for bi, ai0 in enumerate(bead_atoms0):
            ln = lines[ai0 + 2]
            xyz = np.array([float(ln[20:28]), float(ln[28:36]), float(ln[36:44])])
            xyz = xyz + delta[bi] / 10.0
            lines[ai0 + 2] = ln[:20] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + ln[44:]
        (d / "seed.gro").write_text("\n".join(lines))
        realized_target = np.array([
            [float(lines[ai0 + 2][20:28]), float(lines[ai0 + 2][28:36]),
             float(lines[ai0 + 2][36:44])]
            for ai0 in bead_atoms0], np.float64) * 10.0
        state_records.append(state_manifest_entry(
            state=k, anchor=slot, direction=S_dir[k], target=realized_target,
            anchor_source=None if src_states is None else src_states[slot],
            anchor_index=anchor_index[slot],
            multiplier=None if S_mult is None else S_mult[k],
            kind=None if S_kind is None else S_kind[k]))
        written += 1

        mdp = rewrite_production_mdp(
            base, ps_per_state=a.ps_per_state, output_ps=a.output_ps,
            gen_seed=a.seed + int(k))
        (d / "production.mdp").write_text("\n".join(mdp) + "\n")

        if a.verify:
            vl = (d / "seed.gro").read_text().split("\n")
            chk = np.array([[float(vl[i + 2][20:28]), float(vl[i + 2][28:36]),
                             float(vl[i + 2][36:44])] for i in bead_atoms0]) * 10.0
            devs.append(float(np.abs(chk - tgt).max()))

    n_anchor_dirs = 0
    for d in a.outdir.glob("state_?????"):
        if d.is_dir() and len(d.name) == len("state_") + 5:
            shutil.rmtree(d, ignore_errors=True); n_anchor_dirs += 1
    _log(f"[cleanup] removed {n_anchor_dirs} anchor-seed dirs (5-digit)")
    _log(f"[expand]  {written} state dirs in {time.time()-t0:.0f} s")
    _log(f"[verify]  max |written bead - intended target| = {max(devs):.4f} A (gro grid 0.01 A)")
    _log(f"[pbc]     min margin {min(margins):.2f} A")

    (a.outdir / "manifest.json").write_text(json.dumps(
        {"n_states": int(written), "n_anchors": int(len(anchors)),
         "anchor_source": "reference_trajectory_or_pool",
         "aa_traj": str(a.aa_traj), "pool_source_campaign": a.pool_source_campaign,
         "pool_case_template": a.pool_case_template,
         "pool_frame_offset": int(a.pool_frame_offset),
         "ps_per_state": a.ps_per_state, "output_ps": a.output_ps,
         "discard_ps": a.discard_ps,
         "frames_per_state": int(round(a.ps_per_state / a.output_ps)),
         "stencil_npz": str(a.stencil_npz),
         "max_seed_deviation_A": float(max(devs)), "min_pbc_margin_A": float(min(margins)),
         "states": state_records},
        indent=2) + "\n")
    _log(f"\nwrote {a.outdir}/manifest.json  ({written} states)")


if __name__ == "__main__":
    main()
