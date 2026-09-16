#!/usr/bin/env python3
"""Materialize one v5.1 AA campaign arm and generate GPU-packed Slurm launchers.

The archive contains seed.gro files and a manifest. Every archived production.mdp is
rewritten from the validated AA reference NPT MDP before launch, so stale pilot settings
cannot enter production. Each 64-state, four-GPU group collects its labels and cleans its
accepted trajectories before the next group starts.

This script prepares files only. It does not submit the generated Slurm array.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from sampling.build_stencil_campaign_v4 import format_seed_gro, rewrite_production_mdp
from sampling.launch import cpus_per_rank, group_ranges, multidir_group_script, submit_script

REPO = Path("/e/project1/cameo/schmidt36/cameo_cg")
DEFAULT_CAMPAIGN_ROOT = REPO / "local_work/v51_campaigns"
DEFAULT_LABELS_ROOT = REPO / "local_work/v51_labels"
VENV = Path("/e/project1/cameo/schmidt36/venv_cameocg_jupiter2026")
PROJECT_TMPDIR = REPO / "local_work/tmp"
REFERENCE_MDP = Path("/e/project1/cameo/edelkoetter2/work/ala2/constrained/production/NPT_production.mdp")
AA_TOP = Path("/e/project1/cameo/edelkoetter2/work/ala2/constrained/production/md_ala2.gro")
TOPOLOGY = Path("/e/project1/cameo/edelkoetter2/work/ala2/constrained/topol.top")
INDEX = REPO / "local_work/input_data/ala2_bb6_frozen_beads.ndx"
WEIGHTS = REPO / "local_work/input_data/ala2_bb6_aggforce_weight_matrix.npz"
ARCHIVE_ROOT = REPO / "local_work/sampling_runs/ala2_bb6_v51_flow_metad_beta_alphaR/v51_campaign_archives"
MAX_CONCURRENT_TASKS = 6
ARCHIVES = {
    "basin_only": ARCHIVE_ROOT / "basin_only.tar.gz",
    "basin_ridge": ARCHIVE_ROOT / "basin_ridge.tar.gz",
    "full_region_control": ARCHIVE_ROOT / "full_region_control.tar.zst",
}


def extract_archive(archive: Path, campaign_root: Path, arm: str) -> Path:
    expected = campaign_root / arm
    if expected.exists():
        raise SystemExit(f"refusing to overwrite existing campaign: {expected}")
    campaign_root.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zst":
        cmd = ["tar", "--use-compress-program=zstd", "-xf", str(archive), "-C", str(campaign_root)]
    else:
        cmd = ["tar", "-xzf", str(archive), "-C", str(campaign_root)]
    subprocess.run(cmd, check=True)
    if not expected.is_dir():
        raise SystemExit(f"archive did not create expected campaign root {expected}")
    return expected


def rewrite_state_inputs(campaign: Path, reference_mdp: Path, reference_gro: Path,
                         n_states: int) -> None:
    base = reference_mdp.read_text().splitlines()
    gro_template = reference_gro.read_text().splitlines()
    states = sorted(campaign.glob("state_*"))
    if len(states) != n_states:
        raise SystemExit(f"manifest says {n_states} states, found {len(states)}")
    for state in states:
        k = int(state.name.split("_")[1])
        mdp = rewrite_production_mdp(base, ps_per_state=3.8, output_ps=0.2,
                                     gen_seed=20260820 + k)
        (state / "production.mdp").write_text("\n".join(mdp) + "\n")

        seed = state / "seed.gro"
        lines = seed.read_text().splitlines()
        n_atoms = int(lines[1].strip())
        if n_atoms != int(gro_template[1].strip()):
            raise SystemExit(f"{seed}: atom count differs from reference GRO")
        xyz = np.asarray([[float(line[20:28]), float(line[28:36]),
                           float(line[36:44])] for line in lines[2:2 + n_atoms]])
        box = np.fromstring(lines[2 + n_atoms], sep=" ")
        seed.write_text("\n".join(format_seed_gro(gro_template, xyz, box)) + "\n")


def render_collector_block(campaign: Path, label_out: Path, start: int, stop: int) -> str:
    """Render the post-GROMACS collector with its own Python module environment."""
    return f"""
# ---- 5. collect this group, verify the shard, then clean accepted trajectories --------
(
cd "{REPO}"
source "{REPO}/env_setup/load_modules_2026.sh"
source "{VENV}/bin/activate"
export PYTHONUNBUFFERED=1
python -m sampling.collect_meanforce --campaign "{campaign}" --weights "{WEIGHTS}" --out "{label_out}" --discard-ps 1.0 --output-ps 0.2 --fast --cleanup --state-start {start} --state-stop {stop}
)
"""


def make_group_scripts(campaign: Path, labels: Path, n_states: int, topology: Path) -> int:
    # The generated scripts enter campaign before using case_dirs; resolve here so direct
    # callers cannot reproduce the relative-path failure from job 1499238.
    campaign = Path(campaign).resolve()
    labels = Path(labels).resolve()
    topology = Path(topology).resolve()
    groups = group_ranges(n_states, 64)
    ntomp = cpus_per_rank(max(len(group) for group in groups))
    case_dirs = [str(campaign / f"state_{k:06d}") for k in range(n_states)]
    labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INDEX, campaign / "beads.ndx")
    for gi, members in enumerate(groups):
        script = multidir_group_script(
            index="../beads.ndx", plumed=False,
            case_dirs=[case_dirs[k] for k in members],
            structure_for=["seed.gro"] * len(members),
            topology=str(topology), ntomp=ntomp, n_gpus=4,
            use_server=False, mps=True, trim=True)
        start, stop = members[0], members[-1] + 1
        label_out = labels / f"shard_{gi:04d}.npz"
        script += render_collector_block(campaign, label_out, start, stop)
        path = campaign / f"run_group_{gi:04d}.sh"
        path.write_text(script)
        path.chmod(0o755)
    submit = submit_script(
        campaign_dir=campaign, groups=groups, job_name=f"v51-{campaign.name}",
        hours=2.0, n_gpus=4, groups_per_task=8,
        max_concurrent_tasks=MAX_CONCURRENT_TASKS)
    submit = submit.replace(
        "set -Eeuo pipefail\n",
        f"set -Eeuo pipefail\nexport TMPDIR={PROJECT_TMPDIR}\nmkdir -p \"$TMPDIR\"\n",
        1,
    )
    (campaign / "submit.slurm").write_text(submit)
    return len(groups)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARCHIVES), required=True)
    ap.add_argument("--archive", type=Path, default=None)
    ap.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    ap.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    ap.add_argument("--reference-mdp", type=Path, default=REFERENCE_MDP)
    args = ap.parse_args()

    archive = (args.archive or ARCHIVES[args.arm]).resolve()
    campaign = extract_archive(archive, args.campaign_root.resolve(), args.arm)
    manifest = json.loads((campaign / "manifest.json").read_text())
    n_states = int(manifest["n_states"])
    rewrite_state_inputs(campaign, args.reference_mdp.resolve(), AA_TOP, n_states)
    n_groups = make_group_scripts(campaign, (args.labels_root / args.arm).resolve(),
                                  n_states, TOPOLOGY)
    tasks = (n_groups + 8 - 1) // 8
    print(f"prepared {args.arm}: {n_states} states, {n_groups} groups, {tasks} array tasks")
    print(f"campaign: {campaign}")
    print(f"labels:   {(args.labels_root / args.arm).resolve()}")
    print(f"submit after the one-state pilot passes: sbatch {campaign / 'submit.slurm'}")


if __name__ == "__main__":
    main()
