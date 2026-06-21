#!/usr/bin/env python3
"""Run multiple MD replicas in parallel, optionally oversubscribing GPUs.

Spawns child processes pinned to GPU slots via CUDA_VISIBLE_DEVICES.
Each child calls run_md.py --replica i and writes its own _repXX.npz file.
All replicas start from the same initial coordinates but use different random seeds
(base_seed + i), giving independent trajectories that can be concatenated for TICA.

When n_replicas exceeds the wave size, replicas are run in sequential waves.
The wave size is n_gpus * procs_per_gpu. For example, n_replicas=8 on 4 GPUs
with procs_per_gpu=1 runs replicas 0-3, then replicas 4-7. With
procs_per_gpu=2, all 8 replicas run in one wave, two processes per GPU.

Requirements:
  - The YAML config must have n_replicas set (read automatically from the config).
  - n_steps and all other MD parameters are shared across replicas.

Usage:
  # Inside a SLURM job with --gres=gpu:4 (reads n_replicas from config):
  python scripts/run_md_parallel.py configs/my_md.yaml

  # Override GPU count (e.g. if CUDA_VISIBLE_DEVICES is not set):
  python scripts/run_md_parallel.py configs/my_md.yaml --n-gpus 4

  # Run 4 replica processes per GPU (16 total processes on 4 GPUs):
  python scripts/run_md_parallel.py configs/my_md.yaml --n-gpus 4 --procs-per-gpu 4

  # Dry-run to see the commands without launching:
  python scripts/run_md_parallel.py configs/my_md.yaml --dry-run

Output:
  Replica i writes to  <output_dir>/<stem>_rep<i:02d>.npz  (same as run_md.py
  --replica mode).  Pass all _repXX.npz files to analyze_traj.py --npz to
  concatenate them for TICA fitting or projection.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_n_gpus() -> int:
    """Return number of visible GPUs from env or nvidia-smi."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in ("NoDevFiles", "-1", ""):
        return len(visible.split(","))
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode()
        n = len(out.strip().splitlines())
        return n if n > 0 else 1
    except Exception:
        return 1


def _read_n_replicas(config_file: str) -> int:
    """Read n_replicas from the md: section of the YAML config."""
    try:
        import yaml
        with open(config_file) as f:
            raw = yaml.safe_load(f)
        return int(raw.get("md", {}).get("n_replicas", 1))
    except Exception:
        return 1


def _build_waves(n_replicas: int, n_gpus: int, procs_per_gpu: int) -> list[list[int]]:
    """Return replica-index waves for the requested GPU oversubscription."""
    if n_gpus < 1:
        raise ValueError("n_gpus must be >= 1")
    if procs_per_gpu < 1:
        raise ValueError("procs_per_gpu must be >= 1")
    wave_size = n_gpus * procs_per_gpu
    return [
        list(range(start, min(start + wave_size, n_replicas)))
        for start in range(0, n_replicas, wave_size)
    ]


def _run_wave(
    wave: list[int],
    n_gpus: int,
    run_md: str,
    python: str,
    config_file: str,
    job_id: str,
    log_dir: Path,
    dry_run: bool,
) -> list[int]:
    """Launch one wave of replicas (one per GPU) and wait for completion.

    Returns the list of replica indices that failed.
    """
    procs: list[tuple[int, subprocess.Popen, object]] = []
    for slot, rep_idx in enumerate(wave):
        gpu_id = slot % n_gpus
        env = os.environ.copy()
        # Pin process to GPU slot.  With SLURM --gres=gpu:N, CUDA labels the
        # allocated GPUs as 0..N-1 regardless of physical index.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Override the login-node CPU guard in run_md.py.
        env["JAX_PLATFORMS"] = "cuda"

        cmd = [python, run_md, config_file, job_id, "--replica", str(rep_idx)]

        if dry_run:
            print(f"  [dry] replica {rep_idx}: GPU={gpu_id}  {' '.join(cmd)}")
            continue

        log_path = log_dir / f"replica_{rep_idx:02d}_{job_id}.log"
        log_fh = log_path.open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((rep_idx, proc, log_fh))
        print(f"  replica {rep_idx} → GPU {gpu_id}  pid={proc.pid}  log={log_path.name}")

    if dry_run:
        return []

    failed: list[int] = []
    for rep_idx, proc, log_fh in procs:
        rc = proc.wait()
        log_fh.close()
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  replica {rep_idx} {status}")
        if rc != 0:
            failed.append(rep_idx)
    return failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config_file", help="MD YAML config with an 'md: n_replicas:' field")
    ap.add_argument(
        "--n-gpus",
        type=int,
        default=None,
        metavar="N",
        help="GPUs available. Default: auto-detect from CUDA_VISIBLE_DEVICES / nvidia-smi.",
    )
    ap.add_argument(
        "--procs-per-gpu",
        type=int,
        default=1,
        metavar="N",
        help="Replica processes to run concurrently on each GPU (default: 1).",
    )
    ap.add_argument(
        "--job-id",
        default=None,
        metavar="ID",
        help="Job ID for output filenames (default: SLURM_JOB_ID or 'local').",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for per-replica log files (default: slurm/ in project root).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching.",
    )
    args = ap.parse_args()

    n_gpus = args.n_gpus if args.n_gpus is not None else _detect_n_gpus()
    if n_gpus < 1:
        raise SystemExit("--n-gpus must be >= 1")
    if args.procs_per_gpu < 1:
        raise SystemExit("--procs-per-gpu must be >= 1")
    n_replicas = _read_n_replicas(args.config_file)
    job_id = args.job_id or os.environ.get("SLURM_JOB_ID", "local")

    run_md = str(Path(__file__).resolve().parent / "run_md.py")
    python = sys.executable
    log_dir = args.log_dir or Path(__file__).resolve().parent.parent / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)

    waves = _build_waves(n_replicas, n_gpus, args.procs_per_gpu)
    n_waves = len(waves)
    wave_size = n_gpus * args.procs_per_gpu
    print(
        f"[run_md_parallel] {n_replicas} replicas  {n_gpus} GPUs  "
        f"{args.procs_per_gpu} proc/GPU  wave_size={wave_size}  "
        f"→ {n_waves} wave(s)  (job_id={job_id})"
    )
    if n_replicas < wave_size:
        print(
            f"  Note: n_replicas={n_replicas} < wave_size={wave_size}. "
            f"Only {n_replicas} replicas will run in the single wave."
        )

    t0 = time.perf_counter()
    all_failed: list[int] = []

    for wave_num, wave in enumerate(waves):
        if n_waves > 1:
            print(f"\n── Wave {wave_num + 1}/{n_waves}: replicas {wave} ──")
        failed = _run_wave(
            wave, n_gpus, run_md, python, args.config_file,
            job_id, log_dir, args.dry_run,
        )
        all_failed.extend(failed)
        if failed and not args.dry_run:
            print(f"  Wave {wave_num + 1} had failures: {failed}. Continuing remaining waves.")

    if args.dry_run:
        return

    total = time.perf_counter() - t0
    if all_failed:
        print(
            f"\n[run_md_parallel] {len(all_failed)} replica(s) FAILED: {all_failed}",
            file=sys.stderr,
        )
        print(f"  Check logs in {log_dir}/ for details.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[run_md_parallel] All {n_replicas} replicas complete in {total:.0f}s")
    print(
        f"\nNext: fit TICA on all replicas:\n"
        f"  python md/analyze_traj.py \\\n"
        f"      --npz <output_dir>/*_rep*.npz \\\n"
        f"      --outdir <results_dir> --prefix md_parallel\n"
        f"\nOr project onto reference TICA — see analyze_traj.py --help for\n"
        f"--reference-model / --reference-pairs flags."
    )


if __name__ == "__main__":
    main()
