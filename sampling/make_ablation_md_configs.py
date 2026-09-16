"""Generate MD configs for the Week-1 ablation models.

Clones the gamma05 MD config so the protocol is byte-identical to every other bb6 run
(8 replicas x 3 ns, dt 2 fs, 298 K, gamma 5.0, same 8 reference start frames, seed 20260805).
Only the model paths and output names change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

TEMPLATE = "local_work/configs/md_ala2_bb6_v3_gamma05_8x3ns_dt2fs_multistart.yaml"
GROUP = "local_work/outputs/20260819_training_suite_ablation_week1"
MODEL_ID = "wide160_fm1000_sparse_t1024_bpd4_ga1"
VARIANTS = ["b10_d0", "b10_d1", "b10_d3", "b10_d6",
            "b50_d1", "b50_d2", "b50_d2tica", "b50_d3", "b50_d3_s2", "b50_d3_s3",
            "b50_d6", "b50_inner", "b50_outer"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("local_work/configs/ablation_week1_md"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path(TEMPLATE).read_text())

    for v in VARIANTS:
        ctx = f"ala2_bb6_v3abl_{v}_wide160_fm1000"
        exp = f"{GROUP}/{ctx}/exports/{ctx}_{MODEL_ID}"
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["md"]["training_config_path"] = f"{exp}_config.yaml"
        cfg["md"]["params_path"] = f"{exp}_params.pkl"
        tag = f"ala2_bb6_v3abl_{v}_8x3ns_dt2fs_multistart"
        cfg["md"]["output_dir"] = f"local_work/md_runs/{tag}"
        cfg["md"]["output_filename"] = f"traj_{tag}.npz"
        for k in ("training_config_path", "params_path"):
            if not Path(cfg["md"][k]).exists():
                raise SystemExit(f"MISSING {cfg['md'][k]}")
        (a.outdir / f"md_{tag}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"  {tag}")
    print(f"\n{len(VARIANTS)} MD configs -> {a.outdir}  ({len(VARIANTS)*8} replicas total)")


if __name__ == "__main__":
    main()
