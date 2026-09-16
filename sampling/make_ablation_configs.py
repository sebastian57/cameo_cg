"""Generate training configs for the Week-1 ablation suite.

Clones the gamma05 config (the current best-model reference point) and changes ONLY:
  data.path, paths.*, model_context, seed, optimizer.adabelief.{warmup_steps,decay_steps}

The LR schedule constants are derived from the four completed v3 runs, where they are exactly
linear in the training-row count (verified to 5 s.f. across 117k/197k/200k/250k):
    decay_steps  = round(0.1730711 * n_train)          n_train = round(0.9 * rows)
    warmup_steps = round(0.1098329 * decay_steps)
Everything else -- architecture, optimizer, gammas, epochs -- is byte-identical to gamma05, so
these runs are directly comparable to it and to stencil250k.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

TEMPLATE = "local_work/configs/ala2_bb6_v3_gamma05_wide160_fm1000.yaml"
ROOT = "/e/project1/cameo/schmidt36/cameo_cg"
DECAY_PER_ROW = 0.1730711
WARMUP_FRAC = 0.1098329

# (variant, seed) -> run name suffix.  Seed 20260728 is the project standard used by every
# completed v3 run; the two extra seeds on b50_d3 measure the training noise floor.
RUNS = [(v, 20260728, "") for v in (
    "b10_d0", "b10_d1", "b10_d3", "b10_d6",
    "b50_d1", "b50_d2", "b50_d3", "b50_d6",
    "b50_inner", "b50_outer", "b50_d2tica")]
RUNS += [("b50_d3", 20260729, "_s2"), ("b50_d3", 20260730, "_s3")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=Path("local_work/configs/ablation_week1"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    base = yaml.safe_load(Path(TEMPLATE).read_text())

    print(f"{'run':34s} {'rows':>8s} {'warmup':>7s} {'decay':>7s} {'est min':>8s}")
    total = 0.0
    for variant, seed, suffix in RUNS:
        npz = f"{ROOT}/local_work/input_data/ala2_bb6_v3abl_{variant}.npz"
        rows = int(np.load(npz)["R"].shape[0])
        n_train = round(0.9 * rows)
        decay = round(DECAY_PER_ROW * n_train)
        warmup = round(WARMUP_FRAC * decay)

        name = f"ala2_bb6_v3abl_{variant}{suffix}_wide160_fm1000"
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["seed"] = seed
        cfg["model_context"] = name
        cfg["data"]["path"] = npz
        out = f"{ROOT}/local_work/outputs/{name}"
        cfg["paths"] = {"output_dir": out, "export_dir": f"{out}/exports",
                        "checkpoint_dir": f"{out}/checkpoints",
                        "profile_dir": f"{out}/profiles", "slurm_dir": out}
        cfg["optimizer"]["adabelief"]["warmup_steps"] = warmup
        cfg["optimizer"]["adabelief"]["decay_steps"] = decay

        (a.outdir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        est = 2.23 * rows / 1000.0
        total += est
        print(f"{name.replace('ala2_bb6_v3abl_','').replace('_wide160_fm1000',''):34s} "
              f"{rows:8,d} {warmup:7d} {decay:7d} {est:8.0f}")

    print(f"\n{len(RUNS)} configs -> {a.outdir}")
    print(f"total {total:.0f} GPU-node-min = {total/60:.1f} node-hours "
          f"(~{total/60/5:.1f} h wall at 5 concurrent)")


if __name__ == "__main__":
    main()
