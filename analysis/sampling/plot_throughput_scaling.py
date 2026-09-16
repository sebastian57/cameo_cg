#!/usr/bin/env python3
"""Replica-packing scaling for the native PLUMED sampling stack.

    python -m sampling.plot_throughput_scaling --root local_work/throughput_mps \
        --compare local_work/throughput --outdir local_work/throughput_mps

Reads `per<N>/replica_*/biased.log` under each root, pulls GROMACS' own Performance and
cycle-accounting numbers, and plots per-replica speed, aggregate node throughput, and the
GPU timings that explain where the ceiling comes from.

WHY THE COMPARISON MATTERS
    The first sweep ran WITHOUT CUDA MPS and concluded that packing past ~4 replicas per node
    did not pay. That was a misconfiguration, not a hardware limit: without MPS, processes
    sharing a GPU time-slice rather than running concurrently. Passing `--compare` overlays
    the two so the difference is visible rather than asserted.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from pathlib import Path

_PATS = {
    "ns_day": r"Performance:\s+([\d.]+)",
    "pme_gpu": r"PME GPU mesh\s+\d+\s+\d+\s+\d+\s+([\d.]+)",
    "wait_gpu": r"Wait GPU NB local\s+\d+\s+\d+\s+\d+\s+([\d.]+)",
    "force": r"\n Force\s+\d+\s+\d+\s+\d+\s+([\d.]+)",
}


def harvest(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.glob("per*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        n = int(re.sub(r"\D", "", d.name) or 0)
        acc = {k: [] for k in _PATS}
        logs = sorted(glob.glob(str(d / "replica_*" / "biased.log")))
        for f in logs:
            t = Path(f).read_text()
            for k, pat in _PATS.items():
                m = re.search(pat, t)
                if m:
                    acc[k].append(float(m.group(1)))
        if not acc["ns_day"]:
            continue
        rec = {"reps_per_node": n, "n_logs": len(acc["ns_day"]),
               "rerun_ok": len(glob.glob(str(d / "replica_*" / "unbiased_forces.trr")))}
        for k, v in acc.items():
            rec[k] = st.median(v) if v else float("nan")
        rec["ns_day_node"] = n * rec["ns_day"]
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="sweep dir with per<N>/ inside")
    ap.add_argument("--compare", type=Path, default=None, help="second sweep to overlay")
    ap.add_argument("--label", default="MPS on")
    ap.add_argument("--compare-label", default="MPS off")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_rows = harvest(a.root)
    cmp_rows = harvest(a.compare) if a.compare else []
    if not main_rows:
        raise SystemExit(f"no biased.log with a Performance line under {a.root}")
    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / "scaling_summary.json").write_text(
        json.dumps({a.label: main_rows, a.compare_label: cmp_rows}, indent=1))

    x = [r["reps_per_node"] for r in main_rows]
    solo = main_rows[0]["ns_day"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))

    ax[0].plot(x, [r["ns_day"] for r in main_rows], "o-", lw=2, label=a.label)
    if cmp_rows:
        ax[0].plot([r["reps_per_node"] for r in cmp_rows],
                   [r["ns_day"] for r in cmp_rows], "s--", color="crimson",
                   label=a.compare_label)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(x); ax[0].set_xticklabels(x)
    ax[0].set_xlabel("replicas per node"); ax[0].set_ylabel("ns/day per replica")
    ax[0].set_title("per-replica speed"); ax[0].grid(alpha=.3); ax[0].legend()

    ax[1].plot(x, [r["ns_day_node"] for r in main_rows], "o-", lw=2, label=a.label)
    if cmp_rows:
        ax[1].plot([r["reps_per_node"] for r in cmp_rows],
                   [r["ns_day_node"] for r in cmp_rows], "s--", color="crimson",
                   label=a.compare_label)
    ax[1].plot(x, [solo * n for n in x], ":", color="grey", label="ideal (linear)")
    best = max(main_rows, key=lambda r: r["ns_day_node"])
    ax[1].annotate(f"peak {best['ns_day_node']:.0f} ns/day\n@ {best['reps_per_node']}/node",
                   (best["reps_per_node"], best["ns_day_node"]),
                   textcoords="offset points", xytext=(-15, -42), fontsize=9,
                   arrowprops=dict(arrowstyle="->", lw=.8))
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xticks(x); ax[1].set_xticklabels(x)
    ax[1].set_xlabel("replicas per node"); ax[1].set_ylabel("aggregate ns/day per node")
    ax[1].set_title("node throughput"); ax[1].grid(alpha=.3, which="both"); ax[1].legend()

    # the GPU terms are what actually sets the ceiling; CPU-side Force is flat
    for key, lab in (("pme_gpu", "PME GPU mesh"), ("wait_gpu", "Wait GPU NB local"),
                     ("force", "Force (CPU, incl. PLUMED)")):
        ax[2].plot(x, [r[key] for r in main_rows], "o-", label=lab)
    ax[2].set_xscale("log", base=2); ax[2].set_xticks(x); ax[2].set_xticklabels(x)
    ax[2].set_xlabel("replicas per node"); ax[2].set_ylabel("wall seconds (20 ps run)")
    ax[2].set_title("where the ceiling comes from"); ax[2].grid(alpha=.3); ax[2].legend()

    fig.tight_layout()
    fig.savefig(a.outdir / "fig_throughput_scaling.png", dpi=130)
    plt.close(fig)

    print(f"{'reps/node':>9} {'ns/day each':>12} {'ns/day node':>12} {'speedup':>8} {'%solo':>7} "
          f"{'rerun':>8}")
    for r in main_rows:
        print(f"{r['reps_per_node']:>9} {r['ns_day']:>12.1f} {r['ns_day_node']:>12.0f} "
              f"{r['ns_day_node']/solo:>7.2f}x {r['ns_day']/solo*100:>6.1f}% "
              f"{r['rerun_ok']:>4}/{r['n_logs']}")
    print(f"\npeak aggregate: {best['ns_day_node']:.0f} ns/day at "
          f"{best['reps_per_node']} replicas/node")
    print(f"wrote {a.outdir}/fig_throughput_scaling.png")


if __name__ == "__main__":
    main()
