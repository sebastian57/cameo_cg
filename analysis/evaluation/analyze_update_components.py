#!/usr/bin/env python3
"""Summarize [UpdateFnComponents] timings from profiling logs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


LINE_RE = re.compile(r"\[UpdateFnComponents\]\s+(.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize UpdateFnComponents timing split from SLURM logs."
    )
    parser.add_argument("logs", nargs="+", help="Log files (e.g. outputs/slurm-*.out)")
    parser.add_argument(
        "--include-step0",
        action="store_true",
        help="Include step=0 warmup/compile samples (default: excluded).",
    )
    return parser.parse_args()


def to_number(raw: str):
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def main() -> int:
    args = parse_args()

    rows = []
    for p in args.logs:
        path = Path(p)
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        for line in path.read_text(errors="ignore").splitlines():
            m = LINE_RE.search(line)
            if not m:
                continue
            payload = m.group(1)
            row = {k: to_number(v) for k, v in KV_RE.findall(payload)}
            row["file"] = str(path)
            rows.append(row)

    if not rows:
        print("No [UpdateFnComponents] lines found.")
        return 1

    if not args.include_step0:
        rows = [r for r in rows if int(r.get("step", -1)) > 0]
        if not rows:
            print("Only step=0 samples found; rerun with --include-step0 if needed.")
            return 1

    groups = defaultdict(list)
    for r in rows:
        key = (r.get("mesh_size"), r.get("rank"), r.get("microbatch_count"))
        groups[key].append(r)

    print(
        "mesh rank micro n  "
        "local_ms  collective_ms  optimizer_ms  total_ms  "
        "local_%  collective_%  optimizer_%"
    )
    for key in sorted(groups):
        mesh, rank, micro = key
        rs = groups[key]
        local = mean(float(r["local_grad_total_ms"]) for r in rs)
        collective = mean(float(r["collective_total_ms"]) for r in rs)
        optimizer = mean(float(r["optimizer_total_ms"]) for r in rs)
        total = mean(float(r["sync_total_ms"]) for r in rs)
        denom = total if total > 0 else 1.0
        print(
            f"{mesh:>4} {rank:>4} {micro:>5} {len(rs):>2}  "
            f"{local:>8.3f} {collective:>14.3f} {optimizer:>12.3f} {total:>9.3f}  "
            f"{(100.0 * local / denom):>7.2f} {(100.0 * collective / denom):>12.2f} {(100.0 * optimizer / denom):>10.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
