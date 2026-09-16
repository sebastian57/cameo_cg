#!/usr/bin/env python3
"""Generate a two-dimensional histogram FES from a two-column coordinate file.

The convention matches the historical charron utility: zero-count bins are assigned
10 kcal/mol, occupied bins are shifted so the most populated bin is zero, and the
result is written both as a PNG and as a blank-line-separated three-column table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="two-column data file")
    ap.add_argument("--output", type=Path, required=True, help="PNG output")
    ap.add_argument("--temperature-K", type=float, default=300.0)
    ap.add_argument("--bins-x", type=int, default=100)
    ap.add_argument("--bins-y", type=int, default=100)
    ap.add_argument("--xlabel", default="x")
    ap.add_argument("--ylabel", default="y")
    ap.add_argument("--table", type=Path)
    a = ap.parse_args(argv)
    rows = []
    for line in a.input.read_text().splitlines():
        if line.startswith(("#", "@")):
            continue
        fields = line.split()
        if len(fields) >= 2:
            rows.append((float(fields[0]), float(fields[1])))
    if not rows:
        raise SystemExit(f"no numeric two-column rows in {a.input}")
    values = np.asarray(rows, float)
    x, y = values.T
    H, xe, ye = np.histogram2d(x, y, bins=[a.bins_x, a.bins_y])
    kT = 0.0019872042586 * a.temperature_K
    occupied = H > 0
    fes = np.full_like(H, 10.0, dtype=float)
    fes[occupied] = -kT * np.log(H[occupied] / H[occupied].max())
    table = a.table or a.output.with_name(a.output.stem + "_FES.dat")
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w") as handle:
        for i in range(fes.shape[0]):
            for j in range(fes.shape[1]):
                handle.write(f"{0.5*(xe[i]+xe[i+1]):.8g}\t"
                             f"{0.5*(ye[j]+ye[j+1]):.8g}\t{fes[i,j]:.8g}\n")
            handle.write("\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots()
    im = ax.imshow(fes.T, cmap="inferno",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   origin="lower", aspect="auto")
    ax.set(title="Free Energy Surface", xlabel=a.xlabel, ylabel=a.ylabel)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im, cax=cax, label="dG [kcal/mol]")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.output} and {table}")


if __name__ == "__main__":
    main()
