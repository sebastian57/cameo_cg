#!/usr/bin/env python3
"""Measure the phi~0 energy barrier that separates the two (degenerate) enantiomer wells.

Both models are exactly inversion-invariant (chirality_head_enabled defaults False, so the
energy readout takes only even-parity scalars). The phi>0 basin is therefore EXACTLY as deep
as phi<0, and what keeps a model in the correct well is purely KINETIC: the barrier at phi~0.

The reference trajectory never visits phi~0, so reference structures cannot probe it. The
weighted model's MD did visit it, so those frames are used as the common probe set and BOTH
models are evaluated on the SAME structures -- the comparison is then a property of the two
energy functions, not of their sampling.

E_min(phi) per bin approximates the lowest-energy path; with a finite sample it is an UPPER
bound on the true barrier, but the bound is identical for both models.
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.labels.chirality_test import load_model
from sampling.mapping import get_mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="LABEL=cfg.yaml,params.pkl")
    ap.add_argument("--probe-glob", required=True, help="MD npz files spanning phi~0")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--n-frames", type=int, default=6000)
    ap.add_argument("--bins", type=int, default=36)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)
    import jax.numpy as jnp

    files = sorted(glob.glob(a.probe_glob))
    R = np.concatenate([np.load(f)["R"] for f in files[:40]], axis=0).astype(np.float32)
    idx = np.linspace(0, len(R) - 1, min(a.n_frames, len(R))).astype(int)
    R = R[idx]
    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(R)
    print(f"probe set: {len(R)} frames from {len(files[:40])} files; "
          f"phi coverage {phi.min():.0f}..{phi.max():.0f} deg, "
          f"{100*np.mean(np.abs(phi) < 20):.1f}% within |phi|<20")

    edges = np.linspace(-180, 180, a.bins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    curves = {}
    for spec in a.model:
        label, paths = spec.split("=", 1); cfgp, parp = paths.split(",", 1)
        model, params, mask0, species0, pre, box, shift, Rref, maskref = load_model(cfgp, parp, a.reference)
        efn = model.energy_fn_template(params); nbrs = model.ml_model.nbrs_init
        Rp = pre.center_and_park(R, np.ones(R.shape[:2], np.float32), box, shift)
        E = np.empty(len(Rp))
        for k in range(len(Rp)):
            x = jnp.asarray(Rp[k])
            nb = model.ml_model.nneigh_fn.update(x, nbrs, mask=mask0)
            E[k] = float(efn(x, neighbor=nb, mask=mask0, species=species0))
        prof = np.full(a.bins, np.nan)
        ib = np.clip(np.digitize(phi, edges) - 1, 0, a.bins - 1)
        for b in range(a.bins):
            s = ib == b
            if s.sum() >= 5:
                prof[b] = np.percentile(E[s], 5)      # robust stand-in for E_min
        prof -= np.nanmin(prof)
        curves[label] = prof
        neg = ctr < -30; near0 = np.abs(ctr) <= 20
        well = np.nanmin(prof[neg]) if np.isfinite(prof[neg]).any() else np.nan
        barr = np.nanmin(prof[near0]) if np.isfinite(prof[near0]).any() else np.nan
        print(f"\n{label}")
        print(f"   phi<0 well minimum      {well:7.2f} kcal/mol")
        print(f"   lowest E at |phi|<=20   {barr:7.2f} kcal/mol")
        print(f"   ==> phi~0 BARRIER       {barr-well:7.2f} kcal/mol   "
              f"({(barr-well)/0.5922:5.1f} kT)")

    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, prof in curves.items():
        ax.plot(ctr, prof, "o-", lw=2, label=lab)
    ax.axvspan(-20, 20, color="grey", alpha=0.15, label="phi~0 barrier region")
    ax.set_xlabel("phi [deg]"); ax.set_ylabel("E - min  [kcal/mol]")
    ax.set_title("phi~0 barrier: what keeps an ACHIRAL model in the correct enantiomer well\n"
                 "(both models evaluated on the SAME structures)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.outdir / "phi_barrier.png", dpi=150); plt.close(fig)
    print(f"\nwrote {a.outdir/'phi_barrier.png'}")


if __name__ == "__main__":
    main()
