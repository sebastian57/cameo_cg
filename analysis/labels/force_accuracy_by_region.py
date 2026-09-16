#!/usr/bin/env python3
"""Force accuracy per Ramachandran region, against INDEPENDENT mean-force labels.

Test set: the v4 stencil mean forces (`local_work/v4_stencil_meanforce.npz`), which carry a
measured SE ~1.0 kcal/mol/A per state -- i.e. the label noise floor is known. Neither the
pseudo-label models nor reference200k were trained on these, so the comparison is fair for
them. v4 WAS trained on them and will look artificially good; it is included only for scale.

The question this answers: does a 50k pseudo-mean-force dataset buy the accuracy of a 200k
raw-frame dataset, at 1/4 the training cost?
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.labels.chirality_test import load_model
from sampling.mapping import get_mapping

REGIONS = {
    "alphaR":  lambda p, s: (p > -180) & (p < 0) & (s > -120) & (s < 50),
    "beta":    lambda p, s: (p < 0) & ((s > 100) | (s < -150)),
    "alphaL":  lambda p, s: (p > 0) & (p < 120) & (s > -50) & (s < 100),
    "phi>0":   lambda p, s: p > 0,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="LABEL=cfg.yaml,params.pkl")
    ap.add_argument("--test", default="local_work/v4_stencil_meanforce.npz")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--n-test", type=int, default=4000)
    a = ap.parse_args()
    import jax, jax.numpy as jnp

    t = np.load(a.test)
    Rt = np.asarray(t["R"], np.float32); Ft = np.asarray(t["F"], np.float32)
    SE = np.sqrt((np.asarray(t["SE"], np.float32) ** 2).mean(axis=(1, 2)))
    idx = np.linspace(0, len(Rt) - 1, min(a.n_test, len(Rt))).astype(int)
    Rt, Ft, SE = Rt[idx], Ft[idx], SE[idx]
    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(Rt); psi = m.cvs["psi"].evaluate(Rt)
    masks = {k: f(phi, psi) for k, f in REGIONS.items()}
    print(f"test set: {len(Rt)} stencil mean-force states; label noise floor "
          f"SE = {np.median(SE):.2f} kcal/mol/A (median)")
    print("region sizes: " + "  ".join(f"{k} {int(v.sum())}" for k, v in masks.items()))
    print(f"\n{'model':22s} {'ALL':>8} " + " ".join(f"{k:>8}" for k in REGIONS))
    print(f"{'':22s} " + " ".join(f"{'RMSE':>8}" for _ in range(len(REGIONS) + 1)))

    for spec in a.model:
        label, paths = spec.split("=", 1); cfgp, parp = paths.split(",", 1)
        model, params, mask0, species0, pre, box, shift, _, _ = load_model(cfgp, parp, a.reference)
        efn = model.energy_fn_template(params); nbrs = model.ml_model.nbrs_init
        Rp = pre.center_and_park(Rt, np.ones(Rt.shape[:2], np.float32), box, shift)

        def force_of(x, nb):
            g = jax.grad(lambda y: efn(y, neighbor=nb, mask=mask0, species=species0))(x)
            return -g
        Fpred = np.empty_like(Ft)
        for k in range(len(Rp)):
            x = jnp.asarray(Rp[k])
            nb = model.ml_model.nneigh_fn.update(x, nbrs, mask=mask0)
            Fpred[k] = np.asarray(force_of(x, nb))
        err = np.sqrt(((Fpred - Ft) ** 2).mean(axis=(1, 2)))
        row = f"{label:22s} {np.sqrt((err**2).mean()):8.2f} "
        row += " ".join(f"{np.sqrt((err[v]**2).mean()):8.2f}" if v.sum() > 20 else f"{'--':>8}"
                        for v in masks.values())
        print(row)
    print(f"\nRMSE is vs mean-force labels whose own noise is ~{np.median(SE):.2f}; a model at that")
    print("level is at the test-set floor. Larger numbers are model error, not label noise.")


if __name__ == "__main__":
    main()
