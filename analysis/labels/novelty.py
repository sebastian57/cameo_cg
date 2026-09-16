#!/usr/bin/env python3
"""Did the campaign buy NEW structure, or re-sample what the reference already had?

For each campaign frame, the distance in the full pair-distance descriptor to its nearest
REFERENCE frame. If campaign frames sit closer to the reference than reference frames sit to
each other, the campaign bought redundancy, not information. The reference's own
nearest-neighbour distribution is the yardstick, so the comparison is self-calibrating.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping


def desc(R, pairs):
    return np.linalg.norm(R[:, pairs[:, 0], :] - R[:, pairs[:, 1], :], axis=-1)


def nn_to(Q, Ref, chunk=2000):
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        d = np.linalg.norm(Q[i:i+chunk, None, :] - Ref[None, :, :], axis=-1)
        out[i:i+chunk] = d.min(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", required=True)
    ap.add_argument("--n-ref", type=int, default=40000, help="reference subsample to search against")
    ap.add_argument("--n-query", type=int, default=6000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    D = np.load(a.assembled)
    R, origin = D["R"], D["origin"]
    nb = R.shape[1]
    pairs = np.array([(i, j) for i in range(nb) for j in range(i + 1, nb)], dtype=int)
    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(R); psi = m.cvs["psi"].evaluate(R)
    BAS = {"alphaR": (phi > -180) & (phi < 0) & (psi > -120) & (psi < 50),
           "beta":   (phi < 0) & ((psi > 100) | (psi < -150)),
           "alphaL": (phi > 0) & (phi < 120) & (psi > -50) & (psi < 100)}
    BAS["transition"] = ~(BAS["alphaR"] | BAS["beta"] | BAS["alphaL"])

    ref = origin == 0
    ridx = rng.choice(np.where(ref)[0], min(a.n_ref, int(ref.sum())), replace=False)
    Xref = desc(R[ridx], pairs)

    print(f"searching against {len(Xref)} reference frames, {len(pairs)}-D descriptor\n")
    print(f"{'set':26s} {'n':>6s} {'median NN dist':>15s} {'p10':>8s} {'p90':>8s}")
    print("-" * 68)

    def report(name, sel, exclude_self=False):
        idx = np.where(sel)[0]
        if len(idx) < 50:
            print(f"{name:26s} {len(idx):6d}   (too few)"); return
        q = rng.choice(idx, min(a.n_query, len(idx)), replace=False)
        d = nn_to(desc(R[q], pairs), Xref)
        if exclude_self:                      # a reference query finds itself at distance 0
            d = np.where(d < 1e-9, np.nan, d)
        print(f"{name:26s} {len(idx):6d} {np.nanmedian(d):15.4f} "
              f"{np.nanpercentile(d,10):8.4f} {np.nanpercentile(d,90):8.4f}")

    report("reference (self, yardstick)", ref, exclude_self=True)
    for k, mk in BAS.items():
        report(f"  reference / {k}", ref & mk, exclude_self=True)
    print()
    report("campaign (all)", origin > 0)
    for k, mk in BAS.items():
        report(f"  campaign / {k}", (origin > 0) & mk)
    print("\nA campaign median ABOVE the reference yardstick = genuinely new structure.")
    print("At or below it = the campaign re-sampled regions the reference already covered.")


if __name__ == "__main__":
    main()
