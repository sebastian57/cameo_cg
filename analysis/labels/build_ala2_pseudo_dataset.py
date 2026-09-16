#!/usr/bin/env python3
"""Emit an ala2 bb6 TRAINING dataset with pseudo mean-force labels + accuracy scores.

Each row carries
    F            pseudo mean-force  (falls back to the instantaneous label when K==0)
    K            number of partners averaged
    sigma_label  the accuracy score, sqrt(bias(eps)^2 + 3*sigma_n^2/K)
so a heteroscedastic loss can weight rows by 1/sigma_label^2. Rows with K==0 keep
their raw label and receive the raw error sqrt(3)*sigma_n -- honest, and they are
then down-weighted automatically rather than dropped, which preserves coverage.

Calibration constants come from `analysis/labels/ala2_kmap.py`
(DESIGN/PSEUDO_MEANFORCE_LABELS.md): sigma_n from the FLAT region of the CV curve,
bias(eps) from the same sweep. Defaults are the measured ala2 values.

Descriptor: 15 ORDERED pair distances + signed volume -- complete and
chirality-aware for 6 distinguishable beads. Partners are aligned with a
proper-rotation (det=+1) Kabsch and their forces rotated with the environment.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def descriptor(R):
    iu = np.triu_indices(R.shape[1], 1)
    d = np.linalg.norm(R[:, iu[0]] - R[:, iu[1]], axis=-1)
    v = np.einsum("ni,ni->n", np.cross(R[:, 1] - R[:, 0], R[:, 2] - R[:, 0]),
                  R[:, 4] - R[:, 0])[:, None] / 10.0
    return np.concatenate([d, v], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=None,
                    help="ala2_kmap.json; overrides --sigma-n/--bias if given")
    ap.add_argument("--eps", type=float, default=0.12)
    ap.add_argument("--sigma-n", type=float, default=12.343)
    ap.add_argument("--bias", type=float, default=None,
                    help="bias at --eps; read from --calibration when available")
    ap.add_argument("--n-rows", type=int, default=50000, help="training rows to emit")
    ap.add_argument("--min-sep", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import jax, jax.numpy as jnp

    sig_n, bias = a.sigma_n, a.bias
    if a.calibration and a.calibration.exists():
        j = json.load(open(a.calibration))
        eps_arr = np.array(j["eps"]); i = int(np.argmin(np.abs(eps_arr - a.eps)))
        sig_n = float(j["sigma_c"]); bias = float(j["bias"][i])
        print(f"calibration {a.calibration.name}: eps={eps_arr[i]:.3f} "
              f"sigma_n={sig_n:.3f} bias={bias:.3f}")
    if bias is None:
        raise SystemExit("--bias required when no --calibration is given")

    d = np.load(a.dataset)
    R = np.asarray(d["R"], np.float32); F = np.asarray(d["F"], np.float32)
    T, NB, _ = R.shape
    species = np.asarray(d["species"]) if "species" in d else None
    mask = np.asarray(d["mask"], np.float32) if "mask" in d else np.ones((T, NB), np.float32)
    print(f"{a.dataset.name}: {T} frames x {NB} beads; pool = all {T}, emitting {a.n_rows} rows")

    X = descriptor(R); Xj = jnp.asarray(X)

    @jax.jit
    def dists(qb):
        return jnp.sqrt(jnp.maximum(((qb[:, None, :] - Xj[None, :, :]) ** 2).sum(-1), 0.0))

    rng = np.random.default_rng(a.seed)
    q = np.sort(rng.choice(T, min(a.n_rows, T), replace=False))
    Fbar = np.zeros((len(q), NB, 3), np.float32)
    Kout = np.zeros(len(q), np.int32)

    for s in range(0, len(q), a.chunk):
        qi = q[s:s+a.chunk]
        dd = np.array(dists(jnp.asarray(X[qi])))
        dd[np.abs(qi[:, None] - np.arange(T)[None, :]) < a.min_sep] = np.inf
        for r, gi in enumerate(qi):
            idx = np.flatnonzero(dd[r] <= a.eps)
            if len(idx) == 0:
                Fbar[s+r] = F[gi]; Kout[s+r] = 0      # keep the raw label
                continue
            Rc = R[gi] - R[gi].mean(0)
            Rj = R[idx] - R[idx].mean(1, keepdims=True)
            H = np.einsum("kmi,mj->kij", Rj, Rc)
            U, S, Vt = np.linalg.svd(H)
            dsg = np.sign(np.linalg.det(np.einsum("kij,kjl->kil",
                                                  Vt.transpose(0,2,1), U.transpose(0,2,1))))
            D = np.zeros((len(idx), 3, 3)); D[:,0,0]=1; D[:,1,1]=1; D[:,2,2]=dsg
            Rot = np.einsum("kij,kjl,klm->kim", Vt.transpose(0,2,1), D, U.transpose(0,2,1))
            Fbar[s+r] = np.einsum("kij,kmj->kmi", Rot, F[idx]).mean(0)
            Kout[s+r] = len(idx)
        if (s // a.chunk) % 20 == 0:
            print(f"  {s}/{len(q)}")

    raw_err = float(np.sqrt(3.0) * sig_n)
    sigma_label = np.where(
        Kout > 0,
        np.sqrt(bias**2 + 3.0*sig_n**2/np.maximum(Kout, 1)),
        raw_err,                                    # K==0: honest raw-label error
    ).astype(np.float32)

    lab = Kout > 0
    print(f"\nlabelled {100*lab.mean():.1f}% of rows; median K = {np.median(Kout[lab]):.0f}")
    print(f"sigma_label: p5 {np.percentile(sigma_label,5):.2f}  p50 {np.percentile(sigma_label,50):.2f}  "
          f"p95 {np.percentile(sigma_label,95):.2f}  (raw label = {raw_err:.2f})")
    print(f"median gain vs raw: {raw_err/np.median(sigma_label):.2f}x")
    out = dict(R=R[q], F=Fbar, F_instantaneous=F[q], K=Kout, sigma_label=sigma_label,
               mask=mask[q], eps=np.float32(a.eps), sigma_n=np.float32(sig_n),
               bias=np.float32(bias), raw_label_error=np.float32(raw_err),
               origin=np.ones(len(q), np.int8))     # 1 = mean-force-like, matches assemble convention
    if species is not None:
        out["species"] = species[q].astype(np.int32)   # int32 to match existing training sets
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
