#!/usr/bin/env python3
"""Control: is the log-K residual extra-dimensional information, or a density-gradient artifact?

K counts partners in a BALL, so it reflects density integrated over a neighbourhood,
while the comparison used each cell's POINT density. A density gradient alone would
leave a structured residual (edges high, interiors low) that has nothing to do with the
extra internal dimensions.

Test: recompute the correlation against 2D density SMOOTHED over a range of scales. If
the residual survives smoothing, it is extra-dimensional. If it collapses, it was the
projection geometry.
"""
import json, sys
from pathlib import Path
import numpy as np

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "local_work/md_analysis/ala2_kmap_stratified")
z = np.load(SRC/"ala2_kmap.npz"); j = json.load(open(SRC/"ala2_kmap.json"))
Kmap = np.array(z["Kmap"]); dens = np.array(z["dens"]).astype(float)
rel = j["kmap_reliability"]
dm = np.where(dens > 0, dens, np.nan)

def smooth_periodic(A, sigma_bins):
    """Gaussian smoothing on the periodic (phi,psi) torus, NaN-aware."""
    if sigma_bins <= 0: return A.copy()
    n = A.shape[0]
    k = np.arange(n); k = np.minimum(k, n-k)
    g = np.exp(-0.5*(k/sigma_bins)**2); g /= g.sum()
    G = np.outer(g, g)
    Gf = np.fft.fft2(np.fft.ifftshift(np.fft.fftshift(G)))
    M = np.isfinite(A).astype(float); B = np.where(np.isfinite(A), A, 0.0)
    num = np.real(np.fft.ifft2(np.fft.fft2(B)*np.fft.fft2(G)))
    den = np.real(np.fft.ifft2(np.fft.fft2(M)*np.fft.fft2(G)))
    out = np.where(den > 1e-9, num/np.maximum(den,1e-9), np.nan)
    return np.where(np.isfinite(A), out, np.nan)

deg_per_bin = 360.0/Kmap.shape[0]
print(f"grid {Kmap.shape[0]}x{Kmap.shape[0]}  ({deg_per_bin:.1f} deg/bin)   K-map reliability {rel:.3f}\n")
print(f"{'smoothing sigma':>16} {'cells':>7} {'r(logdens,logK)':>17} {'% RELIABLE K explained':>24}")
best = None
for sb in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    S = smooth_periodic(dm, sb)
    g = np.isfinite(Kmap) & np.isfinite(S) & (S > 0)
    if g.sum() < 20: continue
    lk = np.log10(np.maximum(Kmap[g],0.5)); ld = np.log10(S[g])
    r = float(np.corrcoef(ld, lk)[0,1]); expl = 100*min(r*r/rel, 1.0)
    print(f"{sb*deg_per_bin:13.1f} deg {int(g.sum()):7d} {r:17.3f} {expl:23.1f}%")
    if best is None or expl > best[1]: best = (sb*deg_per_bin, expl)
print(f"\nBEST smoothed density explains {best[1]:.1f}% of reliable K variance "
      f"(sigma = {best[0]:.1f} deg)")
print(f"-> residual NOT explained by 2D density at ANY smoothing scale: {100-best[1]:.1f}%")
print("\nIf this is close to the unsmoothed residual, the structure is extra-dimensional.")
print("If smoothing absorbs most of it, the residual was a density-gradient artifact.")
