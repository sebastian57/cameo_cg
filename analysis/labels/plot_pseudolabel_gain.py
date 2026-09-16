#!/usr/bin/env python3
"""Figures for the pseudo mean-force label scheme: the gain, and the new information."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("local_work/md_analysis/ala2_kmap_stratified")
OUT = Path("/e/project1/cameo/schmidt36/KNOWLEDGE_BASE/P_cameo_cg/FIGURES")
OUT.mkdir(parents=True, exist_ok=True)
D = "2026-08-26"

j = json.load(open(SRC / "ala2_kmap.json"))
z = np.load(SRC / "ala2_kmap.npz")
eps = np.array(j["eps"]); K = np.array(j["mean_K"]); bias = np.array(j["bias"])
err = np.array(j["err"]); gain = np.array(j["gain"]); raw = j["raw_label_error"]
sig_n = j["sigma_c"]
Kq = np.array(z["K"]); phi = np.array(z["phi"]); psi = np.array(z["psi"])
Kmap = np.array(z["Kmap"]); dens = np.array(z["dens"])
best = int(np.argmax(gain))

# ---------------------------------------------------------------- FIGURE 1
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

a = ax[0]
noise = np.sqrt(3.0 * sig_n**2 / np.maximum(K, 1))
bplot = np.where(bias > 0, bias, np.nan)      # clipped zeros are "below resolution", not 0.001
a.loglog(eps, bplot, "o-", color="C3", label=r"bias($\epsilon$)")
if (bias <= 0).any():          # may be empty once sigma_n is refined
    a.plot(eps[bias <= 0], np.full((bias <= 0).sum(), err[bias <= 0].min()*0.35), "v",
           color="C3", mfc="none", label="bias below resolution")
a.loglog(eps, noise, "s-", color="C0", label=r"noise $\sqrt{3}\sigma_n/\sqrt{K}$")
a.loglog(eps, err, "^-", color="k", lw=2.2, label="total label error")
a.axhline(raw, color="grey", ls="--", label=f"raw single label ({raw:.1f})")
a.axvline(eps[best], color="C2", ls=":", lw=2)
a.annotate(f"optimum {eps[best]:.2f} $\\AA$\n{gain[best]:.2f}x  (K={K[best]:.0f})",
           (eps[best], err[best]), textcoords="offset points", xytext=(10, -34),
           fontsize=9, color="C2", weight="bold")
a.set_xlabel(r"tolerance $\epsilon$  [$\AA$]"); a.set_ylabel("error  [kcal/mol/$\\AA$]")
a.set_title("A. Bias-variance trade-off (ala2 bb6)"); a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

b = ax[1]
# MEASURED per-eps sigma_n. Do NOT back-compute it from (err, bias): those are
# defined from one sigma_n, so that returns a flat line by construction.
sig_by_eps = np.array(j["sigma_n_by_eps"])
b.plot(eps, sig_by_eps, "o-", color="C4", lw=2)
b.axhline(sig_n, color="k", ls="--", label=f"$\\sigma_n$ = {sig_n:.2f} (flat region)")
flat = eps <= eps[best]
b.fill_between([eps.min(), eps[best]], 0, 100, color="C2", alpha=0.12)
b.text(eps.min()*1.02, sig_n*1.28, "flat: bias negligible\n(harmonic modes average exactly)",
       fontsize=8, color="C2")
b.text(eps[best]*1.05, sig_n*1.02, "rise = bias onset", fontsize=8, color="C3")
b.set_ylim(sig_n*0.85, max(sig_by_eps)*1.12)
b.set_xlabel(r"tolerance $\epsilon$  [$\AA$]"); b.set_ylabel(r"$\sigma_n$ from CV  [kcal/mol/$\AA$]")
b.set_title("B. Self-check: where bias begins"); b.legend(fontsize=8); b.grid(alpha=0.3)

c = ax[2]
kb = Kq[:, best].astype(float)
per = np.sqrt(bias[best]**2 + 3.0*sig_n**2/np.maximum(kb, 1))
per = per[kb > 0]
c.hist(per, bins=60, color="C0", alpha=0.85, label=f"pseudo-label (median {np.median(per):.1f})")
c.axvline(raw, color="grey", ls="--", lw=2, label=f"every raw label ({raw:.1f})")
c.axvline(np.median(per), color="C0", ls="-", lw=2)
c.set_xlabel("per-label error  [kcal/mol/$\\AA$]"); c.set_ylabel("frames")
c.set_title(f"C. The gain, per label ($\\epsilon$={eps[best]:.2f} $\\AA$, {100*(kb>0).mean():.0f}% labelled)")
c.text(0.97, 0.55, "frames with K=1 get\nno improvement", transform=c.transAxes,
       ha="right", fontsize=8, color="grey")
c.legend(fontsize=8); c.grid(alpha=0.3)
fig.suptitle("Pseudo mean-force labels from near-duplicate configurations — ala2 bb6, 200,001 frames", y=1.02)
fig.tight_layout(); fig.savefig(OUT / f"{D}_fig1_pseudolabel_gain.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- FIGURE 2
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
dm = np.where(dens > 0, dens, np.nan)
g = np.isfinite(Kmap) & np.isfinite(dm)
lk = np.log10(np.maximum(Kmap[g], 0.5)); ld = np.log10(dm[g])
sl, ic = np.polyfit(ld, lk, 1)
r = float(np.corrcoef(ld, lk)[0, 1]); rel = j["kmap_reliability"]

a = ax[0]
a.scatter(ld, lk, s=9, alpha=0.35, color="C0")
xx = np.linspace(ld.min(), ld.max(), 50)
a.plot(xx, sl*xx + ic, "k-", lw=2, label=f"fit  r={r:+.3f}")
a.set_xlabel("log10 reference density"); a.set_ylabel("log10 median K")
a.set_title(f"D. K vs 2D density\ndensity explains {100*r*r/rel:.1f}% of RELIABLE K variance")
a.legend(fontsize=8); a.grid(alpha=0.3)

resid = np.full_like(Kmap, np.nan)
resid[g] = lk - (sl*ld + ic)
b = ax[1]
v = np.nanpercentile(np.abs(resid), 98)
im = b.imshow(resid.T, origin="lower", extent=[-180,180,-180,180], aspect="auto",
              cmap="coolwarm", vmin=-v, vmax=v)
b.set_xlabel("phi [deg]"); b.set_ylabel("psi [deg]")
b.set_title(f"E. WHERE density fails\n(log K residual; {100*(1-r*r/rel):.1f}% of reliable variance)")
fig.colorbar(im, ax=b, label="log10 K  -  density prediction")

c = ax[2]
c.imshow(np.log10(np.maximum(Kmap, 0.5)).T, origin="lower", extent=[-180,180,-180,180],
         aspect="auto", cmap="viridis")
lowk = resid < -np.nanpercentile(np.abs(resid), 85)
yy, xx2 = np.where(lowk.T)
edg = np.linspace(-180, 180, Kmap.shape[0]+1); ctr = 0.5*(edg[1:]+edg[:-1])
c.scatter(ctr[xx2], ctr[yy], s=18, facecolors="none", edgecolors="red", lw=1.2,
          label="fewer partners than density predicts")
c.set_xlabel("phi [deg]"); c.set_ylabel("psi [deg]")
c.set_title("F. Free labels fail here\n(target for restrained sampling)")
c.legend(fontsize=8, loc="upper right")
fig.suptitle("Is the K-map new information, or just density? — ala2 bb6", y=1.02)
fig.tight_layout(); fig.savefig(OUT / f"{D}_fig2_kmap_vs_density.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / f"{D}_fig1_pseudolabel_gain.png")
print("wrote", OUT / f"{D}_fig2_kmap_vs_density.png")
print(f"optimum eps={eps[best]:.2f} gain={gain[best]:.2f}x  sigma_n={sig_n:.2f}  raw={raw:.2f}")
