"""Compare batched (vmap) MD output against the reference MDRunner output.

Both routes are run from configs that differ ONLY in `batched:`, so any
disagreement beyond float32 round-off is a bug in md/runner_vmap.py.

Chaos caveat: Langevin dynamics is chaotic, so tiny float differences grow
exponentially. Equivalence is therefore judged on the FIRST recorded frame
(where divergence has had least time to amplify) and reported as a per-frame
trace so growth is visible rather than hidden behind a single pass/fail.
"""
import sys
from pathlib import Path
import numpy as np

d = Path("local_work/md_runs/vmap_eqtest")
n_rep, fails = 4, []
print(f"{'rep':>3} {'frame':>5} {'max|dR| (A)':>13} {'rms|dR| (A)':>13} {'rel':>10}")
print("-" * 50)
for i in range(n_rep):
    a = np.load(d / f"eqtest_step_ref_rep{i:02d}.npz")["R"].astype(np.float64)
    b = np.load(d / f"eqtest_step_vmap_rep{i:02d}.npz")["R"].astype(np.float64)
    if a.shape != b.shape:
        print(f"  rep{i}: SHAPE MISMATCH ref={a.shape} vmap={b.shape}")
        fails.append(f"rep{i} shape")
        continue
    for f in range(a.shape[0]):
        dR = np.abs(a[f] - b[f])
        scale = max(np.abs(a[f]).max(), 1e-12)
        print(f"{i:>3} {f:>5} {dR.max():>13.3e} "
              f"{np.sqrt((dR**2).mean()):>13.3e} {dR.max()/scale:>10.2e}")
    # first recorded frame is the equivalence criterion
    # frame 0 is the copied initial structure -- identical by construction and
    # therefore meaningless as a check. frame 1 = step 1 is the first frame
    # produced by the integrator.
    d0 = np.abs(a[1] - b[1]).max()
    tol = 1e-6 * max(np.abs(a[1]).max(), 1.0)   # one step: float32 eps ~1.2e-7, allow ~10x
    if not np.isfinite(d0) or d0 > tol:
        fails.append(f"rep{i} step1 max|dR|={d0:.3e} > tol={tol:.3e}")
    print("-" * 50)

if fails:
    print("\nFAIL:"); [print("  " + f) for f in fails]; sys.exit(1)
print("\nPASS: batched route reproduces MDRunner to float32 precision "
      "at step 1 for all replicas.")
