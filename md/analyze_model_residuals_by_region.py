"""Where is each trained model wrong, and which model makes the best NEGATIVE bias for alphaL?

Two questions, one pass over the reference frames.

1. FORCE RESIDUAL BY REGION. Per-frame |F_model - F_ref| binned by Ramachandran basin. This
   is the residual analysis the acquisition loop currently lacks: every bias field so far
   (tica_regional, the committor attractor, R_TP) is built from AA reference statistics
   alone and is completely blind to where the model actually fails.

2. NEGATIVE-BIAS RANKING. `mlcg_teacher` samples U_AA - alpha*U_ML, so a region where the
   model's energy is spuriously HIGH gets its sampling energy LOWERED and is driven INTO.
   Every force-matched bb6 model ejects the molecule from alphaL in 0.2-2.0 ps against an AA
   median residence of 85 ps, i.e. U_ML slopes downhill OUT of alphaL, i.e. U_ML is too high
   there. That defect is exactly what makes these models useful as alphaL-seeking biases.

   Ranking statistic: <U_ML>_alphaL - <U_ML>_beta, evaluated on the SAME reference frames so
   the geometry is identical across models and only the learned surface differs. Larger =
   stronger alphaL-directed pull. This is a COMPARATIVE ranking, not an absolute free
   energy: U_ML is a potential energy and the reference's basin weights are free energies,
   so the two are not directly comparable and no attempt is made to equate them.

The bias this informs acts on BEAD POSITIONS only, like every other term in the registry.
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")


def _load_model(training_config_path: str, params_path: str, dataset_path: str, n_beads: int):
    """Same construction path as sampling/biases/teacher.py -- deliberately, so the energy
    surface measured here is byte-for-byte the one a teacher bias would apply."""
    import sys
    import tempfile

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.jax_setup import apply_jax_compat_shims
    apply_jax_compat_shims()
    import jax.numpy as jnp
    from config.manager import ConfigManager
    from data.preprocessor import CoordinatePreprocessor
    from models.combined_model import CombinedModel

    # Independent-frame batches overflow a cell list built from one reference frame and
    # silently drop edges -- BUGS/2026-07-31_neighbor-list-overflow-independent-frames_open.md
    cfg_dict = yaml.safe_load(open(training_config_path))
    cfg_dict.setdefault("model", {})["neighbor_disable_cell_list"] = True
    tf = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg_dict, tf); tf.close()
    cfg = ConfigManager(tf.name)

    with np.load(dataset_path, allow_pickle=False) as d:
        R = np.asarray(d["R"], np.float32)
        species = np.asarray(d["species"], np.int32)
        mask = (np.asarray(d["mask"], np.float32) if "mask" in d
                else np.ones(R.shape[:2], np.float32))
    pre = CoordinatePreprocessor(cutoff=cfg.get_cutoff(),
                                buffer_multiplier=cfg.get_buffer_multiplier(),
                                park_multiplier=cfg.get_park_multiplier())
    box, shift = pre.compute_box_extent(R, mask)
    mask0, species0 = jnp.asarray(mask[0]), jnp.asarray(species[0])
    R0 = pre.center_and_park(R[:1], mask[:1], box, shift)[0]
    n_species = max(int(species.max()) + 1,
                    int(cfg.get("model", "allegro", "num_types", default=0) or 0))
    model = CombinedModel(config=cfg, R0=jnp.asarray(R0), box=box, species=species0,
                          N_max=int(R.shape[1]), prior_only=cfg.prior_only_enabled(),
                          n_species_override=n_species)
    params = pickle.load(open(params_path, "rb"))
    if isinstance(params, dict):
        if isinstance(params.get("params"), dict):
            params = params["params"]
        elif isinstance(params.get("best_params"), dict):
            params = params["best_params"]
    return model, params, mask0, species0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    help="LABEL=training_config.yaml,params.pkl  (repeatable)")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--n-frames", type=int, default=20000,
                    help="evenly strided reference frames to evaluate")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from sampling.mapping import dihedral_deg, get_mapping, wrap_deg

    m = get_mapping(a.mapping)
    d = np.load(a.reference)
    R_all, F_all = d["R"], d["F"]
    idx = np.linspace(0, len(R_all) - 1, a.n_frames).astype(int)
    R, Fref = R_all[idx].astype(np.float32), F_all[idx].astype(np.float64)

    cv = lambda n: wrap_deg(dihedral_deg(R.astype(np.float64), m.cvs[n].bead_indices)
                            + m.cvs[n].shift_deg)
    phi, psi = cv("phi"), cv("psi")
    region = np.full(len(R), "other", dtype=object)
    region[(phi > -180) & (phi < -20) & ((psi > 90) | (psi < -150))] = "beta"
    region[(phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)] = "alphaR"
    region[(phi > 20) & (phi < 100) & (psi > -20) & (psi < 100)] = "alphaL"
    region[(phi > -15) & (phi < 15)] = "alphaL_corridor"   # the dividing surface
    names = ["beta", "alphaR", "alphaL", "alphaL_corridor", "other"]

    rows = {}
    for spec in a.model:
        label, _, paths = str(spec).partition("=")
        cfg_p, par_p = paths.split(",")
        model, params, mask0, species0 = _load_model(cfg_p, par_p, a.reference, m.n_beads)

        def energy(r):
            return model.compute_energy(params, r, mask0, species0)
        e_and_g = jax.jit(jax.value_and_grad(energy))

        U = np.zeros(len(R)); Fm = np.zeros_like(Fref)
        for s in range(0, len(R), a.batch):
            for k in range(s, min(s + a.batch, len(R))):
                u, g = e_and_g(jnp.asarray(R[k]))
                U[k] = float(u); Fm[k] = -np.asarray(g, dtype=np.float64)
        dF = Fm - Fref                                            # (n_frames, n_beads, 3)
        res = np.linalg.norm(dF, axis=-1).mean(axis=-1)            # per-frame RMS-ish |dF|
        rows[label] = dict(U=U, res=res, dF=dF)
        print(f"\n=== {label}")
        print(f"{'region':18s} {'n':>6s} {'<|dF|> RMS':>11s} {'|<dF>| BIAS':>12s} {'bias/RMS':>9s} "
              f"{'<U_ML>':>10s}")
        for nm in names:
            sel = region == nm
            if sel.sum() == 0:
                continue
            # BIAS is the vector mean of the force error over the region. This -- not the
            # RMS -- is what survives the path integral that sets dF between basins:
            # random error cancels on integration, systematic error accumulates. Measured
            # 2026-08-08: RMS |dF| is FLAT across regions (11.5-12.3) for every model,
            # including ones whose basin populations are wrong by 35x, so RMS cannot be an
            # acceptance criterion for "the forces are right in this basin".
            bias = np.linalg.norm(dF[sel].mean(axis=0))            # ||<dF>||, all beads
            r = res[sel].mean()
            print(f"  {nm:16s} {sel.sum():6d} {r:11.2f} {bias:12.3f} {bias/r:9.4f} "
                  f"{U[sel].mean():10.2f}")
            rows[label].setdefault("bias", {})[nm] = float(bias)
        dU = U[region == "alphaL"].mean() - U[region == "beta"].mean()
        print(f"  --> NEGATIVE-BIAS STRENGTH  <U>_alphaL - <U>_beta = {dU:+.2f} kcal/mol "
              f"(larger = pulls harder into alphaL)")
        rows[label]["dU"] = dU

        # ---- BINDING GATE ------------------------------------------------------------
        # Allegro's energy goes to 0 as the beads separate (per-atom contributions vanish
        # with no neighbours). So <U_ML> on BOUND reference frames IS the binding energy
        # relative to dissociation, and its SIGN decides whether the molecule is bound at
        # all. <U_ML> > 0 means flying apart is thermodynamically DOWNHILL and MD will
        # dissociate no matter how small the timestep.
        #
        # Measured 2026-08-15 -- <U_ML> tracks the MD's step-0 PE almost exactly:
        #     ref50k_mf  11.37 -> 16.84     dhh300k 26.58 -> 28.81
        #     mfonly33k  33.64 -> 34.15     all three dissociated 8/8 at BOTH 2 fs and 1 fs
        # Every stable model in the project's history has step-0 PE in -773..-17.8; the only
        # historical positive-PE model (alphaLboost_msam500, +22.7) is the one that loses
        # replicas to dissociation. Force accuracy does NOT protect against this: dhh300k
        # had a perfectly normal alphaL force RMS of 12.76 and still dissociated, because
        # binding is set by the integral of the force out to separation -- a region with no
        # training data, which force matching never constrains.
        u_bound = float(U.mean())
        rows[label]["U_bound"] = u_bound
        if u_bound > 0:
            print(f"  !! BINDING GATE FAILED: <U_ML> = {u_bound:+.2f} kcal/mol > 0 on bound "
                  f"reference frames.\n"
                  f"     The dissociated state (U=0) is LOWER. MD will fall apart; a smaller "
                  f"dt will not help.\n"
                  f"     DO NOT SPEND GPU TIME ON MD FOR THIS MODEL.")
        else:
            print(f"  binding gate OK: <U_ML> = {u_bound:+.2f} kcal/mol "
                  f"(bound state is {-u_bound:.1f} below dissociation)")

    print("\n" + "=" * 96)
    print("SUMMARY -- ranked by alphaL force BIAS (the criterion for 'forces right in this "
          "basin',\nindependent of whether the ensemble weight is right)\n")
    print(f"{'model':26s} {'bias aL':>9s} {'bias b':>8s} {'bias aR':>8s} {'bias tr':>8s} "
          f"{'RMS aL':>8s} {'dU(aL-b)':>9s} {'<U_ML>':>9s} {'BOUND?':>8s}")
    for k, v in sorted(rows.items(), key=lambda kv: kv[1].get("bias", {}).get("alphaL", 9e9)):
        b = v.get("bias", {})
        ub = v.get("U_bound", float("nan"))
        print(f"{k:26s} {b.get('alphaL',float('nan')):9.3f} {b.get('beta',float('nan')):8.3f} "
              f"{b.get('alphaR',float('nan')):8.3f} {b.get('alphaL_corridor',float('nan')):8.3f} "
              f"{v['res'][region=='alphaL'].mean():8.2f} {v['dU']:+9.2f} {ub:+9.2f} "
              f"{'no -- WILL DISSOCIATE' if ub > 0 else 'yes':>8s}")

    failed = [k for k, v in rows.items() if v.get("U_bound", -1) > 0]
    if failed:
        print(f"\n!! BINDING GATE FAILED for: {', '.join(failed)}")
        print("   <U_ML> > 0 means the dissociated state is lower in energy than the bound "
              "one.\n   These models cannot run MD at any timestep. Check this BEFORE "
              "submitting MD.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, phi=phi, psi=psi, region=region.astype(str),
                        **{f"{k}__U": v["U"] for k, v in rows.items()},
                        **{f"{k}__res": v["res"] for k, v in rows.items()},
                        **{f"{k}__bias_{r}": v.get("bias", {}).get(r, np.nan)
                           for k, v in rows.items() for r in names})
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
