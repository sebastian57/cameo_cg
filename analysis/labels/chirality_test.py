#!/usr/bin/env python3
"""Is the learned potential CHIRAL, or has it collapsed to a distance-only surface?

Allegro's inputs are interatomic DISTANCES, which are reflection-invariant. Chirality
can only live in the parity-ODD irreps (0o, 1o). If the training signal never drives
them -- e.g. the only chirality-distinguishing region was starved of loss weight --
they stay near zero and the model degenerates to a distance-only potential, which is
EXACTLY mirror-symmetric. That produces a Ramachandran map with a spurious mirror
basin at phi>0, which is what job 1505004 showed (phi>0 -> 51.6% vs 3.18% reference).

Test: E(x) vs E(Mx) for an improper transform M (point inversion, det = -1).
  chiral model   -> |dE| comparable to the spread of E across configurations
  collapsed model-> |dE| ~ 0 regardless of configuration
Reported relative to kT (0.5921 kcal/mol at 298 K) and to sd(E).
"""
from __future__ import annotations
import argparse, pickle, sys, tempfile
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
KT = 0.5921868690749673


def load_model(training_config_path, params_path, dataset_path):
    from utils.jax_setup import apply_jax_compat_shims
    apply_jax_compat_shims()
    import jax.numpy as jnp
    from config.manager import ConfigManager
    from data.preprocessor import CoordinatePreprocessor
    from models.combined_model import CombinedModel

    cfg_dict = yaml.safe_load(open(training_config_path))
    cfg_dict.setdefault("model", {})["neighbor_disable_cell_list"] = True
    tf = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg_dict, tf); tf.close()
    cfg = ConfigManager(tf.name)
    with np.load(dataset_path, allow_pickle=False) as d:
        R = np.asarray(d["R"], np.float32); species = np.asarray(d["species"], np.int32)
        mask = np.asarray(d["mask"], np.float32) if "mask" in d else np.ones(R.shape[:2], np.float32)
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
    p = pickle.load(open(params_path, "rb"))
    if isinstance(p, dict):
        p = p.get("params") if isinstance(p.get("params"), dict) else p.get("best_params", p)
    return model, p, mask0, species0, pre, box, shift, R, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="LABEL=training_config.yaml,params.pkl (repeatable)")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--n-frames", type=int, default=2000)
    a = ap.parse_args()
    import jax, jax.numpy as jnp

    print(f"Chirality test: E(x) vs E(Mx), M = point inversion (det=-1). kT={KT:.4f} kcal/mol\n")
    print(f"{'model':16s} {'sd(E)':>9} {'mean|dE|':>10} {'median|dE|':>11} "
          f"{'|dE|/kT':>9} {'|dE|/sd(E)':>11}  verdict")
    for spec in a.model:
        label, paths = spec.split("=", 1)
        cfgp, parp = paths.split(",", 1)
        model, params, mask0, species0, pre, box, shift, R, mask = load_model(cfgp, parp, a.reference)
        efn = model.energy_fn_template(params)
        nbrs = model.ml_model.nbrs_init

        idx = np.linspace(0, len(R) - 1, min(a.n_frames, len(R))).astype(int)
        Rc = pre.center_and_park(R[idx], mask[idx], box, shift)
        # point inversion about each frame's own centroid, then re-park identically
        cen = Rc.mean(axis=1, keepdims=True)
        Rm = pre.center_and_park((-(Rc - cen) + cen).astype(np.float32), mask[idx], box, shift)

        def energies(X):
            out = []
            for k in range(len(X)):
                x = jnp.asarray(X[k])
                nb = model.ml_model.nneigh_fn.update(x, nbrs, mask=mask0)
                out.append(float(efn(x, neighbor=nb, mask=mask0, species=species0)))
            return np.asarray(out)

        # CONTROLS. A test that returns "achiral" for every model is indistinguishable
        # from a broken one, so establish the numerical floor and the sensitivity
        # scale before interpreting dE.
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1.0                       # ensure a PROPER rotation (det=+1)
        cen_r = Rc.mean(axis=1, keepdims=True)
        Rrot = pre.center_and_park(((Rc - cen_r) @ Q.T + cen_r).astype(np.float32),
                                   mask[idx], box, shift)

        E, Em = energies(Rc), energies(Rm)
        Er = energies(Rrot)
        floor = float(np.mean(np.abs(Er - E)))          # rotation: MUST be ~0
        shuffle = float(np.mean(np.abs(E - E[rng.permutation(len(E))])))  # different frames
        print(f"  [control] rotation |dE| = {floor:.4f} (numerical floor)   "
              f"different-frame |dE| = {shuffle:.4f} (sensitivity scale)")
        dE = Em - E
        sdE = float(np.std(E))
        m_abs, med_abs = float(np.mean(np.abs(dE))), float(np.median(np.abs(dE)))
        ratio_kt, ratio_sd = m_abs / KT, m_abs / max(sdE, 1e-9)
        # only call it achiral if the mirror difference is at the ROTATION floor
        verdict = ("TEST INSENSITIVE" if shuffle < 10 * max(floor, 1e-9) else
                   "ACHIRAL (mirror == rotation floor)" if m_abs < 3 * max(floor, 1e-9) else
                   "weakly chiral" if ratio_sd < 0.25 else "CHIRAL")
        print(f"{label:16s} {sdE:9.3f} {m_abs:10.3f} {med_abs:11.3f} "
              f"{ratio_kt:9.2f} {ratio_sd:11.3f}  {verdict}")
    print("\n|dE|/sd(E) is the discriminator: a chiral potential separates a structure from its")
    print("mirror by an amount comparable to how much E varies across configurations.")


if __name__ == "__main__":
    main()
