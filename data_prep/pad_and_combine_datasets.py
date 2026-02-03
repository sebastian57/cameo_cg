import argparse
import logging
import sys
import numpy as np
from pathlib import Path

# Logger — matches clean_code_base [Name] format from utils/logging.py
logger = logging.getLogger("PadCombine")
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _normalize_resname_array(resname):
    """
    Ensure resname is a 1D numpy array of python strings.
    Handles bytes/object arrays.
    """
    resname = np.asarray(resname)
    out = []
    for aa in resname:
        if isinstance(aa, (bytes, np.bytes_)):
            out.append(aa.decode("utf-8"))
        else:
            out.append(str(aa))
    return np.array(out, dtype=object)

def combine_and_pad_npz(
    paths,
    out_path,
    pad_resid=-1,
    pad_resname="PAD",
    pad_Z=0,
    pad_species=-1,
):
    """
    Combine multiple NPZ datasets with different N into one padded NPZ.
    Adds `species` computed from a GLOBAL AA->ID mapping across all datasets.

    Output keys:
      R, F, Z, resid, resname, species, mask, n_atoms, protein_id, paths,
      aa_to_id, id_to_aa, N_max
    """
    if len(paths) == 0:
        raise ValueError("paths is empty")

    # Load all datasets
    datasets = []
    for p in paths:
        data = np.load(p, allow_pickle=True)
        d = {
            "R": data["R"].astype(np.float32),
            "F": data["F"].astype(np.float32),
            "Z": data["Z"].astype(np.int32),
            "resid": data["resid"].astype(np.int32),
            "resname": _normalize_resname_array(data["resname"]),
        }
        datasets.append(d)

    # GLOBAL AA mapping across all proteins
    all_resnames = set()
    for d in datasets:
        all_resnames.update(set(d["resname"].tolist()))
    # remove PAD if present in inputs
    all_resnames.discard(pad_resname)

    id_to_aa = sorted(all_resnames)
    aa_to_id = {aa: i for i, aa in enumerate(id_to_aa)}

    logger.info(f"Global AA mapping: {aa_to_id}")

    # Determine N_max and total frames
    Ns = [d["R"].shape[1] for d in datasets]
    Ts = [d["R"].shape[0] for d in datasets]
    N_max = int(max(Ns))
    T_total = int(sum(Ts))

    logger.info(f"Combining {len(paths)} datasets: T_total={T_total}, N_max={N_max}")
    for i, (p, N, T) in enumerate(zip(paths, Ns, Ts)):
        logger.debug(f"  [{i}] {Path(p).name}: {T} frames, {N} atoms")

    # Allocate merged arrays
    R_all = np.zeros((T_total, N_max, 3), dtype=np.float32)
    F_all = np.zeros((T_total, N_max, 3), dtype=np.float32)

    resid_all = np.full((T_total, N_max), pad_resid, dtype=np.int32)
    Z_all = np.full((T_total, N_max), pad_Z, dtype=np.int32)

    resname_all = np.empty((T_total, N_max), dtype=object)
    resname_all[:] = pad_resname

    species_all = np.full((T_total, N_max), pad_species, dtype=np.int32)

    mask_all = np.zeros((T_total, N_max), dtype=np.float32)
    n_atoms_all = np.zeros((T_total,), dtype=np.int32)
    protein_id_all = np.zeros((T_total,), dtype=np.int32)

    cursor = 0
    for pid, (p, d) in enumerate(zip(paths, datasets)):
        R = d["R"]; F = d["F"]
        Z = d["Z"]; resid = d["resid"]; resname = d["resname"]

        T, N, _ = R.shape
        sl = slice(cursor, cursor + T)

        # Compute species for this protein using GLOBAL map
        species_1d = np.array([aa_to_id[aa] for aa in resname], dtype=np.int32)  # (N,)

        # Insert
        R_all[sl, :N, :] = R
        F_all[sl, :N, :] = F
        Z_all[sl, :N] = Z[None, :]
        resid_all[sl, :N] = resid[None, :]
        resname_all[sl, :N] = resname[None, :]
        species_all[sl, :N] = species_1d[None, :]

        mask_all[sl, :N] = 1.0
        n_atoms_all[sl] = N
        protein_id_all[sl] = pid

        cursor += T

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        R=R_all,
        F=F_all,
        Z=Z_all,
        resid=resid_all,
        resname=resname_all,
        species=species_all,
        mask=mask_all,
        n_atoms=n_atoms_all,
        protein_id=protein_id_all,
        paths=np.array([str(p) for p in paths], dtype=object),
        aa_to_id=np.array([aa_to_id], dtype=object),
        id_to_aa=np.array([id_to_aa], dtype=object),
        N_max=np.array([N_max], dtype=np.int32),
    )

    logger.info(f"Saved combined dataset: {out_path}")
    logger.info(f"  proteins={len(paths)}, T_total={T_total}, N_max={N_max}, n_species={len(id_to_aa)}")
    return str(out_path)



def main():
    parser = argparse.ArgumentParser(description="Combine multiple CG NPZ datasets with padding.")
    parser.add_argument("--paths", action="append", required=True,
                        help="Input NPZ file path. Repeat for each dataset "
                             "(e.g. --paths a.npz --paths b.npz).")
    parser.add_argument("--out", required=True, help="Output combined NPZ file path.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    combine_and_pad_npz(args.paths, args.out)

if __name__ == "__main__":
    main()
