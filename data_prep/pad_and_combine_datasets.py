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


def _load_datasets(paths):
    """Load a list of CG NPZ files into a list of dicts."""
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
    return datasets


def _build_global_aa_mapping(datasets, pad_resname="PAD"):
    """Build a global AA→ID mapping across all datasets."""
    all_resnames = set()
    for d in datasets:
        all_resnames.update(set(d["resname"].tolist()))
    all_resnames.discard(pad_resname)
    id_to_aa = sorted(all_resnames)
    aa_to_id = {aa: i for i, aa in enumerate(id_to_aa)}
    return aa_to_id, id_to_aa


def _combine_datasets(datasets, paths, aa_to_id, id_to_aa, out_path,
                      pad_resid=-1, pad_resname="PAD", pad_Z=0, pad_species=-1):
    """
    Internal: pad and combine a list of pre-loaded datasets into one NPZ.

    Uses the supplied aa_to_id mapping (allows a globally consistent mapping
    when called from combine_and_pad_npz_bucketed).
    """
    Ns = [d["R"].shape[1] for d in datasets]
    Ts = [d["R"].shape[0] for d in datasets]
    N_max = int(max(Ns))
    T_total = int(sum(Ts))

    logger.info(f"Combining {len(paths)} datasets: T_total={T_total}, N_max={N_max}")
    for i, (p, N, T) in enumerate(zip(paths, Ns, Ts)):
        logger.debug(f"  [{i}] {Path(p).name}: {T} frames, {N} atoms")

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

        species_1d = np.array([aa_to_id[aa] for aa in resname], dtype=np.int32)

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
    logger.info(f"Saved: {out_path}")
    logger.info(f"  proteins={len(paths)}, T_total={T_total}, N_max={N_max}, n_species={len(id_to_aa)}")
    return str(out_path)


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

    datasets = _load_datasets(paths)
    aa_to_id, id_to_aa = _build_global_aa_mapping(datasets, pad_resname)
    logger.info(f"Global AA mapping: {aa_to_id}")

    return _combine_datasets(datasets, paths, aa_to_id, id_to_aa, out_path,
                             pad_resid, pad_resname, pad_Z, pad_species)


def pad_individual_npz(
    paths,
    out_dir,
    pad_resid=-1,
    pad_resname="PAD",
    pad_Z=0,
    pad_species=-1,
):
    """
    Pad each CG NPZ to the global N_max (across all inputs) and save as
    individual files in out_dir.  Uses a global AA->ID mapping for consistent
    species encoding across files.

    Used with --no_combine to keep per-protein files separate.

    Args:
        paths:   list of input CG NPZ paths
        out_dir: directory for padded output files

    Returns:
        list of output file path strings
    """
    if len(paths) == 0:
        raise ValueError("paths is empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = _load_datasets(paths)
    aa_to_id, id_to_aa = _build_global_aa_mapping(datasets, pad_resname)
    N_max = int(max(d["R"].shape[1] for d in datasets))

    logger.info(f"pad_individual_npz: {len(paths)} files, global N_max={N_max}, "
                f"n_species={len(id_to_aa)}")
    logger.info(f"Global AA mapping: {aa_to_id}")

    out_paths = []
    for p, d in zip(paths, datasets):
        T, N, _ = d["R"].shape
        out_path = out_dir / (Path(p).stem + "_padded.npz")

        R_pad = np.zeros((T, N_max, 3), dtype=np.float32)
        F_pad = np.zeros((T, N_max, 3), dtype=np.float32)
        Z_pad = np.full((T, N_max), pad_Z, dtype=np.int32)
        resid_pad = np.full((T, N_max), pad_resid, dtype=np.int32)
        resname_pad = np.empty((T, N_max), dtype=object)
        resname_pad[:] = pad_resname
        species_pad = np.full((T, N_max), pad_species, dtype=np.int32)
        mask_pad = np.zeros((T, N_max), dtype=np.float32)

        species_1d = np.array([aa_to_id[aa] for aa in d["resname"]], dtype=np.int32)

        R_pad[:, :N, :] = d["R"]
        F_pad[:, :N, :] = d["F"]
        Z_pad[:, :N] = d["Z"][None, :]
        resid_pad[:, :N] = d["resid"][None, :]
        resname_pad[:, :N] = d["resname"][None, :]
        species_pad[:, :N] = species_1d[None, :]
        mask_pad[:, :N] = 1.0

        np.savez(
            out_path,
            R=R_pad,
            F=F_pad,
            Z=Z_pad,
            resid=resid_pad,
            resname=resname_pad,
            species=species_pad,
            mask=mask_pad,
            n_atoms=np.full((T,), N, dtype=np.int32),
            aa_to_id=np.array([aa_to_id], dtype=object),
            id_to_aa=np.array([id_to_aa], dtype=object),
            N_max=np.array([N_max], dtype=np.int32),
        )
        logger.debug(f"  Saved: {out_path.name} (T={T}, N={N}→{N_max})")
        out_paths.append(str(out_path))

    logger.info(f"Saved {len(out_paths)} padded files to {out_dir}")
    return out_paths


def combine_and_pad_npz_bucketed(
    paths,
    out_dir,
    bucket_boundaries=None,
    n_buckets=3,
    pad_resid=-1,
    pad_resname="PAD",
    pad_Z=0,
    pad_species=-1,
):
    """
    Combine CG NPZs into per-bucket NPZ files, one file per length bucket.

    Proteins are grouped by N_real (number of atoms before padding) into
    buckets. Each bucket's NPZ is padded only to that bucket's N_max,
    reducing wasted compute when training on proteins of varied lengths.

    A GLOBAL aa_to_id mapping is built across ALL proteins so species IDs
    are consistent across all bucket files.

    Args:
        paths:             list of input per-protein CG NPZ paths
        out_dir:           output directory; files named bucket_N{nmax:04d}.npz
        bucket_boundaries: list of N_real boundaries (e.g. [100, 200] gives
                           3 buckets: ≤100, 101-200, >200).
                           If None, auto-compute n_buckets equal-count groups.
        n_buckets:         number of auto-computed buckets (ignored when
                           bucket_boundaries is supplied)

    Returns:
        dict: {N_max: out_path_str} for each non-empty bucket, sorted by N_max
    """
    if len(paths) == 0:
        raise ValueError("paths is empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets and get per-protein N_real
    datasets = _load_datasets(paths)
    n_reals = [d["R"].shape[1] for d in datasets]

    # Build GLOBAL aa_to_id across all proteins
    aa_to_id, id_to_aa = _build_global_aa_mapping(datasets, pad_resname)
    logger.info(f"Global AA mapping ({len(id_to_aa)} types): {aa_to_id}")

    # Determine bucket boundaries
    if bucket_boundaries is None:
        sorted_ns = sorted(set(n_reals))
        # Compute n_buckets quantile-based boundaries
        boundaries = []
        for i in range(1, n_buckets):
            idx = (i * len(sorted_ns)) // n_buckets
            boundaries.append(sorted_ns[min(idx, len(sorted_ns) - 1)])
        # Remove duplicates while preserving order
        seen = set()
        bucket_boundaries = [b for b in boundaries if not (b in seen or seen.add(b))]

    logger.info(f"Bucket boundaries: {bucket_boundaries} "
                f"({len(bucket_boundaries) + 1} buckets)")

    def assign_bucket(n):
        for i, b in enumerate(bucket_boundaries):
            if n <= b:
                return i
        return len(bucket_boundaries)

    # Group datasets by bucket
    buckets = {}
    for p, d, n in zip(paths, datasets, n_reals):
        bid = assign_bucket(n)
        if bid not in buckets:
            buckets[bid] = ([], [])
        buckets[bid][0].append(p)
        buckets[bid][1].append(d)

    result = {}
    for bid in sorted(buckets):
        bpaths, bdatasets = buckets[bid]
        bucket_n_max = int(max(d["R"].shape[1] for d in bdatasets))
        out_path = out_dir / f"bucket_N{bucket_n_max:04d}.npz"
        _combine_datasets(bdatasets, bpaths, aa_to_id, id_to_aa, str(out_path),
                          pad_resid, pad_resname, pad_Z, pad_species)
        result[bucket_n_max] = str(out_path)
        logger.info(f"  Bucket {bid}: {len(bpaths)} proteins, "
                    f"N_max={bucket_n_max} → {out_path.name}")

    logger.info(f"Bucketed output: {len(result)} bucket files in {out_dir}")
    return dict(sorted(result.items()))


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
