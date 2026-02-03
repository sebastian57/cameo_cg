#!/usr/bin/env python3
"""
Dataset Analysis Tool for CG Protein Force Fields

Analyzes NPZ datasets and provides:
- Basic statistics (shape, dtype, NaN/Inf checks)
- Force magnitude analysis
- Training loss prediction (what chemtrain will report)
- Comparison to literature benchmarks

Usage:
    python analyze_dataset.py --npz path/to/dataset.npz
"""

import numpy as np
import argparse


def check_array(name, arr):
    """Basic array statistics and sanity checks."""
    print(f"\n=== Checking {name} ===")
    print(f"shape: {arr.shape}")
    print("dtype:", arr.dtype)

    print("min:", np.min(arr))
    print("max:", np.max(arr))
    print("mean abs:", np.mean(np.abs(arr)))

    print("contains NaNs:", np.isnan(arr).any())
    print("contains infs:", np.isinf(arr).any())

    # print random sample
    idx = np.random.randint(0, arr.shape[0])
    print(f"sample[{idx}]:")
    print(arr[idx])


def diagnose_forces(F, mask=None):
    """
    Comprehensive force diagnosis for CG datasets.

    Computes statistics that help interpret training loss values
    and compare to literature benchmarks.

    Args:
        F: Force array, shape (n_frames, n_atoms, 3) in kcal/mol/Å
        mask: Optional validity mask, shape (n_frames, n_atoms)

    Returns:
        dict with force statistics
    """
    print("\n" + "="*60)
    print("FORCE DIAGNOSIS FOR CG TRAINING")
    print("="*60)

    n_frames, n_atoms, _ = F.shape

    # Handle mask
    if mask is None:
        mask = np.ones((n_frames, n_atoms), dtype=np.float32)

    # Ensure mask is 2D
    if mask.ndim == 1:
        # Single mask for all frames
        mask = np.tile(mask, (n_frames, 1))

    # Expand mask for 3D operations
    mask_3d = mask[..., None]  # (n_frames, n_atoms, 1)

    # =========================================================================
    # 1. Per-atom force magnitudes (3D vectors)
    # =========================================================================
    F_magnitude = np.sqrt(np.sum(F**2, axis=-1))  # (n_frames, n_atoms)
    F_magnitude_valid = F_magnitude * mask

    valid_count = np.sum(mask)
    mean_magnitude = np.sum(F_magnitude_valid) / valid_count
    max_magnitude = np.max(F_magnitude_valid)

    print(f"\n1. FORCE VECTOR MAGNITUDES (|F| = sqrt(Fx² + Fy² + Fz²))")
    print(f"   Mean |F|:  {mean_magnitude:.2f} kcal/mol/Å")
    print(f"   Max |F|:   {max_magnitude:.2f} kcal/mol/Å")
    print(f"   Valid atoms: {int(valid_count)} (across all frames)")

    # Percentiles
    F_mag_flat = F_magnitude_valid[mask > 0]
    print(f"   Percentiles:")
    print(f"     50th (median): {np.percentile(F_mag_flat, 50):.2f} kcal/mol/Å")
    print(f"     90th:          {np.percentile(F_mag_flat, 90):.2f} kcal/mol/Å")
    print(f"     99th:          {np.percentile(F_mag_flat, 99):.2f} kcal/mol/Å")

    # =========================================================================
    # 2. Per-component statistics (what chemtrain loss uses)
    # =========================================================================
    F_masked = F * mask_3d
    n_components = np.sum(mask) * 3  # Total valid force components

    mean_component = np.sum(np.abs(F_masked)) / n_components
    rms_component = np.sqrt(np.sum(F_masked**2) / n_components)

    print(f"\n2. PER-COMPONENT STATISTICS (Fx, Fy, Fz separately)")
    print(f"   Mean |Fx,y,z|:  {mean_component:.2f} kcal/mol/Å")
    print(f"   RMS component:  {rms_component:.2f} kcal/mol/Å")
    print(f"   (This is what chemtrain normalizes by)")

    # =========================================================================
    # 3. What chemtrain loss will look like
    # =========================================================================
    print(f"\n3. PREDICTED CHEMTRAIN LOSS VALUES")
    print(f"   (Assuming model predicts ZERO forces - worst case)")

    # MSE if model predicts zeros (worst case baseline)
    mse_baseline = np.sum(F_masked**2) / n_components
    rmse_baseline = np.sqrt(mse_baseline)

    print(f"   MSE (zero model):  {mse_baseline:.2f}")
    print(f"   RMSE (zero model): {rmse_baseline:.2f} kcal/mol/Å")

    # What different relative errors would look like
    print(f"\n   Expected loss for different model qualities:")
    print(f"   (relative error = RMSE / RMS_component)")
    print(f"   {'Rel.Err':>8} {'MSE':>10} {'RMSE':>10} {'Quality':>15}")
    print(f"   {'-'*8} {'-'*10} {'-'*10} {'-'*15}")

    quality_labels = {
        0.50: "Poor",
        0.30: "Fair",
        0.20: "Good",
        0.15: "Very Good",
        0.10: "Excellent",
        0.05: "Exceptional",
    }

    for rel_error in [0.50, 0.30, 0.20, 0.15, 0.10, 0.05]:
        expected_rmse = rel_error * rms_component
        expected_mse = expected_rmse**2
        quality = quality_labels.get(rel_error, "")
        print(f"   {rel_error*100:>6.0f}% {expected_mse:>10.2f} {expected_rmse:>10.2f} {quality:>15}")

    # =========================================================================
    # 4. Comparison to literature
    # =========================================================================
    print(f"\n4. LITERATURE COMPARISON")
    print(f"   ")
    print(f"   All-atom ML potentials (NequIP, Allegro on QM data):")
    print(f"     Typical RMSE: 0.5 - 2.0 kcal/mol/Å")
    print(f"     Typical force magnitudes: ~10-20 kcal/mol/Å")
    print(f"   ")
    print(f"   CG protein models (1-bead-per-residue):")
    print(f"     Expected RMSE: 5 - 15 kcal/mol/Å (10-20% relative error)")
    print(f"     Irreducible error due to coarse-graining")
    print(f"   ")
    print(f"   YOUR DATA:")
    print(f"     Force RMS:    {rms_component:.2f} kcal/mol/Å")
    print(f"     Target RMSE:  {0.15 * rms_component:.1f} - {0.20 * rms_component:.1f} kcal/mol/Å (15-20% error)")
    print(f"     Target MSE:   {(0.15 * rms_component)**2:.1f} - {(0.20 * rms_component)**2:.1f}")

    # =========================================================================
    # 5. Potential issues
    # =========================================================================
    print(f"\n5. POTENTIAL ISSUES")

    issues = []

    if mean_magnitude > 100:
        issues.append(f"   ⚠️  Mean force magnitude ({mean_magnitude:.1f}) is high.")
        issues.append(f"       This could indicate: high-energy conformations, units issue,")
        issues.append(f"       or strained structures. Check if units are kcal/mol/Å.")

    if max_magnitude > 500:
        issues.append(f"   ⚠️  Max force ({max_magnitude:.1f}) is very high.")
        issues.append(f"       Consider filtering outlier frames or checking for overlapping atoms.")

    if np.any(np.isnan(F)):
        issues.append(f"   ❌  NaN values detected in forces!")

    if np.any(np.isinf(F)):
        issues.append(f"   ❌  Inf values detected in forces!")

    # Check for force imbalance (should sum to ~0 for isolated system)
    F_per_frame = F * mask_3d
    total_force = np.sum(F_per_frame, axis=1)  # (n_frames, 3)
    max_imbalance = np.max(np.abs(total_force))
    mean_imbalance = np.mean(np.linalg.norm(total_force, axis=1))

    if mean_imbalance > 10.0:
        issues.append(f"   ⚠️  Force imbalance detected (mean={mean_imbalance:.2f}, max={max_imbalance:.2f}).")
        issues.append(f"       For isolated systems, forces should sum to ~zero.")
        issues.append(f"       This is normal for CG with aggforce projection.")

    if len(issues) == 0:
        print(f"   ✓  No obvious issues detected.")
    else:
        for issue in issues:
            print(issue)

    print("\n" + "="*60)

    return {
        "mean_magnitude": float(mean_magnitude),
        "max_magnitude": float(max_magnitude),
        "rms_component": float(rms_component),
        "mse_baseline": float(mse_baseline),
        "rmse_baseline": float(rmse_baseline),
        "n_frames": n_frames,
        "n_atoms": n_atoms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect NPZ dataset for CG protein force field training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_dataset.py --npz datasets/protein_cg.npz
    python analyze_dataset.py --npz datasets/combined.npz --force-only
        """
    )
    parser.add_argument("--npz", required=True, help="Path to dataset .npz file")
    parser.add_argument("--force-only", action="store_true",
                        help="Only run force diagnosis (skip basic checks)")
    args = parser.parse_args()

    print(f"Loading dataset: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)

    # Required fields
    R = data["R"]
    F = data["F"]

    # Optional fields
    box = data["box"] if "box" in data else None
    Z = data["Z"] if "Z" in data else None
    mask = data["mask"] if "mask" in data else None

    if not args.force_only:
        print("\n" + "="*40)
        print("DATASET BASIC SUMMARY")
        print("="*40)
        print(f"R shape:   {R.shape}")
        print(f"F shape:   {F.shape}")
        if box is not None:
            print(f"box shape: {box.shape}")
        if Z is not None:
            print(f"Z shape:   {Z.shape}")
        if mask is not None:
            print(f"mask shape: {mask.shape}")

        # Check arrays
        check_array("Positions R", R)
        check_array("Forces F", F)
        if box is not None:
            check_array("Box", box)
        if Z is not None:
            print("\nAtom types (Z):", np.unique(Z))

        # Distance sanity check
        print("\n=== Distance sanity check for first frame ===")
        R0 = R[0]

        # Handle mask for distance calculation
        if mask is not None:
            if mask.ndim == 2:
                mask0 = mask[0]
            else:
                mask0 = mask
            valid_idx = np.where(mask0 > 0)[0]
            R0_valid = R0[valid_idx]
        else:
            R0_valid = R0

        dists = np.linalg.norm(R0_valid[:, None, :] - R0_valid[None, :, :], axis=-1)
        # Ignore self-distances
        np.fill_diagonal(dists, np.inf)

        print(f"Number of valid atoms: {len(R0_valid)}")
        print(f"Min non-self distance: {np.min(dists):.2f} Å")
        print(f"Max distance: {np.max(dists[dists < np.inf]):.2f} Å")
        print(f"Mean distance: {np.mean(dists[dists < np.inf]):.2f} Å")

        if np.min(dists) < 2.0:
            print("⚠️  WARNING: Very close atoms detected (<2 Å)")
            print("   This may cause issues with repulsive priors.")

        if np.max(dists[dists < np.inf]) > 100:
            print("⚠️  WARNING: Very large distances detected (>100 Å)")

    # =========================================================================
    # Force diagnosis (the main event)
    # =========================================================================
    stats = diagnose_forces(F, mask)

    # Summary
    print("\n" + "="*40)
    print("QUICK REFERENCE")
    print("="*40)
    print(f"Your force RMS:     {stats['rms_component']:.2f} kcal/mol/Å")
    print(f"Baseline MSE:       {stats['mse_baseline']:.2f} (if model predicts zero)")
    print(f"Target MSE (15%):   {(0.15 * stats['rms_component'])**2:.2f}")
    print(f"Target MSE (20%):   {(0.20 * stats['rms_component'])**2:.2f}")
    print(f"\nIf chemtrain reports MSE ~ {(0.15 * stats['rms_component'])**2:.0f}-{(0.20 * stats['rms_component'])**2:.0f}, you're doing well!")
    print()


if __name__ == "__main__":
    main()
