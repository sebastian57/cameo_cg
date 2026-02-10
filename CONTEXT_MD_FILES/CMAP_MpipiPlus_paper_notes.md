# CMAP potential in "Refined Bonded Terms in Coarse-Grained Models for Intrinsically Disordered Proteins Improve Backbone Conformations"

## Sources consulted
- ACS article page (abstract + supporting-information list). citeturn0search0
- PubMed abstract (PMID: 38950000). citeturn0search1

Note: The full ACS paper and the supporting information PDF are not accessible from this environment. The details below are constrained to the abstract and the publicly visible supporting-information list.

## What the paper explicitly says about CMAP development
- The model introduces residue-specific angular, refined dihedral, and correction map (CMAP) potentials. citeturn0search0turn0search1
- These bonded terms are derived from statistics of a customized coil database and integrated into Mpipi to create Mpipi+. citeturn0search0turn0search1
- The ACS supporting-information list mentions a (theta1, theta2) PMF during CMAP iterations, indicating a 2D correction map over two consecutive backbone angles (or closely related variables) and an iterative fitting process. citeturn0search0

## Known gaps (need full paper/SI)
The following specifics are not available without the full text / SI:
- Exact definition of CMAP variables (theta1/theta2) and indexing scheme.
- Grid resolution and interpolation method.
- Whether CMAP is residue-specific by central residue, residue pair, or another context.
- Details of the fitting workflow (e.g., smoothing, iteration count, convergence criteria).

## Inferred CMAP construction (assumptions)
- Given the (theta1, theta2) PMF mention, CMAP likely uses two consecutive backbone angles, e.g., angle i-1,i,i+1 and angle i,i+1,i+2. This is an inference that must be verified. citeturn0search0
- The CMAP likely represents a 2D PMF derived from coil-database statistics and may be iteratively refined. This is inferred from the SI list and should be verified. citeturn0search0

## Plan to implement a CMAP replacement in cameo_cg

### Goal
Add a CMAP-style correction map potential and optionally replace the current angle/dihedral prior terms with CMAP, while retaining bonds and repulsion unless disabled.

### Fit with current codebase
Relevant files:
- `models/prior_energy.py` (bond/angle/dihedral/repulsive energy terms)
- `models/topology.py` (bond/angle/dihedral index arrays)
- `data_prep/prior_fitting_script.py` (Boltzmann inversion for bonded terms)

### Proposed additions and changes
1. **Topology extension**
   - Add a CMAP index builder that yields the two consecutive angle triplets (i-1,i,i+1) and (i,i+1,i+2) for each valid central index.

2. **PriorEnergy: CMAP term**
   - Implement `compute_cmap_energy(R, mask, params)`.
   - Compute theta1/theta2 for each CMAP entry.
   - Lookup energy via bilinear interpolation on a 2D grid.
   - Add a `priors.weights.cmap` scaling factor.

3. **Config additions**
   - Add `model.priors.cmap` block with:
     - `theta_min`, `theta_max`, `n_bins`
     - `grid` (2D array) or `grid_path` (NPZ) for large maps
   - Add a toggle, e.g. `model.priors.use_cmap_only`, to disable angle/dihedral when CMAP is on.

4. **Fitting pipeline changes**
   - Add `data_prep/cmap_fitting_script.py` (new) that:
     - Computes theta1/theta2 for all valid entries from the coil database or training data.
     - Builds P(theta1, theta2), converts to PMF (U = -kT ln P).
     - Optionally iterates if the paper uses an iterative CMAP refinement.
   - Save grid to YAML or NPZ; load via config.

5. **Exporter updates**
   - Extend `export/exporter.py` to pass CMAP parameters and grid to the exported model.

### Development effort estimate
- **Small (1-2 days)**: CMAP term in `PriorEnergy` + config plumbing + grid loader.
- **Medium (3-5 days)**: CMAP fitting script with histogram/PMF generation.
- **Large (1-2 weeks)**: Matching the exact paper/SI procedure, residue specificity, and iteration details.

### Verification checklist
- Unit test: CMAP energy zero if grid is zero.
- Gradient check: finite difference vs autodiff for theta1/theta2.
- Sanity: CMAP-only runs produce backbone angle distributions closer to fitted PMF.

## Next steps recommended
- Obtain full paper + SI PDF to confirm CMAP variable definitions, grid resolution, and fitting protocol.
- Decide whether CMAP is residue-specific and define indexing accordingly.
- If residue-specific, define parameter storage and lookup in `PriorEnergy`.

