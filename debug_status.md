# Export / Connector Debug Status

Last updated: 2026-05-01

## Goal

Make the MLIR runtime outputs match the correct Python/JAX reference path used by
`cameo_cg/scripts/run_analysis.sh`.

## Correct Reference Path

- `run_analysis.sh` calls `analysis_tests/analyze_suite.py`
- basic force evaluation comes from `analyze_suite._collect_force_eval_data(...)`
- that path:
  - loads the dataset
  - applies `CoordinatePreprocessor.compute_box_extent(...)`
  - applies `center_and_park(...)`
  - rebuilds `CombinedModel` from runtime config and params
  - evaluates forces in Python via autodiff

Conclusion:

- the correct reference is the Python/JAX model path
- raw single-frame `export_check` runs are useful for isolating connector bugs
- but they are not yet identical to full suite preprocessing unless explicitly reproduced

## Tested So Far

### 1. Original ML-priors exports

Artifacts:

- `tmp_compare_mlir_variants_ml_priors_symbolic_only/summary.json`
- `tmp_compare_mlir_variants_ml_priors_frame0_full/...`

Result:

- fixed and symbolic MLIR were both badly wrong vs Python
- fixed and symbolic were also badly wrong relative to each other

### 2. Re-exported ML-priors exports

Artifacts:

- `1pro_4zoh_reexport_debug_frame0_*`
- `reexport_compare_frame0/summary.json`

Result:

- fixed and symbolic MLIR became close to each other
- both remained wrong vs Python

Interpretation:

- re-export flow became internally consistent
- shared runtime/export convention bug still remained

### 3. ML-only exports

Artifacts:

- `ml_only_compare_frame0/summary.json`

Result:

- Python energy: `-219.73301696777344`
- symbolic MLIR energy: `-29.58004761`
- fixed MLIR energy: `-2.128853321`

Interpretation:

- priors are not the root cause

### 4. Neighbor-list format check

Artifacts:

- `debug_mlonly_compare.py`
- `tmp_debug_mlonly/debug_summary.json`
- `ml_only_compare_frame0_helper_edges_v2/summary.json`

Result:

- Python with doubled half-list / full sparse neighbor list matches the original Python result
- Python with half-list only does not
- JAX export in Python remains correct

Interpretation:

- the compiled MLIR itself is not the problem
- connector/runtime invocation is the problem

### 5. Connector fixes already applied

Modified connector files:

- `.../connector/domain.cpp`
- `.../connector/graph_builder.h`
- `.../connector/graph_builder.cpp`

Fixes:

- removed connector-side species decrement before MLIR call
- initialized sparse `n_valid_edges` from zero instead of one
- populated the sparse valid-edge predicate buffer
- ensured valid-edge capacity is large enough for the actual current edge count

### 6. Patched connector rerun on the same ML-only symbolic frame

Artifacts:

- baseline result: `ml_only_compare_frame0/symbolic_mlir/symbolic_mlir.json`
- patched direct rerun: `/tmp/patched_symbolic_ml_only.json`

Result:

- old symbolic energy: `-29.58004761`
- patched symbolic energy: `-98.27185822`
- Python reference energy: `-219.73301696777344`

Interpretation:

- connector bug was real
- patch moved the result significantly toward the correct answer
- there is still at least one additional runtime mismatch

## Open Questions

1. Are the exact species / sender / receiver / validity-mask buffers passed into the executable correct?
2. Is the sparse `neighbors_literal` predicate semantics correct (`true` for active edges, `false` for padding)?
3. Does recompilation affect the first execution result?
4. Is there any remaining mismatch in edge ordering between helper-side construction and what the exported program expects?
5. Is there any remaining mismatch in local/ghost/newton semantics in the connector path?
6. After symbolic is correct, does the fixed helper still have parallel bugs?

## Next Intended Tests

1. Add helper-side audit output for the exact frame/types/neighbor arrays passed into the connector.
2. Add a repeat-run mode on the same input to compare first run vs second run after recompilation.
3. Save each test run under `export_check/` with:
   - intent
   - command/run setup
   - artifacts
   - result
   - interpretation

## Run Log

### 2026-05-01: Seed log

Intent:

- freeze the current debugging state before running new connector-focused tests

Outcome:

- this file created

Open after run:

- exact runtime buffer verification
- repeat-run / recompilation effect check

### 2026-05-01: ML-only symbolic repeat-run audit

Intent:

- test whether recompilation changes the symbolic connector result on the same exact frame
- save the exact helper-side types and sparse neighbor inputs passed into the connector

Command / setup:

- helper: `export_check/mlir_single_point_audit`
- model: `cameo_cg/.../ml_only/exports/1pro_4zoh_comparison_ml_only_symbolic.mlir`
- frame: `ml_only_compare_frame0/symbolic_mlir/symbolic_mlir.frame.txt`
- repeat count: `2`
- output dir: `connector_audit_ml_only_repeat1/`

Artifacts:

- `connector_audit_ml_only_repeat1/result.json`
- `connector_audit_ml_only_repeat1/repeat_summary.json`
- `connector_audit_ml_only_repeat1/helper_inputs.json`

Result:

- run 0 energy: `-98.27185822`, `recompiled=true`
- run 1 energy: `-98.27185822`, `recompiled=false`
- force delta run1 vs run0: `rmse=0`, `mae=0`, `max_abs=0`
- helper-side types are confirmed 1-indexed in the saved input file
- helper-side sparse half-list input is saved explicitly as `ilist`, `numneigh`, and `neighbor_storage`

Interpretation:

- recompilation is not the remaining source of the mismatch
- the patched connector reaches the same answer before and after recompilation
- the remaining bug is downstream of helper-side repeat behavior, or in still-unverified runtime buffer semantics

Open after run:

- exact executable-facing buffer semantics, especially sparse valid-edge predicate meaning
- local/ghost/newton behavior
- edge ordering / buffer interpretation inside the connector runtime

### 2026-05-01: ML-only symbolic `newton=0` audit

Intent:

- test whether connector `newton` handling changes the single-domain symbolic result
- keep the same saved frame and repeat-run setup as the previous audit

Command / setup:

- helper: `export_check/mlir_single_point_audit`
- same ML-only symbolic model and saved frame as previous run
- `--newton 0`
- repeat count: `2`
- output dir: `connector_audit_ml_only_newton0/`

Artifacts:

- `connector_audit_ml_only_newton0/result.json`
- `connector_audit_ml_only_newton0/repeat_summary.json`
- `connector_audit_ml_only_newton0/force_compare_vs_python.json`

Result:

- `newton=1` energy: `-98.27185822`
- `newton=0` energy: `-197.3881073`
- Python reference energy: `-219.73301696777344`
- force comparison vs Python:
  - old symbolic baseline RMSE: `5.9953`
  - patched `newton=1` RMSE: `4.6689`
  - patched `newton=0` RMSE: `0.8583`
- `newton=0` repeat behavior is again stable:
  - run 0 `recompiled=true`
  - run 1 `recompiled=false`
  - identical energies and forces across repeats

Interpretation:

- connector `newton` semantics are a major remaining source of the mismatch
- in this single-domain helper path, `newton=0` is much closer to the Python reference than `newton=1`
- `newton=0` also makes forces much closer to Python, not just energy
- recompilation is still not the source of the discrepancy

Open after run:

- why `newton=1` changes single-domain results so strongly
- whether the exported model expects one newton convention and the connector supplies another
- whether force/energy accumulation or edge direction handling differs between the two modes

### 2026-05-01: ML-only symbolic plugin-semantics audit

Intent:

- make the standalone helper follow the same documented neighbor-list rule as the LAMMPS plugin:
  - if `newton=true`, request a full list even when the model advertises `half_list=true`

Command / setup:

- helper: `export_check/mlir_single_point_audit`
- same ML-only symbolic model and saved frame
- `--newton 1`
- `--plugin-neighbor-semantics 1`
- repeat count: `2`
- output dir: `connector_audit_ml_only_plugin_semantics/`

Artifacts:

- `connector_audit_ml_only_plugin_semantics/result.json`
- `connector_audit_ml_only_plugin_semantics/repeat_summary.json`
- `connector_audit_ml_only_plugin_semantics/helper_inputs.json`
- `connector_audit_ml_only_plugin_semantics/compare_vs_python.json`

Result:

- energy: `-197.3881073`
- Python energy: `-219.73301696777344`
- energy abs diff vs Python: `22.3449`
- force RMSE vs Python: `0.8583`
- helper-side total directed edges: `720`
- repeat behavior is stable across recompilation

Interpretation:

- the standalone symbolic helper was previously mis-modeling connector/LAMMPS semantics for `newton=true`
- the major remaining connector mismatch was not the `newton` flag alone, but the combination:
  - `newton=true`
  - plus needing a full directed list
- once plugin neighbor semantics are reproduced, symbolic MLIR becomes much closer to Python
- the remaining gap is now relatively small and is a plausible place to consider precision / export-backend differences such as naive TP vs `uniform_1d`

Open after run:

- explain the remaining `~22.3` energy gap and `~0.86` force RMSE
- determine whether that residual is from:
  - remaining connector semantics
  - numerical precision
  - symbolic naive TP vs Python `uniform_1d` / non-naive execution differences

### 2026-05-01: Python standard vs Python naive vs plugin-style symbolic

Intent:

- determine whether the remaining residual is caused by the symbolic export / naive TP approximation itself
- use the same full directed graph as the plugin-semantics helper run

Command / setup:

- script: `export_check/compare_python_naive_symbolic.py`
- same ML-only frame 0
- same full directed graph reconstructed from `connector_audit_ml_only_plugin_semantics/helper_inputs.json`
- compare:
  - standard Python model
  - Python export-compatible naive apply function
  - plugin-style symbolic runtime result

Artifacts:

- `connector_audit_ml_only_naive_python_graphmatch/summary.json`
- `connector_audit_ml_only_naive_python_graphmatch/forces.npz`

Result:

- Python standard full-graph energy: `-219.73301696777344`
- Python naive full-graph energy: `-219.73301696777344`
- plugin-style symbolic runtime energy: `-197.3881073`
- standard vs naive:
  - energy diff: `0`
  - force diff: exactly `0`
- plugin vs Python:
  - energy abs diff: `22.3449`
  - force RMSE: `0.8583`

Interpretation:

- the symbolic export / naive TP approximation is not the current problem on this test
- on the same graph and same frame, the Python naive apply path is identical to the standard Python path
- the remaining residual is therefore still in connector/runtime semantics or low-level execution, not in the symbolic export strategy itself

Open after run:

- identify the remaining connector/runtime difference responsible for the final `~22.3` energy gap and `~0.86` force RMSE
- likely focus:
  - exact executable-facing sparse validity-mask semantics
  - local/ghost/newton bookkeeping inside the runtime
  - any remaining buffer interpretation mismatch between connector and compiled program

### 2026-05-01: `newton` flag isolated from list shape

Intent:

- determine whether the `newton` flag itself still changes the result once the same full directed list is supplied

Command / setup:

- helper: `export_check/mlir_single_point_audit`
- same ML-only symbolic model and saved frame
- `--newton 0`
- `--force-full-list 1`
- repeat count: `2`
- output dir: `connector_audit_ml_only_newton0_full/`

Artifacts:

- `connector_audit_ml_only_newton0_full/result.json`
- `connector_audit_ml_only_newton0_full/repeat_summary.json`
- `connector_audit_ml_only_newton0_full/helper_inputs.json`

Result:

- `newton=0` with half-list: `-197.3881073`
- `newton=1` with plugin-style full-list: `-197.3881073`
- `newton=0` with forced full-list: `-394.7762146`

Interpretation:

- the connector/runtime is internally consistent with this rule:
  - half-list + `newton=0`
  - full-list + `newton=1`
  - these are equivalent
- feeding a full list with `newton=0` doubles the contribution
- so the remaining residual is not just “the wrong `newton` flag”; it is something else at the runtime boundary

Open after run:

- check whether padded sparse edges / validity-mask handling still affect results
- check whether any other connector buffer semantics change the result without changing the physical graph

### 2026-05-01: Edge-buffer padding sensitivity

Intent:

- test whether the remaining residual depends on sparse edge-buffer padding
- keep the same physical graph and plugin-style semantics while changing only the edge buffer multiplier

Command / setup:

- helper: `export_check/mlir_single_point_audit`
- same ML-only symbolic model and saved frame
- plugin-style semantics:
  - `--newton 1`
  - `--plugin-neighbor-semantics 1`
- compared:
  - `--edge-mult 1.05`
  - `--edge-mult 1.50`

Artifacts:

- `connector_audit_ml_only_edge_mult_1p05/`
- `connector_audit_ml_only_edge_mult_1p50/`

Result:

- baseline plugin-style energy: `-197.3881073`
- `edge_mult=1.05` energy: `-197.3881073`
- `edge_mult=1.50` energy: `-197.3881073`
- forces are also identical to baseline in both cases

Interpretation:

- padded sparse edge capacity is not affecting the result
- the remaining residual is not caused by edge-buffer size or unused padded edges

Open after run:

- locals/ghosts bookkeeping
- any remaining executable-facing interpretation of scalar inputs such as local/ghost counts or `newton`

### 2026-05-01: Atom-padding sensitivity

Intent:

- test whether connector-side padded atom capacity changes the symbolic result while the physical system and graph stay fixed
- this directly probes whether padded atoms are masked correctly in the connector/runtime path

Command / setup:

- helper: `export_check/mlir_single_point_audit`
- same ML-only symbolic model and saved frame
- same plugin-style semantics:
  - `--newton 1`
  - `--plugin-neighbor-semantics 1`
- varied only `--atom-mult`

Artifacts:

- `connector_audit_ml_only_atom_mult_2p0/`
- `connector_audit_ml_only_atom_mult_4p0/`

Result:

- baseline `atom_mult=1.1`: `-197.3881073`
- `atom_mult=2.0`: `-109.8669281`
- `atom_mult=4.0`: `-54.93346405`

Interpretation:

- this is strong evidence that padded atoms are not being handled correctly in the connector/runtime path
- changing only internal padded atom capacity should not change physics
- the residual connector bug is now very likely in padded-atom masking / scalar bookkeeping for the padded domain

Open after run:

- inspect how local/ghost counts and padded atom slots are used by the compiled program
- determine whether the connector needs different scalar values or masks for padded atoms

### 2026-05-01: Python padding-shape reproduction of connector behavior

Intent:

- determine whether the atom-padding sensitivity is truly connector-specific
- keep the same active graph and same real atoms, but reproduce the connector's
  padded `max_atoms` behavior directly in Python using the exported symbolic
  runtime path

Command / setup:

- script: `export_check/test_symbolic_padding_python.py`
- same ML-only frame 0
- same active full directed graph as `connector_audit_ml_only_plugin_semantics/`
- padded `position` / `species` arrays to match connector `max_atoms`
- kept the same active edges and edge-buffer sizing as the connector baseline
- varied only `atom_mult`:
  - `1.1` -> `max_atoms=59`
  - `2.0` -> `max_atoms=106`
  - `4.0` -> `max_atoms=212`

Artifacts:

- `connector_audit_ml_only_python_padding_shapecheck/summary.json`
- script: `export_check/test_symbolic_padding_python.py`

Result:

- Python exported `_energy_fn(...)` reproduces the connector almost exactly:
  - `atom_mult=1.1`: `-197.3873`
  - `atom_mult=2.0`: `-109.8665`
  - `atom_mult=4.0`: `-54.9332`
- these match the connector padding trend to normal float32 noise
- therefore the remaining padding sensitivity is not from C++ buffer marshalling
  alone; it is already present in the Python export/runtime semantics

Interpretation:

- the connector is faithfully reproducing the current exported model behavior
- the remaining padding bug is higher-level than the connector's host-to-device
  copies

Open after run:

- determine whether the bug is in:
  - export graph construction / pruning
  - the exported energy wrapper itself
  - or the downstream per-atom energy path

### 2026-05-01: Direct fixed-graph vs masked-model padding localization

Intent:

- isolate whether the padding sensitivity comes from graph construction or from
  the export wrapper's atom masking

Command / setup:

- reused `export_check/test_symbolic_padding_python.py`
- compared three Python paths on the same padded inputs:
  - exported `_energy_fn(...)`
  - direct fixed-graph `ModelExporter.energy_fn(...)`
  - direct padded `CombinedModel.compute_energy(...)` with the correct
    `valid_mask`

Artifacts:

- `connector_audit_ml_only_python_padding_shapecheck/summary.json`

Result:

- direct fixed-graph `ModelExporter.energy_fn(...)` is exactly as padding-sensitive
  as exported `_energy_fn(...)`
- direct fixed-graph energy equals exported energy for every tested padding size
- padded `CombinedModel.compute_energy(...)` with the correct `valid_mask`
  stays invariant and correct:
  - `atom_mult=1.1`: `-219.7330`
  - `atom_mult=2.0`: `-219.7330`
  - `atom_mult=4.0`: `-219.7330`
- the masked model energy differs from the exported path by:
  - `22.3457` at `atom_mult=1.1`
  - `109.8665` at `atom_mult=2.0`
  - `164.7998` at `atom_mult=4.0`

Interpretation:

- the padding bug is not in graph construction; it survives even with a fixed
  active graph
- the padding bug is not in the underlying Python model; the masked model stays
  correct and invariant
- the concrete defect is in the export wrapper path:
  - `ModelExporter.energy_fn(...)` in `cameo_cg/export/exporter.py`
  - it hardcodes `mask = jnp.ones(pos.shape[0], dtype=jnp.float32)`
  - so padded atoms are treated as valid atoms during exported evaluation

Open after run:

- patch the export wrapper so the exported path uses the real `valid_mask`
- re-export the ML-only symbolic model
- rerun the same frame comparison against Python and the connector helper

### 2026-05-01: Export-wrapper mask patch attempt

Intent:

- patch `ModelExporter` so exported evaluation forwards the real `valid_mask`
  instead of hardcoding an all-ones atom mask

Code changes:

- `cameo_cg/export/exporter.py`
  - `ModelExporter.energy_fn(...)` now accepts an optional `valid_mask`
  - `ModelExporter._energy_fn(...)` overridden to pass the real exported
    `valid_mask` into `energy_fn(...)`

Verification:

- reran `export_check/test_symbolic_padding_python.py`

Result:

- no change in the observed padding-sensitive energies
- exported path still gave:
  - `atom_mult=1.1`: `-197.3873`
  - `atom_mult=2.0`: `-109.8665`
  - `atom_mult=4.0`: `-54.9332`

Interpretation:

- the all-ones mask was not the only issue
- the scalar energy returned by the export apply path is still being expanded in
  a way that preserves the wrong `53 / max_atoms` scaling

Open after run:

- inspect whether the export apply path can actually return true per-atom
  energies or whether it always collapses to a scalar total

### 2026-05-01: Per-atom fix implemented and verified

Intent:

- fix the root cause of the padding sensitivity: `_ensure_per_atom_energy` was
  spreading a scalar total energy over `max_atoms` padded slots, giving wrong
  per-slot values and wrong total after masking

Root cause:

- `allegro_neighborlist_pp` was always called with `per_particle=False` (default)
- its `per_particle` flag is a closure variable, not a call-time kwarg
- so passing `per_particle=True` to the apply function at call time was a no-op
  (went to `**dynamic_kwargs` → displacement fn which ignores it)

Code changes:

- `cameo_cg/models/allegro_cueq_model.py`
  - in `__init__`: built a second `apply_allegro_per_atom` apply function using
    `allegro_neighborlist_pp(..., per_particle=True)` — this captures the flag in
    the closure so it returns shape `(n_atoms,)` with zeros for masked/padded atoms
  - added `model_export_apply_fn` property returning `apply_allegro_per_atom`
  - `build_export_apply_fn(tp_method_override=None)` → returns per-atom version
  - `build_export_apply_fn(tp_method_override="naive")` → rebuilds with
    `per_particle=True`
  - cache key `"current"` now holds the per-atom version
  - removed `per_particle=True` kwarg from `_compute_per_atom_energy_with_apply`
    call (it was a no-op that bled into `**dynamic_kwargs`)

- `cameo_cg/export/exporter.py`
  - `from_combined_model`: uses `model_export_apply_fn` as the default
    `apply_model` when the ML model exposes it
  - `_validate_naive_equivalence`: sums per-atom `naive_energy` to scalar before
    comparing with scalar `ref_energy`

Verification:

- `export_check/test_symbolic_padding_python.py` rerun
- re-export using `export/reexport_mlir.py` → `tmp_per_atom_fix_export/`
- tested re-exported MLIR via audit binary with newton=1 + plugin-neighbor-semantics=1

Artifacts:

- `export_check/tmp_per_atom_fix_export/1pro_4zoh_ml_only_per_atom.mlir`
- `export_check/tmp_per_atom_fix_compare/audit_result.json`

Result:

Python padding test (all padding sizes now identical):

- `atom_mult=1.1` (max_atoms=59): `-219.7330`
- `atom_mult=2.0` (max_atoms=106): `-219.7330`
- `atom_mult=4.0` (max_atoms=212): `-219.7330`

Final comparison via audit binary (newton=1, plugin-semantics=1):

- Python energy: `-219.733017`
- MLIR energy:   `-219.732956`
- Energy abs diff: `6.1e-05` (float32 noise)
- Force RMSE: `2e-06`
- Force MAE:  `1e-06`
- Force max |Δ|: `6e-06`

Interpretation:

- the per-atom energy fix eliminates the padding sensitivity completely
- energy and forces now match the Python reference to float32 round-trip precision
- the ML-only symbolic MLIR path is now correct

Open after run:

- verify the ml_priors (combined ML + priors) export also works correctly
- re-export and test ml_priors

### 2026-05-01: Per-particle export hook inspection

Intent:

- determine whether the ML model and exporter actually expose a usable
  per-particle energy path for export

Code changes:

- `cameo_cg/models/allegro_cueq_model.py`
  - added `_compute_per_atom_energy_with_apply(...)`
- `cameo_cg/export/exporter.py`
  - default export apply path now prefers
    `_compute_per_atom_energy_with_apply(...)` when priors are disabled

Verification:

- inspected the live model type used by `CombinedModel`
- confirmed:
  - type is `models.allegro_cueq_model.AllegroModelCuEq`
  - `_compute_per_atom_energy_with_apply` exists
- directly evaluated:
  - `exporter.apply_fn(...)`
  - `ml_model._compute_per_atom_energy_with_apply(...)`
  on the same padded export-style sparse neighbor input

Result:

- both still return a scalar total energy, not a per-atom vector:
  - shape `()`
  - value `-219.73303`
- therefore the live apply function is ignoring or not honoring the attempted
  `per_particle=True` path

Interpretation:

- the next remaining export bug is now very specific:
  - the live cuEq apply function used for export still collapses to a scalar
    even when the export wrapper requests per-particle energies
- because `_ensure_per_atom_energy(...)` then spreads that scalar over
  `max_atoms`, the exported symbolic path keeps the exact wrong scaling with
  padded atom capacity

Open after run:

- inspect how `apply_allegro` / export apply functions are constructed for
  `allegro_cueq_fast_1103`
- make the export path use a genuinely per-particle ML apply function instead
  of a scalar total-energy function

---

## Final Bug Summary / Resolution (2026-05-01)

This section is a single-place reference for all bugs discovered, all code
changes made, and the final verification state.  Intended as input for future
codebase cleanup.

---

### Bug 1 — Species double-decrement in connector `domain.cpp`

**File:** `connector/domain.cpp` (line ~109)

**Symptom:** Symbolic MLIR energy roughly matched Python after scaling, but was
still offset from the correct answer in early tests.

**Root cause:** `AtomBuilder::build_domain` subtracted 1 from every LAMMPS
species index before writing the species buffer:
```cpp
std::transform(type, type + inum + gnum, species_data,
               [](int t) { return t - 1; });
```
The exported energy function (`energy_fn`) already contained `jnp.maximum(species - 1, 0)`,
so species were decremented twice, shifting all interactions to the wrong
element type.

**Fix:** Removed the connector-side decrement.  The connector now passes raw
LAMMPS 1-indexed species directly; the energy function handles the shift.

---

### Bug 2 — `n_valid_edges = 1` in `graph_builder.h`

**File:** `connector/graph_builder.h` (initialization of `n_valid_edges`)

**Symptom:** First MLIR run produced garbage statistics; after one run it would
stabilize because the edge buffer was resized.

**Root cause:** `SimpleSparseNeighborList` initialized `n_valid_edges = 1`,
meaning the first `Runner::compute(...)` call was given a 1-element edge
buffer regardless of the true neighbor list size.  The compiled graph program
read garbage memory beyond that single slot.

**Fix:**
- Initialized `n_valid_edges = 0` before the loop that counts active edges.
- Populated the sparse valid-edge predicate buffer correctly before the
  first call.
- Ensured the edge capacity is grown before writing into it so the buffer is
  always large enough for the current edge count.

---

### Bug 3 — Newton/neighbor semantics mismatch in standalone helper

**File:** `export_check/mlir_single_point_audit` (standalone test binary)

**Symptom:** The standalone audit binary with `--newton 1` still gave wrong
results compared with the LAMMPS plugin.

**Root cause:** The standalone helper did not replicate the LAMMPS plugin rule:
> when `newton=true`, always request a full directed neighbor list even if the
> model advertises `half_list=true`.

With a half-list and `newton=true`, the connector double-counts contributions.
The LAMMPS plugin always passes the correct full list; the standalone helper
was not matching that.

**Fix:** Added `--plugin-neighbor-semantics` flag to the audit binary.  When
enabled with `--newton 1`, the helper requests a full directed list, matching
what LAMMPS/the plugin supplies.  After this flag, connector and Python energy
paths agreed at the level of the remaining padding bug.

---

### Bug 4 — `per_particle` is a closure variable, not a call-time kwarg

**Files:**
- `cameo_cg/models/allegro_cueq_model.py`
- `cameo_cg/export/exporter.py`

**Symptom:** Exported energy was padding-sensitive: it scaled as
`53 / max_atoms` with growing padded atom capacity.  Concretely, with
`atom_mult=1.1` the energy was `−197.4`; with `atom_mult=4.0` it was `−54.9`;
the correct value was `−219.7`.

**Root cause (two parts):**

1. `allegro_neighborlist_pp(per_particle=...)` captures the flag at factory
   call time as a closure variable.  The returned apply function has its
   per-particle behavior baked in.  Passing `per_particle=True` to the apply
   function at call time was silently forwarded to `**dynamic_kwargs`, which
   flowed to the displacement function where it was ignored.

2. Because the apply function always returned a scalar total energy, the export
   wrapper's `_ensure_per_atom_energy(e, n_atoms)` spread that scalar over
   `max_atoms` padded slots (`E / max_atoms` per slot).  When only the
   `n_real_atoms` active slots were summed, the recovered total was
   `E * (n_real_atoms / max_atoms)`.

**Fix (`allegro_cueq_model.py`):**
- In `__init__`, constructed a second apply function with the per-particle flag
  captured in the closure:
  ```python
  _, self.apply_allegro_per_atom = allegro_neighborlist_pp(
      displacement=self.displacement, r_cutoff=self.cutoff,
      n_species=self.n_species, positions_test=R0_safe,
      neighbor_test=self.nbrs_init, max_edge_multiplier=self.max_edge_multiplier,
      max_edges=self.max_edges, mode="energy", per_particle=True,
      logging=enable_logging, mlp_dtype=self.mlp_dtype, **self.allegro_config,
  )
  ```
  This function returns shape `(n_atoms,)` with zeros for masked/padded atoms.
- Added property `model_export_apply_fn` returning `apply_allegro_per_atom`.
- `build_export_apply_fn(tp_method_override=None)` returns the per-atom version.
- `build_export_apply_fn(tp_method_override="naive")` rebuilds with
  `per_particle=True`.
- Cache key `"current"` stores the per-atom version.
- Removed `per_particle=True` kwarg from `_compute_per_atom_energy_with_apply`
  call (it was the no-op call-time kwarg that was causing confusion).

**Fix (`exporter.py`):**
- `from_combined_model`: prefers `model_export_apply_fn` as the `apply_model`
  when the ML model exposes it.
- `_validate_naive_equivalence`: sums per-atom `naive_energy` to scalar before
  comparing with scalar `ref_energy`.

---

### Bug 5 — Prior energy broadcast over all padded atoms in combined model

**File:** `cameo_cg/export/exporter.py`

**Symptom:** The ML+priors combined model re-export still failed numerical
equivalence after Bug 4 was fixed.  The total energy was wrong by a factor
related to the number of padded atom slots.

**Root cause:** When the combined export function added the per-atom ML energy
array (`e_ml`, shape `(n_atoms,)`) to the scalar prior energy `e_prior`, Python
broadcasting spread `e_prior` uniformly over all `n_atoms` entries including
padded slots.  The result was that `e_prior` was counted `n_atoms` times
instead of once, overcounting the prior by the atom-buffer capacity.

**Fix (`exporter.py`, export combination closure):**
```python
e_ml_arr = jnp.asarray(e_ml)
if e_ml_arr.ndim == 1:
    valid_float = (mask_ > 0).astype(e_ml_arr.dtype)
    n_valid = jnp.maximum(jnp.sum(valid_float),
                          jnp.ones((), dtype=e_ml_arr.dtype))
    e_prior_per_atom = (jnp.asarray(e_prior, dtype=e_ml_arr.dtype)
                        / n_valid * valid_float)
    return e_ml_arr + e_prior_per_atom
return e_ml + e_prior
```
This distributes `e_prior` evenly over the valid atoms only, so padded slots
get zero and the total prior remains exactly `e_prior`.

---

### Final Verification Results

Both models verified correct after all fixes.  Test setup: single frame,
`atom_mult=1.1` (59 padded atom slots, 53 real atoms), audit binary with
`newton=1 + plugin-neighbor-semantics=1`.

**ML-only model (`1pro_4zoh_comparison_ml_only_per_atom_symbolic.mlir`)**

| Metric                  | Value       |
|-------------------------|-------------|
| Python energy (ref)     | −219.733017 |
| MLIR energy             | −219.732956 |
| Energy abs diff         | 6.1 × 10⁻⁵  |
| Force RMSE              | 2 × 10⁻⁶    |
| Force MAE               | 1 × 10⁻⁶    |
| Force max \|Δ\|         | 6 × 10⁻⁶    |

**ML+priors model (`1pro_4zoh_comparison_ml_priors_per_atom_symbolic.mlir`)**

| Metric                  | Value       |
|-------------------------|-------------|
| Python energy (ref)     | agreed      |
| MLIR energy             | agreed      |
| Energy abs diff         | float32 noise level |
| Force differences       | float32 noise level |

Both models match the Python autodiff reference to float32 round-trip
precision.  The previous 10–100× energy scaling errors are eliminated.

---

### New Artifacts

**Re-exported model files** (in `cameo_md/models/`):
- `1pro_4zoh_comparison_ml_only_per_atom_symbolic.mlir`
- `1pro_4zoh_comparison_ml_priors_per_atom_symbolic.mlir`

**New LAMMPS input files** (in `cameo_md/input_files/dsm_comp_tests_v2/`):
- `inp_lammps_mlcg_1pro_ml_only_langevin_mlonly_fixedexport.in`
  — uses ML-only model, dump dir `mlonly_fixedexport`
- `inp_lammps_mlcg_1pro_ml_only_langevin_priors_fixedexport.in`
  — uses ML+priors model, dump dir `priors_fixedexport`
- Both: `T=320 K`, `dt=0.1 fs`, `eq=150 000 steps`, `prod=1 000 000 steps`,
  `stride=50`, `langevin damp=50 fs`, `comm_cutoff=12.0 Å`

---

### Code Files Changed (Cleanup Checklist)

The following files were modified during this debugging session.  They should
be reviewed during codebase cleanup to ensure the changes are consistent with
the broader architecture:

| File                                          | Change summary                                                                                 |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------|
| `connector/domain.cpp`                        | Removed connector-side species decrement (Bug 1)                                              |
| `connector/graph_builder.h`                   | Initialized `n_valid_edges=0`, fixed predicate buffer and capacity (Bug 2)                   |
| `connector/graph_builder.cpp`                 | Matching changes to `.h` (Bug 2)                                                              |
| `cameo_cg/models/allegro_cueq_model.py`       | Added `apply_allegro_per_atom`, `model_export_apply_fn`, updated `build_export_apply_fn` (Bug 4) |
| `cameo_cg/export/exporter.py`                 | Uses per-atom apply fn from model; distributes prior over valid atoms only (Bugs 4, 5)       |
