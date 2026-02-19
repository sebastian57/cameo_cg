# Plan: Per-Residue Evaluation, Protein-Aware Batching, EMA

All three features are **optional** — enabled via flags; default behaviour is unchanged.

---

## 1. Per-Residue Force Error Decomposition

### Goal
Identify which amino acid types and chain positions are hardest to model,
to guide architecture decisions and prior design.

### How it works
For each atom `i` across `N_frames` validation frames:
```
RMSE[i] = sqrt( mean_n( ||F_pred[n,i] - F_ref[n,i]||² ) )
```
Grouped by `species[i]` (AA type) → per-AA bar chart.
Optionally by chain position index → N/C-terminal effect plot.

### Files created/modified

**New: `evaluation/per_residue.py`**
```
compute_per_residue_errors(model, params, dataset, config, n_frames=100)
    → dict: {
        'rmse_per_atom':    float32[N_max],    # per-position mean RMSE
        'species':          int32[N_max],      # AA type for each position
        'mask':             float32[N_max],    # 1=real atom, 0=padded
        'aa_names':         list[str],         # species_id -> AA name mapping
      }

plot_per_residue_rmse(results, output_path)
    → bar chart of RMSE grouped by AA type (sorted by RMSE)
    → line plot of RMSE vs chain position
```

**Modify: `scripts/evaluate_forces.py`**
- Add `--mode per-residue` option to `choices`
- Reuse existing checkpoint loading and model setup
- Output: `<model_id>_per_residue_rmse.png` + `.txt`

### Key implementation notes
- The `aa_to_id` dict is saved in the NPZ as a pickled object; load with
  `np.load(..., allow_pickle=True)['aa_to_id'].item()` to get the mapping.
- For per-AA grouping, invert `aa_to_id` to get `id_to_aa`.
- species array has shape `[n_frames, N_max]`; take `species[0]` since it's
  constant across frames for a single-protein dataset.
- Use `jax.vmap` over frames for efficient batch force computation.
- Only include atoms where `mask[i] > 0` (exclude padding).

---

## 2. EMA (Exponential Moving Average) of Parameters — Option B

### Goal
Export EMA-smoothed parameters without changing the training loop.

### Design: Post-hoc EMA from checkpoints
With `checkpoint_freq=10`, load each `epoch*.pkl` and compute running EMA
over the saved checkpoints. No changes to `trainer.py` required.

EMA update:
```python
ema = decay * ema + (1 - decay) * current_params   # decay = 0.999
```

### New file: `scripts/compute_ema.py`
```
python scripts/compute_ema.py --checkpoint_dir ./checkpoints_allegro \
    --output exported_models/ema_params.pkl [--decay 0.999]
```

Steps:
1. Glob `checkpoint_dir/epoch*.pkl`, sort by epoch number
2. Load each checkpoint, extract `trainer_state['params']`
3. Compute running EMA over the sequence
4. Save final EMA params as `pkl` file

### Notes
- With 8 checkpoints (epochs 10, 20, ..., 80) the EMA is coarse but free.
- Finer control later via Option A (epoch-by-epoch loop) if needed.

---

## 3. Protein-Aware Batching (for multi-protein training)

### Goal
When training on many proteins of different lengths, padding all to global
N_max wastes compute. Group proteins into length buckets so each bucket
uses a bucket-specific N_max.

### Design: fixed bucket count via explicit boundaries

**Bucket boundaries** (e.g. `--bucket_boundaries 100 200`):
- Bucket 0: N_real ≤ 100 → batch N_max = 100
- Bucket 1: N_real ∈ (100, 200] → batch N_max = 200
- Bucket 2: N_real > 200 → batch N_max = max in this bucket

Default: auto-split into 3 equal-count groups (caps compilation at 3).
User controls compilation count by setting number of boundaries.

**Compilation cost**: one JIT compilation per bucket. With default 3 buckets,
this is 3× the compilation of single-protein training — amortized over
training, this is acceptable.

### Files to create/modify

**`data_prep/pad_and_combine_datasets.py`**
- Fix `pad_individual_npz` (imported by `run_pipeline.py` but missing)
- Refactor `combine_and_pad_npz` to extract `_combine_datasets` helper
  (accepts a pre-computed `aa_to_id` to allow global mapping across buckets)
- Add `combine_and_pad_npz_bucketed(paths, out_dir, bucket_boundaries=None, n_buckets=3)`
  → outputs `out_dir/bucket_N{nmax:04d}.npz` per non-empty bucket
  → global `aa_to_id` built across ALL proteins for consistent species encoding

**`data_prep/run_pipeline.py`** — Step 3 changes
- Add `--bucket_boundaries N [N ...]` (e.g. `--bucket_boundaries 100 200`)
- Add `--n_buckets N` (default 3, used when no explicit boundaries given)
- When either flag is set, call `combine_and_pad_npz_bucketed` instead of
  `combine_and_pad_npz`; output goes to `03_bucketed_npz/`
- `--no_combine` and `--bucket_boundaries` are mutually exclusive
- Prior fitting still receives all bucket paths (all proteins contribute)

**`data/loader.py`**
- Add `BucketedDatasetLoader(bucket_dir_or_paths, max_frames=None, seed=42)`
  → finds/loads all `bucket_N*.npz` files
  → exposes `.buckets` as list of `(N_max, DatasetLoader)` sorted by N_max

**`scripts/train.py`**
- Add `--multi-protein-dir <dir>` argument (optional)
- When given: uses `BucketedDatasetLoader`, trains sequentially over buckets
  with param transfer between them
  - Bucket 0: random init → train all epochs → save params
  - Bucket 1: init from bucket 0 params → train all epochs → save params
  - ...
- Final export uses params from last bucket
- Single-protein mode (default) is unchanged

### Notes
- The topology (bonds, angles, dihedrals, rep_pairs) changes per bucket_max,
  so each bucket gets its own `TopologyBuilder` via `CombinedModel`.
- The `avg_num_neighbors` is computed from data per bucket (already done).
- LAMMPS/MLIR export still uses a single N_max (the largest encountered).
- Round-robin training (interleaving buckets per epoch) is a future extension.

---

## Implementation order

1. **EMA** — new script only, zero risk
2. **Per-residue evaluation** — new file + small extension to evaluate_forces.py
3. **Protein-aware batching** — data prep + loader + train.py extension
