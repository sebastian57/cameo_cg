# Extending cameo_cg — Adding Models, Priors, Optimizers, and Running Experiments

Step-by-step guide for adding new components and running training experiments.

---

## 1. Adding a New ML Model

### What you need

An ML model in this pipeline computes a scalar energy from atomic coordinates,
species, and a neighbor list.  The framework handles forces via `jax.grad`
automatically.

### Steps

**a) Create `models/my_model.py`**

Subclass `BaseMLModel` and decorate with `@register_ml_model`:

```python
import jax
import jax.numpy as jnp
from jax_md import space, partition

from models.base_model import BaseMLModel, register_ml_model, resolve_compute_dtype
from utils.logging import model_logger


def resolve_neighbor_list_format(name):
    """Reuse the helper already in allegro_model.py or factor it out."""
    from models.allegro_model import resolve_neighbor_list_format
    return resolve_neighbor_list_format(name)


@register_ml_model("my_model")
class MyModel(BaseMLModel):

    def __init__(self, config, R0, box, species, N_max, n_species_override=None):
        self.config = config
        self.N_max = N_max
        self.compute_dtype_name, self.compute_dtype = resolve_compute_dtype(config)

        self.cutoff = config.get_cutoff()
        self.dr_threshold = config.get_dr_threshold()
        fmt_name, fmt = resolve_neighbor_list_format(config.get_neighbor_list_format())
        self.neighbor_list_format_name = fmt_name
        self.neighbor_list_format = fmt

        # Read model-specific hyperparameters from config
        my_cfg = dict(config.get("model", "my_model", default={}))
        extra_cap = int(my_cfg.pop("neighbor_extra_capacity", 64))
        self.max_edge_multiplier = float(my_cfg.pop("max_edge_multiplier", 1.25))

        # Build neighbor list
        self.displacement, self.shift = space.free()
        self.nneigh_fn = partition.neighbor_list(
            self.displacement, box=jnp.asarray(box, dtype=jnp.float32),
            r_cutoff=self.cutoff, dr_threshold=self.dr_threshold,
            fractional_coordinates=False, format=self.neighbor_list_format,
        )
        self.nbrs_init = self.nneigh_fn.allocate(R0, extra_capacity=extra_cap)

        # Species handling
        species_safe = jnp.where(jnp.asarray(species) >= 0, species, 0).astype(jnp.int32)
        n_species_data = int(jnp.max(species_safe)) + 1
        self.n_species = (
            max(n_species_data, int(n_species_override))
            if n_species_override is not None else n_species_data
        )

        # Build init/apply from your model library
        # self.init_fn, self.apply_fn = my_model_lib.build(...)

    def initialize_params(self, rng_key):
        return self.init_fn(rng_key, ...)

    def compute_energy(self, params, R, mask, species, neighbor=None, segment_id=None):
        # Your model's forward pass returning a scalar energy
        return self.apply_fn(params, R, neighbor, species)

    @property
    def model_apply_fn(self):
        return self.apply_fn
```

**b) Register for eager import in `models/__init__.py`**

Add one line so the `@register_ml_model` decorator fires at import time:

```python
from . import my_model as _mym  # noqa: F401
```

Add it next to the existing eager imports (allegro_model, mace_model, painn_model).

**c) Add a config accessor (optional)**

In `config/manager.py`, add:

```python
def get_my_model_config(self) -> Dict[str, Any]:
    return self.get("model", "my_model", default={})
```

**d) Add config section to `configs/base_config.yaml`**

```yaml
  # --- MyModel hyperparameters (used when ml_model: my_model) ---
  my_model:
    hidden_dim: 128
    num_layers: 4
    neighbor_extra_capacity: 64
    max_edge_multiplier: 1.25
```

**e) Use it**

In your YAML config:

```yaml
model:
  ml_model: my_model
  my_model:
    hidden_dim: 128
    num_layers: 4
```

No changes needed in `CombinedModel`, `Trainer`, or `train.py` — the registry
handles dispatch.

### Key files to reference

| File | Role |
|------|------|
| `models/base_model.py` | `BaseMLModel` ABC, registry, `resolve_compute_dtype` |
| `models/allegro_model.py` | Reference implementation (neighbor list, species, masking) |
| `models/combined_model.py` | Instantiates ML model via `get_ml_model_class()` |
| `models/__init__.py` | Eager imports for decorator registration |

---

## 2. Adding a New Prior Energy Term

Prior terms are physics-based energy contributions (bonds, angles, repulsive,
etc.) that are summed alongside the ML energy.  Each term is a method on
`PriorEnergy` that returns a scalar energy.

### Steps

**a) Add the energy method to `models/prior_energy.py`**

Add a method following the existing pattern:

```python
def compute_my_term_energy(self, R, mask, species=None, params=None):
    """Compute my new energy term."""
    p = params if params is not None else self.params

    # Use self.topology for index arrays, self.displacement for distances.
    # Example: pairwise term over custom pairs
    pairs = self.my_term_pairs  # set in __init__
    Ri = R[pairs[:, 0]]
    Rj = R[pairs[:, 1]]
    dr = jax.vmap(self.displacement)(Ri, Rj)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-12)

    # Pair mask (both atoms must be real)
    pair_mask = mask[pairs[:, 0]] * mask[pairs[:, 1]]

    # Energy from parameter
    r0 = p["my_term_r0"]
    k = p["my_term_k"]
    E_pair = 0.5 * k * (dist - r0) ** 2
    return jnp.sum(E_pair * pair_mask)
```

**b) Wire it into `PriorEnergy.__init__`**

Set up any index arrays you need:

```python
# In __init__, after the existing topology calls:
self.my_term_pairs = topology.get_some_pairs(...)  # or build your own
```

Add the default parameter to `_build_parametric_params()` or
`_build_typed_params()`:

```python
"my_term_r0": jnp.asarray(prior_cfg.get("my_term_r0", 4.0), dtype=jnp.float32),
"my_term_k": jnp.asarray(prior_cfg.get("my_term_k", 10.0), dtype=jnp.float32),
```

**c) Wire it into `compute_energy()`**

In `PriorEnergy.compute_energy()`:

```python
E_my_term_raw = self.compute_my_term_energy(R, mask, species=species, params=p)

# Apply weight
E_my_term = self.weights.get("my_term", 0.0) * E_my_term_raw

# Add to total
E_total = E_bond + E_angle + ... + E_my_term

# Add to return dict
return {
    ...,
    "E_my_term": E_my_term,
    "E_total": E_total,
}
```

**d) Add the weight default**

In `config/manager.py`, add to `_DEFAULT_PRIOR_WEIGHTS`:

```python
_DEFAULT_PRIOR_WEIGHTS: Dict[str, float] = {
    ...,
    "my_term": 0.0,  # off by default
}
```

And in `configs/base_config.yaml`:

```yaml
    weights:
      ...
      my_term: 0.0
```

**e) If you want force decomposition for the new term**

In `combined_model.py`, method `compute_force_components()`, add your term
to the VJP tuple:

```python
def all_energies(R_):
    comps = self.compute_components(params, R_, mask, species)
    return (
        comps["E_total"],
        comps["E_ml"],
        comps["E_bond"],
        ...
        comps["E_my_term"],    # <-- add here
    )

# Update n in _force(idx, n=...) to match tuple length
# Add to the return dict:
"F_my_term": _force(N),
```

### Key files

| File | Role |
|------|------|
| `models/prior_energy.py` | All prior energy terms, `compute_energy()`, parameter init |
| `models/combined_model.py` | Force decomposition via VJP |
| `models/topology.py` | Index arrays (bonds, angles, dihedrals, pairs) |
| `config/manager.py` | `_DEFAULT_PRIOR_WEIGHTS`, `get_prior_weights()` |

---

## 3. Adding a New Optimizer

Optimizers use a registry pattern in `training/optimizers.py`.  Adding one
requires a single decorated function.

### Steps

**a) Add a factory function to `training/optimizers.py`**

```python
@register_optimizer("my_optimizer")
def _my_optimizer(schedule, cfg):
    return optax.adam(  # or any optax transform
        learning_rate=schedule,
        b1=cfg.get("beta1", 0.9),
        b2=cfg.get("beta2", 0.999),
    )
```

The `schedule` argument is an `optax` learning rate schedule built
automatically by `create_optimizer()` from the config keys `lr`, `peak_lr`,
`end_lr`, `warmup_steps`, `decay_steps`.

If your optimizer handles weight decay internally (like `adamw` or `lamb`),
set `handles_weight_decay=True`:

```python
@register_optimizer("my_optimizer", handles_weight_decay=True)
def _my_optimizer(schedule, cfg):
    return optax.my_optax_optimizer(
        learning_rate=schedule,
        weight_decay=cfg.get("weight_decay", 0.0),
    )
```

**b) Add config section**

In your YAML config file and in `configs/base_config.yaml`:

```yaml
optimizer:
  my_optimizer:
    lr: 0.001
    peak_lr: 0.01
    end_lr: 0.0001
    warmup_steps: 200
    decay_steps: 5000
    beta1: 0.9
    beta2: 0.999
    grad_clip: 5.0
    weight_decay: 0.0
```

**c) Reference in training stages**

```yaml
training:
  stages:
    - optimizer: my_optimizer
      epochs: 100
```

`ConfigManager.get_training_stages()` validates the name against the
registry and raises a clear error if the optimizer is unknown.

### Key files

| File | Role |
|------|------|
| `training/optimizers.py` | Registry, `@register_optimizer`, `create_optimizer()` |
| `config/manager.py` | `get_training_stages()` (validates names), `get_optimizer_config()` |

---

## 4. Adding a New CG Mapping

CG mappings convert all-atom trajectories into coarse-grained representations.
The current mapping (`data_prep/cg_1bead.py`) uses one bead per amino acid
residue.

### Steps

**a) Create `data_prep/cg_<your_mapping>.py`**

Follow the same CLI interface as `cg_1bead.py`:

```python
#!/usr/bin/env python3
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Input all-atom NPZ")
    parser.add_argument("--pdb", required=True, help="PDB topology")
    parser.add_argument("--output", required=True, help="Output CG NPZ")
    args = parser.parse_args()

    # Load all-atom data
    # Apply your mapping (e.g. 2-bead, sidechain centers, ...)
    # Write CG NPZ with keys: R, F, species, mask, box, (optionally id_to_aa)

    np.savez(args.output, R=R_cg, F=F_cg, species=species_cg, mask=mask_cg, box=box)

if __name__ == "__main__":
    main()
```

The downstream pipeline (`DatasetLoader`, `CombinedModel`, etc.) only
requires the NPZ to contain `R`, `F`, `species`, `mask`, and `box`.  The
topology (bond/angle/dihedral indices) is generated automatically by
`TopologyBuilder` based on `N_max` and the linear chain assumption.

**b) If your mapping changes topology**

If your CG representation is not a simple linear chain (e.g. sidechain beads),
you will need to extend `models/topology.py` with a new topology builder or
modify `TopologyBuilder` to accept a custom connectivity.

---

## 5. Running a Training Experiment

### Single run

```bash
# Edit your config
cp configs/base_config.yaml my_config.yaml
# ... customize ...

# Submit
sbatch scripts/run_training.sh my_config.yaml
```

### Suite of experiments (array jobs)

Create a directory with multiple config files:

```
experiments/
├── config_lr_high.yaml
├── config_lr_low.yaml
├── config_3layers.yaml
└── config_5layers.yaml
```

Submit all as a SLURM array job:

```bash
bash scripts/submit_suite.sh --input_dir ./experiments/ --name lr_sweep
```

Each config becomes one array task.  Results land in each config's
`paths.output_dir`.

### Resume from checkpoint

```bash
sbatch scripts/run_training.sh my_config.yaml --resume auto
```

---

## 6. Evaluating Results

All analysis scripts live in `analysis_tests/`.

### Suite-level analysis

After all array jobs complete:

```bash
# Via SLURM wrapper
sbatch scripts/run_analysis.sh /path/to/suite/output/

# Or directly
python analysis_tests/analyze_suite.py /path/to/output/ --detailed-force-eval
```

Produces:
- CSV summary table across all runs
- Tail-loss comparison plots
- Per-run force evaluation (if `--detailed-force-eval`)

### Per-model force evaluation

```bash
# Full model (ML + priors)
python analysis_tests/evaluate_forces.py model_params.pkl config.yaml --frames 50

# Prior-only
python analysis_tests/evaluate_forces.py config.yaml --mode prior-only --frames 20

# ML-only
python analysis_tests/evaluate_forces.py model_params.pkl config.yaml --mode ml-only
```

### Loss curve plotting

```bash
python analysis_tests/visualizer.py train.log config.yaml output.png
```

### Unit tests

```bash
python -m pytest analysis_tests/test_*.py
```

---

## 7. Putting It All Together — Worked Example

> Goal: add a new model (`gnn_v2`), a new prior (`hydrophobic`), and a new
> optimizer (`radam`), then run a sweep of 3 configurations.

### Step 1: Create the model

```
models/gnn_v2_model.py    # BaseMLModel subclass, @register_ml_model("gnn_v2")
```

Register in `models/__init__.py`:

```python
from . import gnn_v2_model as _gnn  # noqa: F401
```

### Step 2: Create the prior

In `models/prior_energy.py`:
- Add `compute_hydrophobic_energy()` method
- Add `"hydrophobic_k"` to parameter dict in `_build_parametric_params()`
- Add to `compute_energy()` and the return dict
- Add `"hydrophobic": 0.0` to `_DEFAULT_PRIOR_WEIGHTS` in `config/manager.py`

In `models/combined_model.py`:
- Add `comps["E_hydrophobic"]` to the VJP tuple in `compute_force_components()`

### Step 3: Create the optimizer

In `training/optimizers.py`:

```python
@register_optimizer("radam")
def _radam(schedule, cfg):
    return optax.radam(learning_rate=schedule, ...)
```

### Step 4: Write configs

```yaml
# experiments/gnn_v2_hydro_radam.yaml
model:
  ml_model: gnn_v2
  use_priors: true
  gnn_v2:
    hidden_dim: 128
  priors:
    weights:
      hydrophobic: 0.5

optimizer:
  radam:
    lr: 0.001
    peak_lr: 0.01
    decay_steps: 5000

training:
  stages:
    - optimizer: radam
      epochs: 100
```

### Step 5: Submit and evaluate

```bash
bash scripts/submit_suite.sh --input_dir ./experiments/ --name gnn_v2_test
# ... wait for completion ...
python analysis_tests/analyze_suite.py ./outputs/gnn_v2_test/ --detailed-force-eval
```

---

## Quick Reference: Config Skeleton

```yaml
seed: 42
model_context: my_experiment
model_id: run_01

debug:
  neighbor_logging: false
  shape_trace: false
  model_logging: false

paths:
  output_dir: ./outputs/my_run

data:
  path: data_prep/datasets/my_data.npz

model:
  ml_model: allegro          # allegro | mace | painn | allegro_cueq | ...
  use_priors: true
  cutoff: 10.0
  allegro:
    num_layers: 3
    max_ell: 2
  priors:
    use_spline_priors: true
    spline_file: data_prep/datasets/fitted_priors_spline.npz
    weights:
      bond: 0.5
      angle: 0.25
      repulsive: 1.0

optimizer:
  adabelief:
    lr: 0.001
    peak_lr: 0.03

training:
  stages:
    - optimizer: adabelief
      epochs: 80
  batch_per_device: 4
```

Full reference with every option: `configs/base_config.yaml`
