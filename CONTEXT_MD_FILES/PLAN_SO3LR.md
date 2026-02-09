# SO3LR Model Integration Plan

> **Status**: Planning
> **Last Updated**: 2026-02-09

---

## 1. Overview

Integrate SO3LR ("Solar") as a third ML backbone option in the cameo_cg framework. Unlike the MACE integration (which is a near drop-in replacement for Allegro), SO3LR presents **fundamental architectural differences** that require a custom wrapper approach.

**SO3LR source**: `/p/project1/cameo/schmidt36/so3lr/` (cloned from the official repo)

---

## 2. Why This Is Challenging

SO3LR differs from Allegro/MACE in several critical ways:

| Aspect | Allegro/MACE | SO3LR |
|--------|-------------|-------|
| **Framework** | chemutils (Haiku + `hk.transform`) | mlff library (Flax-like, orbax checkpoints) |
| **Init pattern** | `init_fn(rng, R, nbrs, species) -> params` | Pre-trained params loaded from `params.pkl` |
| **Apply pattern** | `apply_fn(params, R, nbrs, species, mask) -> E` | `So3lrPotential(graph) -> E` via `Graph` namedtuple |
| **Neighbor lists** | Single jax_md Dense list | **Two lists**: short-range (4.5 A) + long-range (12-100 A) |
| **Graph format** | `(position, neighbor, species, mask)` | Custom `Graph` namedtuple with 17 fields |
| **Energy model** | Pure ML | ML (SO3krates) + ZBL repulsion + Coulomb + Dispersion |
| **Species encoding** | Integer IDs (0-indexed AA types) | Atomic numbers (Z) |
| **Training** | From scratch (random init) | **Pre-trained universal model** (fine-tuning paradigm) |
| **Dependencies** | e3nn_jax, haiku | mlff, jraph, jax-pme, glp |

---

## 3. Integration Strategy Options

### Option A: JAX-MD Interface Wrapper (Recommended)

Use SO3LR's existing `to_jax_md()` function to create a jax_md-compatible energy function, then wrap it to match our `AllegroModel`/`MACEModel` interface.

**Pros**: Uses SO3LR's tested integration path; handles both neighbor lists internally
**Cons**: Requires dual neighbor list management; must adapt species encoding

### Option B: Raw Potential Wrapper

Call `So3lrPotential` directly, constructing the `Graph` namedtuple ourselves.

**Pros**: Full control over graph construction
**Cons**: Must manually handle all 17 Graph fields; fragile to upstream changes

### Option C: ASE Calculator Wrapper

Use SO3LR's ASE calculator as a black box.

**Pros**: Simplest
**Cons**: Not JIT-compilable; no autodiff for forces in training loop; incompatible with chemtrain's `ForceMatching` trainer. **NOT VIABLE for training.**

**Decision**: **Option A** (JAX-MD interface). This gives us a JIT-compilable, autodiff-compatible energy function while leveraging SO3LR's tested code.

---

## 4. Key Challenges and Solutions

### 4.1 Dual Neighbor Lists

SO3LR requires two neighbor lists:
- **Short-range (SR)**: cutoff = 4.5 A (neural network)
- **Long-range (LR)**: cutoff = 12-100 A (Coulomb, dispersion)

Our framework currently manages a single neighbor list via `jax_md.partition.neighbor_list`.

**Solution**: The `SO3LRModel` wrapper manages both internally:
```python
class SO3LRModel:
    def __init__(self, ...):
        # to_jax_md returns neighbor_fn, neighbor_fn_lr, energy_fn
        self.neighbor_fn, self.neighbor_fn_lr, self._energy_fn = to_jax_md(
            potential=So3lrPotential(...),
            displacement_or_metric=self.displacement,
            ...
        )
        # Allocate both
        self.nbrs_sr_init = self.neighbor_fn.allocate(R0)
        self.nbrs_lr_init = self.neighbor_fn_lr.allocate(R0)

    def compute_energy(self, params, R, mask, species, neighbor=None):
        # Update both neighbor lists
        nbrs_sr = self.neighbor_fn.update(R, self.nbrs_sr_init)
        nbrs_lr = self.neighbor_fn_lr.update(R, self.nbrs_lr_init)
        return self._energy_fn(R, neighbor=nbrs_sr.idx, neighbor_lr=nbrs_lr.idx)
```

**Problem**: CombinedModel/Trainer only pass a single `neighbor` object. We need to either:
1. Store both neighbor lists as state inside SO3LRModel (updated on each `compute_energy` call)
2. Pack both neighbor lists into a single container and unpack inside SO3LRModel

**Recommendation**: Option 1 (internal state management). The SO3LRModel handles its own neighbor lists, and the `neighbor` parameter from the framework is **ignored** or used only for the SR list.

### 4.2 Species Encoding

Our framework uses 0-indexed amino acid type IDs (0-19 for 20 AA types).
SO3LR expects **atomic numbers** (Z=1 for H, Z=6 for C, etc.).

For our 1-bead-per-residue CG models, there are no "atomic numbers" — we have CG bead types.

**Solutions**:
1. **Map AA types to dummy atomic numbers**: Create a mapping like {ALA: 6, VAL: 7, ...} using first 20 elements. SO3LR's pre-trained model expects real atomic numbers though, so this would require fine-tuning from scratch.
2. **Train from scratch**: Initialize SO3LR with random parameters and train on CG data. This bypasses the pre-trained weights entirely but uses the SO3LR architecture.
3. **Use SO3LR for all-atom and CG together**: Not applicable to our 1-bead scheme.

**Recommendation**: Option 2. We need to train from scratch for CG anyway, since the pre-trained model was trained on all-atom DFT data, not CG force-matched data. The species mapping just needs to be consistent.

### 4.3 Parameter Management

SO3LR loads pre-trained parameters from orbax checkpoints (`params.pkl`).
Our framework expects `initialize_params(rng_key)` to return fresh random parameters.

**For training from scratch**: We need SO3LR's initialization logic to create random params given a model config. The `So3lrPotential` uses `mlff` internally which has its own parameter initialization.

**Investigation needed**: How does SO3LR/mlff initialize parameters? Can we call the underlying model's `init` with a random key and get fresh parameters?

**Fallback**: If SO3LR doesn't support random initialization easily, we can:
1. Use the pre-trained params as starting point and fine-tune
2. Write custom initialization code for the SO3krates component

### 4.4 Energy Function Template for chemtrain

chemtrain's `ForceMatching` trainer expects:
```python
energy_fn_template(params) -> Callable[[R, neighbor, **kwargs], scalar_energy]
```

SO3LR's `to_jax_md()` returns an energy function that does NOT take explicit params (params are baked into the potential via closure).

**Solution**: We need to make SO3LR's energy function parameterizable. This likely means:
1. Using the raw `So3lrPotential` with explicit parameter passing
2. Or wrapping the mlff model's forward function directly

This is the **hardest part** of the integration and requires deeper investigation of the mlff library internals.

### 4.5 Long-Range Interactions in CG

SO3LR includes Coulomb and dispersion energy terms. For CG proteins:
- **Coulomb**: CG beads don't have well-defined charges. Could set to 0 (disable electrostatics) or assign per-residue charges.
- **Dispersion**: C6 coefficients are atom-type dependent. Not meaningful for CG beads.
- **ZBL repulsion**: Very short-range atomic repulsion. Not physically meaningful at CG resolution (CG distances are ~3.8 A between bonded beads).

**Recommendation**: For CG use, **disable all long-range and classical terms** and use only the SO3krates neural network component. This simplifies the integration significantly — we only need the SR graph and the neural network part.

---

## 5. Detailed Implementation Plan

### Phase 1: Investigation (Before Writing Code)

**1a. Understand mlff's parameter initialization**
- Read `/p/project1/cameo/schmidt36/so3lr/` → find how model params are created from scratch
- Check if `So3lrPotential` supports `init()` with a random key
- Check if the underlying SO3krates model has a standard Flax/Haiku init

**1b. Understand the energy function's parameter structure**
- What does SO3LR's param pytree look like?
- Can we pass params explicitly (like chemutils models) or are they baked in?
- Can we use `hk.transform` or similar to separate init from apply?

**1c. Test SO3LR on a toy system**
- Create a minimal test: initialize SO3LR, compute energy on a small molecule
- Verify the JAX-MD interface works
- Profile memory/compute cost relative to Allegro

### Phase 2: Core Wrapper

**2a. Create `models/so3lr_model.py`**

```python
class SO3LRModel:
    """
    Wrapper for SO3LR force field.

    Unlike AllegroModel/MACEModel (which use chemutils' hk.transform pattern),
    SO3LR uses the mlff library. This wrapper adapts it to match
    the same interface expected by CombinedModel.
    """

    def __init__(self, config, R0, box, species, N_max):
        # Build SO3LR potential (neural network component only for CG)
        # Setup dual neighbor lists
        # Store displacement function for prior compatibility

    def initialize_params(self, rng_key):
        # Initialize SO3krates parameters from random key
        # Return params pytree compatible with our framework

    def compute_energy(self, params, R, mask, species, neighbor=None):
        # Update SR neighbor list (and LR if needed)
        # Build Graph namedtuple from R, species, neighbor lists
        # Call SO3LR potential
        # Apply mask
        # Return scalar energy

    def compute_energy_and_forces(self, params, R, mask, species, neighbor=None):
        # Autodiff through compute_energy

    @property
    def initial_neighbors(self):
        return self.nbrs_sr_init  # Return SR neighbor list for framework compat

    @property
    def displacement(self):
        return self._displacement

    @property
    def nneigh_fn(self):
        return self.neighbor_fn  # SR neighbor function
```

**2b. Modify `combined_model.py`**

Add `"so3lr"` as a third option in `get_ml_model_type()`:

```python
if self.ml_model_type == "so3lr":
    self.ml_model = SO3LRModel(config, R0, box, species, N_max)
elif self.ml_model_type == "mace":
    self.ml_model = MACEModel(config, R0, box, species, N_max)
else:
    self.ml_model = AllegroModel(config, R0, box, species, N_max)
```

**2c. Add SO3LR config to `config/manager.py`**

```python
def get_so3lr_config(self) -> Dict[str, Any]:
    """Get SO3LR model configuration."""
    return self.get("model", "so3lr", default={})
```

### Phase 3: Config and Testing

**3a. Create `config_so3lr.yaml`**

```yaml
model:
  ml_model: "so3lr"
  cutoff: 4.5          # SO3krates SR cutoff
  dr_threshold: 0.5

  so3lr:
    num_layers: 3           # SO3krates transformer layers
    hidden_features: 128    # Feature dimension
    num_heads: 4            # Attention heads
    max_degree: 4           # Max spherical harmonic degree
    n_radial_basis: 32      # Bernstein basis functions
    use_lr: false           # Disable long-range for CG
    lr_cutoff: 0.0          # No LR cutoff when disabled
```

**3b. Integration test**
- Train on small 4zohB01 dataset
- Compare loss convergence with Allegro baseline

### Phase 4: Polish

**4a. Export support**
- Either extend AllegroExporter or create SO3LRExporter
- May not be MLIR-exportable (SO3LR uses different graph format than LAMMPS expects)
- Could export as pickle-only initially

**4b. Documentation**
- Update UPDATED_GENERAL_CONTEXT.md
- Add SO3LR section to COMMANDS.md

---

## 6. Dependency Management

SO3LR requires packages not currently in our environment:

```
mlff (from kabylda/mlff.git)
jraph
jax-pme
glp
orbax-checkpoint
```

**Installation**:
```bash
# From so3lr repo
pip install -e /p/project1/cameo/schmidt36/so3lr
```

This should pull in all dependencies. Need to verify compatibility with our existing JAX 0.4.34 environment (SO3LR requires JAX 0.5.3 — **potential version conflict!**).

**Critical**: SO3LR requires **JAX 0.5.3**, but our environment has **JAX 0.4.34**. This is a potential blocker. Options:
1. Upgrade JAX → risk breaking chemtrain/chemutils
2. Downgrade SO3LR's JAX requirement → may not work
3. Run SO3LR in a separate environment → defeats the purpose of integration

**This needs to be resolved before any code is written.**

---

## 7. Open Questions (Require Investigation)

1. **Can SO3LR's mlff model be initialized from random params?**
   - If not, training from scratch is much harder
   - Need to read mlff source code

2. **Can SO3LR's energy function accept explicit params?**
   - Required for chemtrain's `ForceMatching` trainer
   - If params are baked into closures, we need to restructure

3. **JAX version compatibility**
   - SO3LR needs JAX 0.5.3, we have JAX 0.4.34
   - Is there a compatible middle ground?

4. **Can we extract just the SO3krates component?**
   - For CG, we don't need Coulomb/dispersion/ZBL
   - Using only the neural net part would simplify everything

5. **Performance at CG resolution**
   - SO3krates was designed for atomistic resolution (cutoff 4.5 A)
   - CG systems have different length scales (3.8 A bonds, 10+ A interactions)
   - May need larger cutoff for the neural net, impacting performance

6. **Memory footprint**
   - SO3LR is a large model (pretrained on 3.8M structures)
   - Does it fit in GPU memory alongside priors for CG proteins?

---

## 8. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| JAX version conflict (0.4.34 vs 0.5.3) | **HIGH** | **HIGH** | Test compatibility; possibly pin intermediate version |
| Cannot separate params from model | HIGH | MEDIUM | Investigate mlff internals; may need to fork/patch |
| Training from scratch not supported | MEDIUM | MEDIUM | Fine-tune from pre-trained; or extract architecture only |
| Dual neighbor list overhead | LOW | HIGH | Acceptable; just need to manage internally |
| CG species mapping issues | LOW | LOW | Map AA types to arbitrary integers; train from scratch |
| MLIR export incompatible | MEDIUM | HIGH | Accept pickle-only export initially |

---

## 9. Recommended Approach Order

1. **First**: Resolve JAX version compatibility (blocker)
2. **Second**: Investigate mlff parameter initialization (feasibility check)
3. **Third**: Build minimal prototype — just get energy computation working on a single CG frame
4. **Fourth**: Integrate with CombinedModel and train on small dataset
5. **Fifth**: Handle edge cases (masking, export, multi-node)

---

## 10. Alternative: Extract SO3krates Architecture Only

If the full SO3LR integration proves too difficult due to dependency/API issues, an alternative is to **reimplement the SO3krates architecture** within our chemutils/haiku framework:

- SO3krates is a transformer over equivariant features
- Core components: attention + radial basis + spherical harmonics
- Could build it as a new `so3krates_neighborlist_pp()` following the chemutils pattern
- This would be more work upfront but give us full control

This would be a Phase 5 fallback if Phase 1-4 hit blockers.
