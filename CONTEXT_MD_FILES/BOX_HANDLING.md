# Box Handling in Chemtrain

**Date:** 2026-01-26
**Status:** Documentation

---

## Overview
The simulation box is **computed from data**, not read from config.

## Why?
- Ensures box fits all structures in dataset
- Prevents atoms outside box boundaries
- Automatically adapts to different systems
- No need to manually specify box dimensions for each dataset

## How It Works

### 1. Compute Extent
During data preprocessing, the box extent is computed from all frames:

```python
# In scripts/train.py
preprocessor = CoordinatePreprocessor(cutoff=cutoff)
extent, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
```

The extent is the maximum coordinate range across all frames, plus a buffer:
```python
# In data/preprocessor.py
R_max = jnp.max(R_real, axis=(0, 1))  # Max coordinates across all atoms and frames
extent = R_max * buffer_multiplier    # Default buffer_multiplier = 1.2
```

### 2. Center Coordinates
All coordinates are centered in the computed box:

```python
dataset["R"] = preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, R_shift)
```

This ensures:
- All real atoms are within the box
- Box is centered around the structure
- Neighbor lists work correctly

### 3. Park Masked Atoms
Padded (masked) atoms are moved far from real atoms:

```python
R_parked = jnp.where(
    mask[..., None],
    R_centered,
    extent * park_multiplier  # Default park_multiplier = 10.0
)
```

This prevents masked atoms from interfering with:
- Neighbor list calculations
- Energy computations
- Force predictions

## Config (Old - Removed)

Previously `system.box` existed in config:

```yaml
system:
  box: [406.13, 450.15002, 452.7]  # REMOVED - was never used!
```

**This has been removed** to avoid confusion. The box is always computed from data.

## Customization

To adjust box behavior, modify these config parameters:

```yaml
preprocessing:
  buffer_multiplier: 1.2  # How much padding around atoms (R_max * 1.2)
  park_multiplier: 10.0   # Where to place masked atoms (box * 10.0)
```

**Example:**
- `buffer_multiplier = 1.5` → More conservative, larger box
- `buffer_multiplier = 1.1` → Tighter box, may cause issues with large motions
- `park_multiplier = 5.0` → Park closer (may interfere with neighbor lists)
- `park_multiplier = 20.0` → Park farther away (safer but uses more memory)

## Technical Details

### Box Used by Different Components

| Component | Box Usage |
|-----------|-----------|
| **Data Preprocessing** | Computes extent from data |
| **Allegro Model** | Uses box for neighbor list initialization |
| **Neighbor Lists** | Uses box for periodic boundary conditions (if enabled) |
| **MLIR Export** | Exports box dimensions for LAMMPS |

### Box Computation Formula

```python
# For each dimension (x, y, z):
extent[i] = max(R[:, :, i]) * buffer_multiplier

# Total box:
box = [extent_x, extent_y, extent_z]
```

### Output Example

```
[Preprocessing] Computed box: [228.69  365.94  333.5 ]
[Preprocessing] R_shift: [  0.      0.      0.   ]
```

This box is then used throughout training and exported with the model.

---

## Related Files

- `data/preprocessor.py` - Box computation implementation
- `scripts/train.py` - Box computation call
- `config/manager.py` - Preprocessing configuration
- `config_template.yaml` - Preprocessing parameters

---

**Summary:** The box is **always computed from data**, ensuring it fits the dataset. The old `system.box` config parameter was never used and has been removed.
