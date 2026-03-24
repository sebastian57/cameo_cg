Below is an instruction manual for a coding agent. It is intentionally focused on **what to measure, why it matters, and how to interpret it**, rather than prescribing code-level details. The goal is to diagnose **early plateauing in tiled training**, with particular attention to:

* loss of optimizer updates
* differences between vmapped small-structure batching and tiled batching
* shuffle quality
* effective batch semantics
* whether the current tiling implementation changes optimization conditions in a nontrivial way

---

# Instruction Manual for Coding Agent: Diagnose Early Plateauing in Tiled Training

## Primary objective

Instrument and profile a short training run over a few epochs to determine **why tiled training plateaus earlier or converges more slowly than the original vmapped small-structure batching**.

We currently suspect one or more of the following:

1. **Loss of optimizer updates**

   * tiled training may process much more data per step, causing fewer optimizer updates per amount of data seen

2. **Changed optimization regime**

   * processing many small structures independently with `vmap` is not necessarily equivalent, from an optimizer perspective, to packing those structures into tiles and processing them as larger disconnected compute objects

3. **Shuffling issues**

   * tiles may not be shuffled correctly, or structures may be grouped in a way that reduces gradient diversity

4. **Changed loss weighting / effective batch semantics**

   * tiled batching may alter structure-level weighting, bead-level weighting, or tile-level averaging

5. **Implementation-specific tiling effects**

   * the current tile construction or tile ordering may be creating optimizer conditions that differ substantially from the old regime

The goal of this profiling pass is **not to optimize yet**, but to build a high-confidence diagnosis.

---

# What this investigation should answer

By the end of the profiling run, we want to be able to answer these questions clearly:

## A. Optimizer update accounting

* How many optimizer steps do we perform per epoch?
* How many optimizer steps do we perform per fixed amount of data?
* How many structures, beads, and force components contribute to each optimizer step?
* How does this compare to the old vmapped regime?

## B. Tile production and tile usage

* How many tiles are produced from the dataset?
* What is the distribution of structures per tile?
* What is the distribution of valid beads per tile?
* What is the tile fill ratio?
* How many tiles are seen per epoch and per device?

## C. Shuffle behavior

* Are structures shuffled before tile construction?
* Are tiles shuffled after construction?
* Are tile compositions changing across epochs, or are we reusing nearly identical structure groupings?
* Does each device see a representative sample of tiles?

## D. Objective / weighting differences

* Does tiled training preserve the same effective loss weighting as the old regime?
* Are we averaging per tile, per structure, per bead, or per force component?
* Are larger structures contributing disproportionately in tiled mode?

## E. Gradient / optimization dynamics

* Are gradients smoother because the effective batch is much larger?
* Are updates smaller relative to the batch size?
* Is plateauing due to too few optimizer steps, too little gradient diversity, or a changed objective?

## F. Tiling-specific effects

* Does tiling itself, even when mathematically correct, create a different optimization regime than vmapped independent structures?
* Is there evidence that plateauing is due to tile construction choices rather than just larger batch size?

---

# Conceptual background the agent should use

## 1. Same model math does not imply same optimizer behavior

Even if tiled inference is mathematically equivalent to separate inference on disconnected structures, the **optimizer conditions** may still differ because of:

* different reduction order
* different weighting of structures and beads
* fewer explicit batch items
* fewer optimizer updates per epoch
* different gradient diversity

So the diagnosis must distinguish between:

* **prediction equivalence**
* **training-equivalence**

These are not the same thing.

## 2. `grad_accumulation_steps = 1` does not mean “same effective batch”

Two runs can both use:

* `batch_per_device = 1`
* `gradient_accumulation_steps = 1`

and still be very different if one “batch item” is:

* one small structure
  versus
* one large tile containing many structures

The correct comparison must therefore track:

* structures per optimizer step
* valid beads per optimizer step
* force components per optimizer step
* independent structure count per step
* tile count per step

## 3. Plateauing from large batches often comes from too few effective updates

If each optimizer step sees a large fraction of the dataset, then:

* gradients are smoother
* optimizer steps are fewer per epoch
* exploration is reduced
* plateauing can happen earlier

Therefore, a key diagnostic is to compare training not only by:

* epoch
  or
* optimizer step

but also by:

* structures seen
* valid beads seen
* force components seen
* wall-clock time

## 4. Shuffle quality matters more in tiled mode

In the original vmapped regime, each step may naturally average over many independent small structures. In tiled mode, if tile construction is too deterministic or insufficiently shuffled, then:

* structure co-occurrence patterns may repeat too often
* gradient diversity may drop
* devices may receive correlated tiles
* convergence can suffer even if tile math is correct

## 5. Loss normalization must be made explicit

A major source of confusion in tiled training is that the effective objective can change silently.

We need to know whether the current system is effectively optimizing:

* per-tile mean loss
* per-structure mean loss
* per-bead mean loss
* per-force-component mean loss

This may explain both:

* different convergence speed
* different final loss values

---

# Required profiling scope

Run a **short, fully instrumented training job** over a few epochs in both regimes:

* **baseline regime**: original vmapped small-structure batching
* **tiled regime**: current tiled batching

The purpose is not long-run quality, but **fine-grained measurement of training dynamics**.

Use the same:

* model initialization or seed if feasible
* dataset split
* optimizer
* LR schedule
* number of epochs
* logging cadence

---

# Diagnostics checklist

## 1. Dataset-to-step accounting

Instrument the system so we can answer, per epoch and per step:

### Per optimizer step

* number of structures contributing
* number of valid beads contributing
* number of valid force components contributing
* number of tiles contributing
* number of devices contributing
* number of structures per device
* number of tiles per device

### Per epoch

* total structures seen
* total valid beads seen
* total valid force components seen
* total tiles consumed
* total optimizer steps
* total forward passes
* total backward passes

### Interpretation

This tells us whether plateauing could be explained by:

* far fewer optimizer steps per data volume
* much larger effective batch size
* changed dataset coverage per step

---

## 2. Tile production diagnostics

Measure the properties of the tiled dataset itself.

### Per tile

* tile ID
* number of structures in tile
* number of valid beads in tile
* tile capacity
* tile fill ratio = used beads / capacity
* min / mean / max structure size inside tile
* structure IDs included in the tile

### Over the whole tiled dataset

* total number of tiles
* histogram of structures per tile
* histogram of valid beads per tile
* histogram of fill ratios
* histogram of structure sizes inside tiles
* fraction of underfilled tiles

### Interpretation

This tells us:

* how aggressively the dataset is being compressed into tiles
* whether tile underfilling is a performance issue
* how much information each tile carries
* whether tile composition is highly variable or highly regular

---

## 3. Shuffle diagnostics

We specifically want to know whether we are shuffling correctly.

### Questions to answer

* Are individual structures shuffled before tile packing?
* Are tiles shuffled after packing?
* Are structure groupings re-sampled each epoch or reused?
* Does tile composition stay fixed across epochs?
* Are devices assigned contiguous or stratified chunks of tiles?
* Are some devices consistently seeing similar tile distributions?

### Required diagnostics

For a few epochs, log:

* tile ID order per epoch
* structure IDs within each tile
* epoch-to-epoch overlap of tile compositions
* per-device tile assignment summary
* per-device distribution of:

  * tile size
  * structure count
  * mean structure size

### Optional but useful summary metrics

* Jaccard overlap between tile structure-sets across consecutive epochs
* entropy / diversity measure of structure co-occurrence
* fraction of structures whose tile-neighbors change from epoch to epoch

### Interpretation

If tile compositions are too static or too correlated, plateauing may partly come from reduced stochasticity and diversity.

---

## 4. Loss normalization / weighting diagnostics

This is one of the most important checks.

For the same training data, compute and log multiple views of the loss:

### Candidate normalizations

* per tile
* per structure
* per bead
* per force component

### Goal

Determine which weighting the current implementation actually uses in:

* baseline mode
* tiled mode

### Required outputs

For each logged step:

* reported training loss
* manually recomputed per-tile loss
* manually recomputed per-structure mean loss
* manually recomputed per-bead mean loss
* manually recomputed per-component mean loss

### Interpretation

This tells us whether the tiled regime is optimizing a different effective objective.

This is especially important because a different loss weighting can explain:

* slower convergence
* different final loss values
* apparently “more stable” or “less stable” curves

---

## 5. Gradient and update diagnostics

Measure whether tiled training changes the optimizer input significantly.

### Per optimizer step

* gradient norm
* update norm, if easily available
* learning rate
* ratio of update norm to parameter norm, if available
* moving-window variance of gradient norm
* optionally cosine similarity between consecutive gradients

### Interpretation

These diagnostics help answer:

* are tiled gradients lower variance?
* are updates too conservative?
* is the optimizer receiving less diverse gradients?
* is plateauing consistent with a large-batch low-noise regime?

---

## 6. Timing / throughput diagnostics

We also want to profile the execution regime.

### Per step

* step wall time
* data loading / tile assembly time
* forward time
* backward time
* optimizer step time

### Per epoch

* total epoch time
* optimizer steps per second
* structures per second
* valid beads per second
* tiles per second

### Interpretation

This lets us relate plateauing to:

* fewer optimizer updates per wall-clock
* more data processed per update
* overall training efficiency

---

## 7. Cross-regime comparison metrics

For both baseline and tiled runs, compare training progress against several axes:

* loss vs optimizer step
* loss vs epoch
* loss vs structures seen
* loss vs valid beads seen
* loss vs wall-clock time

### Interpretation

This is essential.

Plateauing that appears severe in:

* loss vs optimizer step

may be much less severe in:

* loss vs wall-clock
  or
* loss vs valid beads seen

This helps separate:

* true optimization degradation
  from
* simple large-batch scaling effects

---

# Additional diagnostics to add to the checklist

These are the extra diagnostics most worth adding.

## A. Effective independent-example count per step

In the old regime, the optimizer may see many explicitly independent structures. In tiled mode, even if the number of structures is large, the grouping into tiles may change how the loss is averaged.

Add a derived metric:

* effective independent structures per optimizer step

This should reflect:

* total structures contributing
* how they are grouped into tiles
* how loss is reduced afterward

This is not a direct code primitive; it is an analytical summary to help compare old vs tiled optimization conditions.

## B. Tile-coherence diagnostics

Check whether tiles are accidentally too homogeneous.

For each tile, summarize:

* mean structure length
* variance of structure length
* optionally composition stats if relevant (species patterns, chain class, etc.)

Then compare across tiles and epochs.

If tiles are too homogeneous or too repetitive, optimization may become less exploratory.

## C. Per-device imbalance diagnostics

Since you are using 4 devices and `batch_per_device = 1`, each device sees one tile at a time.

Therefore, device-level imbalance matters more than usual.

Check:

* whether some devices consistently receive denser or larger tiles
* whether tile distributions differ by device
* whether per-device losses differ systematically

## D. Last / incomplete batch effects

Check whether the final steps of each epoch have unusual tile counts, fill ratios, or update semantics.

This can matter when batch construction is uneven.

## E. Correlation between fill ratio and loss / gradient norm

Add a simple analysis:

* do underfilled tiles correlate with different loss scales?
* do dense tiles produce systematically larger or smaller gradients?

This helps distinguish pure optimization issues from packing-quality issues.

---

# Suggested profiling questions the agent must answer

At the end of the instrumented run, provide concise answers to these:

1. How many optimizer steps per epoch do we get in baseline vs tiled mode?
2. How many structures, valid beads, and force components contribute to one optimizer step in each mode?
3. How many tiles are produced from the tiled dataset, and what are their fill ratios?
4. Are tiles reshuffled each epoch, or is tile composition effectively fixed?
5. Are structures shuffled before tiling, after tiling, or both?
6. Do devices see similar tile distributions?
7. Is the tiled loss effectively weighted per tile, per structure, per bead, or per component?
8. Does tiled mode process a much larger fraction of the dataset per optimizer step than baseline?
9. Is plateauing better explained by:

   * fewer optimizer steps,
   * changed loss weighting,
   * reduced shuffle diversity,
   * device-level imbalance,
   * or some combination?
10. Are there any signs that the current tile implementation changes optimizer conditions in a deeper way than just “larger batch”?

---

# Deliverables expected from the coding agent

## 1. Instrumented short-run profile

A few-epoch profile for both baseline and tiled modes with the diagnostics above.

## 2. Summary report

A concise report containing:

* key counts
* key histograms
* key comparisons
* likely causes of plateauing

## 3. Explicit diagnosis

A ranked list of likely causes, for example:

1. too few optimizer steps per epoch
2. loss weighting changed from per-structure to per-component
3. tile composition insufficiently reshuffled
4. per-device tile imbalance
5. tile underfill not significant

## 4. Recommendation-ready metrics

The report should provide enough evidence to decide later on:

* whether to change shuffle policy
* whether to change tile construction
* whether to change batch size in tiles
* whether to change loss normalization
* whether to retune the schedule

---

# Important framing for the agent

Do not assume the plateau is caused by a bug.
It may be a natural consequence of moving from:

* many explicit small independent batch items

to

* fewer, larger, more information-dense tiled batch items

The purpose of this profiling pass is to determine whether the plateau is primarily caused by:

* **loss of optimizer step frequency**
* **changed effective loss weighting**
* **reduced stochasticity / shuffle diversity**
* **specific implementation details of tiling**

Only after that should we move on to optimization changes.

---

# Final concise mission statement

Profile a short training run in both baseline and tiled modes to determine:

* how tiled batching changes optimizer-step frequency
* how much data each optimizer step contains
* how tiles are formed, filled, shuffled, and distributed across devices
* how the effective loss weighting differs from the original regime
* whether plateauing is due to fewer updates, altered weighting, poor shuffle diversity, or a tiling-specific implementation effect

If you want, I can also turn this into a tighter “agent task brief” with headings like **Scope**, **Required instrumentation**, **Questions to answer**, and **Output format**.

