# Connector Rebuild Guide

This document captures the full rebuild process for the modernized `chemtrain-deploy` connector stack that matches the training/export environment in `env_cueq_allegro_opt`.

The goal is to keep all deployment pieces on the same JAX/XLA/PJRT generation:

- `jax==0.9.1`
- `jaxlib==0.9.1`
- `cuequivariance-jax==0.9.0`

This guide reflects the connector checkout at:

- `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy`

## Why This Rebuild Process Exists

The old deployment flow mixed components from different generations:

- connector framework built from an older vendored XLA snapshot
- wheel-provided CUDA PJRT plugin copied in from a newer JAX install
- models exported from a much newer JAX stack

That can fail in several ways:

- PJRT API mismatch
- MLIR / StableHLO / export format mismatch
- CUDA build-toolchain mismatch
- runtime plugin loading conflicts

The modernized process avoids that by building the connector and the GPU PJRT plugin from the same checkout.

## Supported Deployment Contract

The supported deployment flow is now:

1. Activate an environment matching the training/export stack.
2. Build `libconnector.so` from this checkout.
3. Build the GPU PJRT plugin from this same checkout.
4. Build the LAMMPS plugin against this connector.
5. Expose exactly one PJRT plugin at runtime.

The main rule is:

> Do not use a wheel-provided PJRT plugin as the normal deployment path.

That fallback still exists only for debugging.

## One-Time Modernization Changes

The following pieces were updated to make this checkout work with the modern stack.

### Build system changes

- `build.py` was extended with build profiles.
- Bazel was upgraded to `7.4.1`.
- `HERMETIC_PYTHON_VERSION` now defaults to the active interpreter version if unset.
- The source-built GPU PJRT plugin is now the intended path.
- The `juwels-booster` profile was added to encode stable local defaults.

### XLA / JAX scaffold changes

These files were moved toward the `jax 0.9.1` generation:

- `MODULE.bazel`
- `jax.bazelrc`
- `WORKSPACE`
- `third_party/xla/workspace.bzl`
- `third_party/xla/revision.bzl`

### Connector porting changes

The connector code had to be adapted for the newer XLA/PJRT layout and APIs.

Examples:

- moved XLA Bazel labels
- moved HLO / StableHLO include paths
- new `BufferFromHostBuffer` signature
- `CompileAndLoad(...)` instead of the old `Compile(...)` path in the runner
- new CPU client entrypoint
- removal of the old `ExecuteOptions::untuple_result` field

## Build Profiles

The rebuild process is centered around `build.py --build_profile ...`.

### Generic profile

`generic` keeps behavior close to the upstream-oriented defaults.

Use it when:

- you are on a different machine
- you want to pass explicit CUDA / cuDNN / compiler settings yourself
- the JUWELS defaults are not appropriate

### JUWELS Booster profile

`juwels-booster` is the stable, machine-specific profile that worked for this deployment stack.

It currently applies these defaults in `build.py`:

- `use_clang = False`
- `cuda_version = 12.9.1`
- `cudnn_version = 9.8.0`
- `cuda_compute_capabilities = sm_80`

Why those values matter:

- `use_clang = False`
  Avoids the duplicate `crosstool` module-map conflict seen with the hermetic Clang path during connector builds.
- `cuda_version = 12.9.1`
  Matches the newer JAX/XLA generation better than the older local CUDA baseline.
- `cudnn_version = 9.8.0`
  Keeps the hermetic CUDA stack aligned with the upgraded XLA side.
- `cuda_compute_capabilities = sm_80`
  Matches JUWELS Booster A100 GPUs and avoids unsupported placeholder architectures such as `compute_120`.

## Rebuild Sequence on JUWELS Booster

### 1. Activate the environment

```bash
source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/env_cueq_allegro_opt/bin/activate
```

### 2. Enter the connector checkout

```bash
cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy
```

### 3. Build the GPU PJRT plugin

```bash
python build.py --build_profile juwels-booster --build_gpu_pjrt_plugin --enable_cuda
```

This should produce the source-built GPU PJRT plugin that matches the checkout.

Expected artifact:

- `lib/pjrt_plugin.xla_cuda12.so`

### 4. Build the connector library

```bash
python build.py --build_profile juwels-booster
```

Expected artifact:

- `lib/libconnector.so`

### 5. Rebuild the LAMMPS plugin

```bash
mkdir -p build
cd build
cmake -D LAMMPS_HEADER_DIR=/p/project1/cameo/schmidt36/lammps/src ../lammps_plugin
cmake --build . --clean-first
```

Expected artifact:

- `build/chemtrain_deployplugin.so`

## Expected Artifacts After a Successful Rebuild

From the checkout root:

- `lib/libconnector.so`
- `lib/pjrt_plugin.xla_cuda12.so`
- `build/chemtrain_deployplugin.so`

On the current machine these resolve to:

- `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/lib/libconnector.so`
- `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/lib/pjrt_plugin.xla_cuda12.so`
- `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/build/chemtrain_deployplugin.so`

## Runtime Environment

At runtime, these environment variables must be set consistently:

- `PATH` includes the desired LAMMPS build
- `LAMMPS_PLUGIN_PATH` points to the connector `build/` directory
- `JCN_LIB_PATH` points to the connector `lib/` directory
- `JCN_PJRT_PATH` points to a directory containing exactly one PJRT plugin

The helper script already does this:

- `/p/project1/cameo/schmidt36/set_lammps_paths.sh`

It also creates a single-plugin directory:

- `lib/pjrt_single/`

That is important because it prevents the connector from loading stale backup `.so` files or multiple PJRT plugins at once.

## Launch Sequence

```bash
source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/env_cueq_allegro_opt/bin/activate
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

sbatch /p/project1/cameo/schmidt36/cameo_cg/md_setup/submit_lammps_chemtrain.sh
```

## Portable Rebuild Recipe for Other Machines

The process is portable if you keep the same structure.

### Rule 1: Match the training/export stack

The deployment environment should match the model-export generation.

For this project, that means staying on the `jax 0.9.1` generation.

### Rule 2: Build connector and PJRT plugin from the same checkout

Never rely on a newer wheel plugin inside an older connector framework as the normal path.

### Rule 3: Prefer machine-specific build profiles

Instead of remembering a long custom command on each machine, add a new build profile in `build.py`.

A good profile should encode:

- whether to use clang or not
- CUDA version
- cuDNN version
- compute capabilities
- any stable local compiler behavior

### Suggested pattern

- `generic`
  Upstream-oriented defaults.
- `juwels-booster`
  A100 / NVCC / modern CUDA defaults for JUWELS.
- future profile names
  Add one profile per cluster or machine family if the toolchain is meaningfully different.

## Generic Build Commands

If you are on another system and do not have a dedicated profile yet, use explicit flags.

### GPU PJRT plugin

```bash
python build.py \
  --build_profile generic \
  --build_gpu_pjrt_plugin \
  --enable_cuda \
  --cuda_version <cuda-version> \
  --cudnn_version <cudnn-version> \
  --cuda_compute_capabilities <capabilities>
```

### Connector

```bash
python build.py --build_profile generic
```

If the machine is sensitive to Clang vs NVCC interactions, also decide whether `--use_clang` should remain enabled.

## Common Failure Modes and What They Mean

### 1. PJRT API mismatch at runtime

Example symptom:

- `Unexpected PJRT_Client_Create_Args size`

Meaning:

- the connector framework and PJRT plugin came from different PJRT generations

Fix:

- rebuild the GPU PJRT plugin from the same checkout as the connector
- do not copy a wheel plugin into `lib/` as the main path

### 2. Unsupported CUDA architecture during plugin build

Example symptom:

- `Unsupported gpu architecture 'compute_120'`

Meaning:

- upstream defaults included a future architecture that the local NVCC cannot build

Fix:

- set `cuda_compute_capabilities` explicitly
- on JUWELS Booster use `sm_80`

### 3. CUPTI / CUDA profiler symbol mismatch

Example symptom:

- missing `CUPTI_PROFILER_PM_SAMPLING`

Meaning:

- XLA source expected a newer CUDA/CUPTI stack than the hermetic version being used

Fix:

- keep CUDA and cuDNN aligned with the upgraded JAX/XLA generation

### 4. Clang / toolchain module-map conflicts

Example symptom:

- duplicate `crosstool` module definition

Meaning:

- the local build pulled in conflicting toolchain module maps

Fix:

- disable clang host mode for that machine profile
- on JUWELS Booster the stable choice was `use_clang = False`

### 5. Missing XLA headers or moved Bazel targets

Meaning:

- the connector code or BUILD file still references old XLA paths

Fix:

- port the connector includes and Bazel labels to the new XLA layout

## Files Most Relevant for Future Maintenance

If a rebuild breaks on another machine, these are the first files to inspect:

- `build.py`
- `WORKSPACE`
- `jax.bazelrc`
- `MODULE.bazel`
- `third_party/xla/workspace.bzl`
- `third_party/xla/revision.bzl`
- `connector/BUILD`
- `connector/runner.cpp`
- `connector/buffer.cpp`
- `connector/compiler.cpp`
- `connector/xla_call_module_loader.cpp`
- `/p/project1/cameo/schmidt36/set_lammps_paths.sh`

## Recommended Rebuild Policy

For future work, the cleanest policy is:

1. Keep training and deployment on the same JAX generation.
2. Rebuild the connector whenever the XLA / JAX side changes materially.
3. Rebuild the GPU PJRT plugin from source from the same checkout.
4. Rebuild the LAMMPS plugin after connector-side changes.
5. Keep machine-specific logic in build profiles and runtime wrappers.

## Minimal JUWELS Booster Checklist

```bash
source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/env_cueq_allegro_opt/bin/activate

cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy
python build.py --build_profile juwels-booster --build_gpu_pjrt_plugin --enable_cuda
python build.py --build_profile juwels-booster

cd build
cmake -D LAMMPS_HEADER_DIR=/p/project1/cameo/schmidt36/lammps/src ../lammps_plugin
cmake --build . --clean-first

source /p/project1/cameo/schmidt36/set_lammps_paths.sh
sbatch /p/project1/cameo/schmidt36/cameo_cg/md_setup/submit_lammps_chemtrain.sh
```

## Final Notes

The build profile approach is the main improvement to preserve.

It keeps the rebuild process reproducible, readable, and cluster-specific without scattering one-off flags through shell history. If this connector needs to live on multiple clusters, extending the profile mechanism is the right long-term pattern.
