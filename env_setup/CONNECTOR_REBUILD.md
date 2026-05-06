# Connector Rebuild Guide

This document records the full working rebuild and relinking procedure for the
modernized `chemtrain-deploy` stack used with:

- `jax==0.9.1`
- `jaxlib==0.9.1`
- `cuequivariance-jax==0.9.0`

Status:

- connector build works
- GPU PJRT plugin build works
- LAMMPS plugin build works
- MD runtime is validated on JUWELS Booster

Primary checkout:

- `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy`

## Core Principle

The deployment stack must be built and run as one generation.

Do not mix:

- an older connector framework
- a newer wheel-provided PJRT plugin
- a model exported from a newer JAX/XLA stack

The supported workflow is:

1. Use an environment matching training and export.
2. Build `libconnector.so` from this checkout.
3. Build the GPU PJRT plugin from this same checkout.
4. Build the LAMMPS plugin against the rebuilt connector.
5. Run with a runtime wrapper that exposes exactly one PJRT plugin and the
   required hermetic shared-library paths.

## Why The Old Path Failed

The previous deployment flow failed for several independent reasons:

- PJRT API mismatch between connector framework and copied wheel plugin
- moved XLA / StableHLO headers and Bazel labels after upgrading to JAX `0.9.x`
- newer PJRT API signatures in the connector code
- Clang / toolchain conflicts on JUWELS Booster
- runtime dependency lookup failures for the source-built plugin
- cuDNN version mismatch between compile-time and runtime

## One-Time Modernization Work Already Done

This checkout has already been ported to the modern stack. Future rebuilds
should be procedural rather than investigative.

### Build-system changes

These were updated:

- `build.py`
- `WORKSPACE`
- `MODULE.bazel`
- `jax.bazelrc`
- `third_party/xla/workspace.bzl`
- `third_party/xla/revision.bzl`

Important build-system changes:

- Bazel upgraded to `7.4.1`
- `build.py` now supports build profiles
- `HERMETIC_PYTHON_VERSION` defaults to the active interpreter version
- source-built GPU PJRT plugin is the supported path
- wheel-plugin loading remains only as a debugging fallback

### Connector code changes

The connector sources were updated for modern XLA / PJRT layout and APIs.
Examples include:

- moved HLO / StableHLO include paths
- moved Bazel labels in `connector/BUILD`
- new `BufferFromHostBuffer` signature
- `CompileAndLoad(...)` in the runner instead of the old `Compile(...)` path
- new CPU client entrypoint
- removal of the old `ExecuteOptions::untuple_result`
- safer plugin discovery using `JCN_PJRT_PLUGIN` first, instead of relying on
  the earlier `std::filesystem` startup scan

## Build Profiles

The key improvement is `build.py --build_profile ...`.

### Generic profile

Use `generic` when:

- you are on a different machine
- you need to pass explicit CUDA, cuDNN, or compute-capability values
- the JUWELS defaults are not correct for your target system

### JUWELS Booster profile

`juwels-booster` is the working profile for the validated JUWELS deployment.
It currently applies these defaults in `build.py`:

- `use_clang = False`
- `cuda_version = 12.9.1`
- `cudnn_version = 9.5.0`
- `cuda_compute_capabilities = sm_80`

Why those values were necessary:

- `use_clang = False`
  Avoids the duplicate `crosstool` module-map conflict seen with the hermetic
  Clang path during connector builds.
- `cuda_version = 12.9.1`
  Matches the upgraded JAX/XLA CUDA generation used by this checkout.
- `cudnn_version = 9.5.0`
  Matches the actual JUWELS runtime cuDNN seen by LAMMPS. This was critical.
- `cuda_compute_capabilities = sm_80`
  Matches A100 GPUs and avoids placeholder architectures such as `compute_120`
  that older NVCC releases cannot compile.

### Jupiter Booster profile

`jupiter-booster` is the working profile for the GH200/Hopper deployment on
Jupiter Booster. It uses the same connector/toolchain defaults as
`juwels-booster`, but targets the GH200 GPU architecture:

- `use_clang = False`
- `cuda_version = 12.9.1`
- `cudnn_version = 9.5.0`
- `cuda_compute_capabilities = sm_90`

Use this profile on nodes where `nvidia-smi --query-gpu=compute_cap` reports
`9.0`.

## Full Working Rebuild Procedure on JUWELS Booster

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

Expected artifact:

- `lib/pjrt_plugin.xla_cuda12.so`

Notes:

- this plugin must come from the same checkout as the connector
- after the final validation, the JUWELS profile defaults to `cudnn_version = 9.5.0`

### 4. Build the connector library

```bash
python build.py --build_profile juwels-booster
```

Expected artifact:

- `lib/libconnector.so`

### 5. Build or relink the LAMMPS plugin

```bash
mkdir -p build
cd build
cmake -D LAMMPS_HEADER_DIR=/p/project1/cameo/schmidt36/lammps/src ../lammps_plugin
cmake --build . --clean-first
```

Expected artifact:

- `build/chemtrain_deployplugin.so`

## Expected Artifacts

From the checkout root:

- `lib/libconnector.so`
- `lib/pjrt_plugin.xla_cuda12.so`
- `build/chemtrain_deployplugin.so`

On the current machine these are:

- `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/lib/libconnector.so`
- `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/lib/pjrt_plugin.xla_cuda12.so`
- `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/build/chemtrain_deployplugin.so`

## Runtime Wrapper Requirements

Runtime setup is not just a convenience layer. It is part of the working
configuration.

The current wrapper is:

- `/p/project1/cameo/schmidt36/set_lammps_paths.sh`

It does four important things:

1. exposes the LAMMPS plugin directory
2. points `JCN_PJRT_PLUGIN` at the exact plugin file to load
3. creates `JCN_PJRT_PATH` as a single-plugin directory to avoid stale or
   duplicate plugin loads
4. adds the hermetic Bazel `_solib_x86_64` subdirectories and NVSHMEM runtime
   directory to `LD_LIBRARY_PATH`

That fourth point matters. Without it, the source-built PJRT plugin can fail to
load dependencies such as:

- `libnvshmem_host.so.3`
- `nvshmem_bootstrap_uid.so.3`
- `nvshmem_transport_ibrc.so.3`
- CUDA / NCCL / CUPTI / cuDNN shared objects coming from Bazel’s hermetic build

## Launch Sequence

The validated launch flow is:

```bash
source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/env_cueq_allegro_opt/bin/activate
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

sbatch /p/project1/cameo/schmidt36/cameo_cg/md_setup/submit_lammps_chemtrain.sh
```

## Minimal JUWELS Checklist

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

## Rebuilding On A New Machine

The process is portable, but the build profile should be treated as machine
configuration, not universal truth.

### Rule 1: Match the model stack

The deployment environment should match the model export generation.
For this project that means staying on the `jax 0.9.1` generation unless the
whole connector stack is upgraded again.

### Rule 2: Build connector and PJRT plugin from the same checkout

Never treat a wheel-provided plugin as the normal deployment path.

### Rule 3: Create a machine-specific profile if the toolchain differs

If the new machine differs materially, add a new profile to `build.py` instead
of memorizing a long one-off command.

A good machine profile should encode:

- whether `use_clang` should be on or off
- CUDA toolkit version
- cuDNN version
- compute capabilities
- any stable host-compiler behavior

### Recommended template for a new profile

Start from `juwels-booster` and change only what is machine-specific.

At minimum validate:

- actual GPU architecture
- actual cuDNN runtime version available in jobs
- whether the system provides CUDA / NCCL / CUPTI / NVSHMEM itself, or whether
  you need hermetic runtime paths exposed similarly to JUWELS
- whether the Clang path is stable or causes module-map conflicts

### Generic command pattern

If you do not yet have a dedicated profile:

```bash
python build.py \
  --build_profile generic \
  --build_gpu_pjrt_plugin \
  --enable_cuda \
  --cuda_version <cuda-version> \
  --cudnn_version <cudnn-version> \
  --cuda_compute_capabilities <capabilities>

python build.py --build_profile generic
```

Then rebuild the LAMMPS plugin and validate runtime loading.

## Common Failure Modes

### PJRT API mismatch

Example symptom:

- `Unexpected PJRT_Client_Create_Args size`

Meaning:

- connector framework and PJRT plugin came from different generations

Fix:

- rebuild the GPU PJRT plugin from the same checkout as the connector

### Startup crash during plugin discovery

Meaning:

- the old plugin-discovery path was not robust enough in the LAMMPS runtime

Fix:

- prefer `JCN_PJRT_PLUGIN`
- keep `JCN_PJRT_PATH` as a single-plugin fallback directory

### Missing shared libraries at plugin load time

Example symptom:

- `Failed to open ... pjrt_plugin.xla_cuda12.so: libnvshmem_host.so.3: cannot open shared object file`

Meaning:

- the source-built PJRT plugin’s runtime dependencies are not visible

Fix:

- expose Bazel `_solib_x86_64` subdirectories and the NVSHMEM lib directory in
  `LD_LIBRARY_PATH`
- use the wrapper script, do not hand-roll the runtime environment

### Unsupported CUDA architecture during build

Example symptom:

- `Unsupported gpu architecture 'compute_120'`

Fix:

- set `cuda_compute_capabilities` explicitly
- on JUWELS Booster use `sm_80`
- on Jupiter Booster / GH200 use `sm_90`

### cuDNN compile-time vs runtime mismatch

Example symptom:

- `Loaded runtime CuDNN library: 9.5.0 but source was compiled with: 9.8.0`

Meaning:

- the PJRT plugin was built against a different cuDNN version than the one the
  job actually loads

Fix:

- rebuild the GPU PJRT plugin with a profile that matches the machine’s real
  runtime cuDNN version
- on JUWELS Booster the working value is `9.5.0`

### Clang / module-map conflict

Example symptom:

- duplicate `crosstool` module definition

Fix:

- disable the Clang host path in that machine profile

## Files To Inspect First If A New Machine Fails

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

## Recommended Long-Term Policy

1. Keep training and deployment on the same JAX generation.
2. Rebuild the connector when the XLA / JAX side changes materially.
3. Rebuild the GPU PJRT plugin from source from the same checkout.
4. Rebuild the LAMMPS plugin after connector-side changes.
5. Keep machine-specific behavior in build profiles and runtime wrappers.
6. Treat runtime-library exposure as part of deployment, not as an afterthought.

## Final Summary

The working deployment recipe is now fully validated.

The two key ideas to preserve are:

- source-build the connector and the GPU PJRT plugin from the same checkout
- encode machine-specific behavior in a build profile plus a runtime wrapper

That combination is what made the new JAX `0.9.1` cuequivariance / Allegro
stack run successfully inside LAMMPS.
