#!/bin/bash

# Stages/2026 module environment for the Jupiter cameo_cg venv (Python 3.13).
#
# Key differences from load_modules.sh (Stages/2025):
#   - GCC 14.3.0, Python 3.13.5, CUDA/13
#   - The system JAX module is not used. Install CUDA-enabled JAX in the venv.
#     The validated August 2026 environment uses JAX/JAXlib 0.10.1.
#     (jax[cuda12] pip packages run fine on a CUDA 13 driver)
#   - NVSHMEM is not loaded here: all CUDA-13 NVSHMEM builds require OpenMPI.
#     Load it separately with OpenMPI/5.0.8 when compiling LAMMPS/chemtrain-deploy.
#   - LAMMPS / chemtrain-deploy compiled for CUDA 12 must be recompiled for CUDA 13.
#
# Usage:
#   source load_modules_2026.sh

module purge
module load Stages/2026 StdEnv/2026
module load GCC/14.3.0 Python/3.13.5
module load CUDA/13 ParaStationMPI cuDNN/9.19.0.56-CUDA-13 NCCL/default-CUDA-13

module load CMake/3.31.8
module load Ninja/1.13.0
module load Bazel/7.7.0-Java-21
module load Clang/20.1.8

module load UCX/default
module load UCC/default

module load git/2.50.1
module load HDF5/1.14.6-serial
module load tmux/3.5a

export CC=$(which gcc)
export CXX=$(which g++)
export CLANG_CUDA_COMPILER_PATH=$(which gcc)
