module purge
module load Stages/2025 StdEnv/2025
module load GCC/13.3.0 Python/3.12.3
module load CUDA/12 ParaStationMPI cuDNN/9.5.0.50-CUDA-12 NCCL/default-CUDA-12

module load jax/0.4.34-CUDA-12  

module load CMake/3.30.3      
module load Ninja/1.12.1      

module load Clang/18.1.8

module load UCX/default
module load UCC/default

module load git/2.45.1        
module load HDF5/1.14.5-serial

module load tmux

export CC=$(which gcc)
export CXX=$(which g++)
export CLANG_CUDA_COMPILER_PATH=$(which gcc)

