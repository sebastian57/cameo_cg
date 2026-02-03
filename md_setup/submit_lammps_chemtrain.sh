#!/bin/bash -x

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4


source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/clean_booster_env
source /p/project1/cameo/schmidt36/set_lammps_paths.sh


export CUDA_VISIBLE_DEVICES=0,1,2,3

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

export TF_CPP_MIN_LOG_LEVEL=2
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

srun /p/project1/cameo/schmidt36/lammps/build/lmp -in inp_lammps_trained.in

