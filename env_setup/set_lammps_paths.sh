#! /bin/bash

export PATH=/p/project1/cameo/schmidt36/lammps/build:$PATH
export LAMMPS_PLUGIN_PATH=/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/build

export JCN_PJRT_PATH=/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/lib
export LD_LIBRARY_PATH=$JCN_PJRT_PATH:$LD_LIBRARY_PATH

#export PJRT_PLUGIN_LIBRARY_PATH=$JCN_PJRT_PATH/pjrt_plugin.xla_cuda12.so
#export LD_LIBRARY_PATH=$(python3 -c "import os, jaxlib; print(os.path.dirname(jaxlib.__file__))"):$LD_LIBRARY_PATH


