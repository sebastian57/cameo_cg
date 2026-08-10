#!/bin/bash
# Build the CG_BIAS PLUMED action.
#
# Must be rebuilt whenever sampling/protocol.py changes PROTOCOL_VERSION -- the plugin
# and server refuse to talk across versions rather than misreading each other's bytes.
#
# Usage:  sampling/plugin/compile_plugin.sh
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

module --force purge
module load Stages/2025
module load GCC/13.3.0 ParaStationMPI/5.11.0-1
module load GROMACS/2024.3-PLUMED-2.9.3

cd "$HERE"
export CPLUS_INCLUDE_PATH="$(dirname "$(dirname "$(which plumed)")")/include/plumed/core${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

rm -f CGBias.so
plumed mklib CGBias.cpp
test -s CGBias.so || { echo "ERROR: CGBias.so was not produced" >&2; exit 1; }

echo "built: $HERE/CGBias.so"
echo "load in plumed.dat with:  LOAD FILE=$HERE/CGBias.so"
