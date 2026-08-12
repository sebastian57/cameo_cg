#!/bin/bash
# Build the CG_BIAS PLUMED action.
#
# Must be rebuilt whenever sampling/protocol.py changes PROTOCOL_VERSION -- the plugin
# and server refuse to talk across versions rather than misreading each other's bytes.
#
# Usage:
#   sampling/plugin/compile_plugin.sh                  # socket backend only (default)
#   sampling/plugin/compile_plugin.sh --with-connector # + in-process compiled-model backend
#
# The connector build additionally links chemtrain-deploy's libconnector, which lets
# CG_BIAS evaluate an exported model (MODEL=) in-process instead of shipping positions to
# the Python bias server. libconnector is a prebuilt shared object depending only on libc
# and libstdc++ -- no CUDA, MPI or LAMMPS linkage -- so this needs no Bazel and no XLA
# headers. Runtime requires it on LD_LIBRARY_PATH.
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONNECTOR_ROOT="${CAMEO_CONNECTOR_ROOT:-/e/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain_cameo/chemtrain-deploy}"

WITH_CONNECTOR=0
for arg in "$@"; do
    case "$arg" in
        --with-connector) WITH_CONNECTOR=1 ;;
        -h|--help) sed -n '1,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

module --force purge
module load Stages/2025
module load GCC/13.3.0 ParaStationMPI/5.11.0-1
module load GROMACS/2024.3-PLUMED-2.9.3

cd "$HERE"
export CPLUS_INCLUDE_PATH="$(dirname "$(dirname "$(which plumed)")")/include/plumed/core${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

rm -f CGBias.so CGBias.o

if [[ "$WITH_CONNECTOR" == "1" ]]; then
    CONN_INC="${CONNECTOR_ROOT}/connector"
    CONN_LIB="${CONNECTOR_ROOT}/lib"
    [[ -f "${CONN_INC}/libconnector.h" ]] || {
        echo "ERROR: ${CONN_INC}/libconnector.h not found. Set CAMEO_CONNECTOR_ROOT." >&2
        exit 1; }
    [[ -f "${CONN_LIB}/libconnector.so" ]] || {
        echo "ERROR: ${CONN_LIB}/libconnector.so not found. Set CAMEO_CONNECTOR_ROOT." >&2
        exit 1; }
    echo "building WITH the in-process connector backend"
    echo "  headers: ${CONN_INC}"
    echo "  library: ${CONN_LIB}/libconnector.so"

    # `plumed mklib` takes no flag arguments in PLUMED 2.9 -- it evaluates fixed `compile`
    # and `link_installed` templates out of compile_options.sh. Extra -D/-I/-l therefore
    # cannot be injected through it, and passing them via environment variables silently
    # does nothing: the resulting .so builds fine, takes the #ifndef branch, and only fails
    # at runtime with "requires -DCGBIAS_WITH_CONNECTOR". So source the same templates and
    # append to them, which is exactly what mklib does internally.
    # Use $PLUMED_ROOT from the module, NOT $(plumed info --root): the MPI stack prints a
    # UCX warning to stdout, which lands inside the captured path and yields
    # "File name too long".
    PL_ROOT="${PLUMED_ROOT:?PLUMED_ROOT unset - is the GROMACS/PLUMED module loaded?}"
    # compile_options.sh expands $PLUMED_INCLUDEDIR/$PLUMED_PROGRAM_NAME. The `plumed`
    # launcher normally exports both; sourcing the file directly does not, and `set -u`
    # turns that into an "unbound variable" abort. PLUMED_ROOT is <prefix>/lib/plumed.
    : "${PLUMED_PROGRAM_NAME:=plumed}"
    : "${PLUMED_INCLUDEDIR:=${PL_ROOT%/lib/plumed}/include}"
    export PLUMED_PROGRAM_NAME PLUMED_INCLUDEDIR
    [[ -d "${PLUMED_INCLUDEDIR}/${PLUMED_PROGRAM_NAME}" ]] || {
        echo "ERROR: PLUMED headers not at ${PLUMED_INCLUDEDIR}/${PLUMED_PROGRAM_NAME}" >&2
        exit 1; }
    source "${PL_ROOT}/src/config/compile_options.sh"
    # Both templates END with `-o`, and mklib appends "<output> <input>". Extra flags must
    # therefore be spliced in BEFORE that trailing -o; appending them puts them between -o
    # and its argument, which fails with "linker input file not found".
    # -std=c++17 last so it overrides the older standard in PLUMED's own template;
    # libconnector.h and std::make_unique both need it.
    eval "${compile% -o} -DCGBIAS_WITH_CONNECTOR -I${CONN_INC} -std=c++17 -o" \
         CGBias.o CGBias.cpp
    eval "${link_installed% -o} -L${CONN_LIB} -lconnector -Wl,-rpath,${CONN_LIB} -o" \
         CGBias.so CGBias.o
    rm -f CGBias.o
else
    echo "building socket backend only (pass --with-connector to enable MODEL=)"
    plumed mklib CGBias.cpp
fi

test -s CGBias.so || { echo "ERROR: CGBias.so was not produced" >&2; exit 1; }

if [[ "$WITH_CONNECTOR" == "1" ]]; then
    # Guard against the silent-no-op failure above ever coming back.
    readelf -d CGBias.so 2>/dev/null | grep -q 'libconnector' || {
        echo "ERROR: CGBias.so does not link libconnector -- the connector flags were "
        echo "       dropped. MODEL= would fail at runtime." >&2
        exit 1; }
    echo "verified: CGBias.so links libconnector"
fi

echo "built: $HERE/CGBias.so"
echo "load in plumed.dat with:  LOAD FILE=$HERE/CGBias.so"
if [[ "$WITH_CONNECTOR" == "1" ]]; then
    echo "run with: export LD_LIBRARY_PATH=${CONNECTOR_ROOT}/lib:\$LD_LIBRARY_PATH"
fi
