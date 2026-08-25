#!/bin/bash
# flux: -N 1
# flux: -x
# flux: --amd-gpumode=SPX
# flux: -t 2h
# flux: -q batch
# flux: -B my_bank

set -euo pipefail

BINDCRAFT_MODULES=${BINDCRAFT_MODULES:-"StdEnv PrgEnv-amd/8.7.0 cray-mpich-abi/9.0.1 rocm/7.2.1 rccl python/3.12.2"}
if command -v module >/dev/null 2>&1; then
    module purge --force
    module load ${BINDCRAFT_MODULES}
fi

SETTINGS=""
FILTERS=""
ADVANCED=""
RANKS=""
PYTHON_ENV=""
TEMP=$(getopt -o s:f:a:r:p: --long settings:,filters:,advanced:,ranks:,py-env: -n 'bindcraft.flux' -- "$@")
eval set -- "$TEMP"

while true ; do
    case "$1" in
        -s|--settings) SETTINGS="$2" ; shift 2 ;;
        -f|--filters) FILTERS="$2" ; shift 2 ;;
        -a|--advanced) ADVANCED="$2" ; shift 2 ;;
        -r|--ranks) RANKS="$2" ; shift 2 ;;
        -p|--py-env) PYTHON_ENV="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Invalid Option" ; exit 1 ;;
    esac
done

if [ -n "${PYTHON_ENV}" ] && [ -f "${PYTHON_ENV}/bin/activate" ]; then
    source "${PYTHON_ENV}/bin/activate"
else
    echo "Error: pass --py-env with an existing non-Conda Python environment containing bin/activate."
    exit 1
fi

python -c "import jax, colabdesign, pyrosetta" >/dev/null 2>&1 || {
    echo "Error: active Python environment is missing one or more required BindCraft modules."
    echo "       Python: $(command -v python)"
    exit 1
}

command -v flux >/dev/null 2>&1 || {
    echo "Error: flux is not available on PATH after loading modules."
    exit 1
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

if [ -z "$SETTINGS" ]; then
    echo "Error: The -s or --settings option is required."
    exit 1
fi

echo "Running the BindCraft pipeline with Flux"
flux run -N 1 -n "${RANKS}" -g 1 python -u "${SCRIPT_DIR}/bindcraft.py" --settings "${SETTINGS}" --filters "${FILTERS}" --advanced "${ADVANCED}"
