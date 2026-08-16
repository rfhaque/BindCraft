#!/bin/bash
set -euo pipefail

BINDCRAFT_MODULES=${BINDCRAFT_MODULES:-"StdEnv PrgEnv-amd/8.7.0 cray-mpich-abi/9.0.1 rocm/6.4.3 rccl python/3.12.2 flux"}
if command -v module >/dev/null 2>&1; then
    module purge --force
    module load ${BINDCRAFT_MODULES}
fi

if [ -n "${BINDCRAFT_ENV_ACTIVATE:-}" ]; then
    source "${BINDCRAFT_ENV_ACTIVATE}"
elif [ -n "${BINDCRAFT_PYTHON_ENV:-}" ] && [ -f "${BINDCRAFT_PYTHON_ENV}/bin/activate" ]; then
    source "${BINDCRAFT_PYTHON_ENV}/bin/activate"
else
    echo "Error: set BINDCRAFT_PYTHON_ENV to an existing non-Conda Python environment with bin/activate,"
    echo "       or set BINDCRAFT_ENV_ACTIVATE to an activation script."
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

SETTINGS=""
FILTERS=""
ADVANCED=""
TEMP=$(getopt -o s:f:a: --long settings:,filters:,advanced: -n 'bindcraft.flux' -- "$@")
eval set -- "$TEMP"

while true ; do
    case "$1" in
        -s|--settings) SETTINGS="$2" ; shift 2 ;;
        -f|--filters) FILTERS="$2" ; shift 2 ;;
        -a|--advanced) ADVANCED="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Invalid Option" ; exit 1 ;;
    esac
done

if [ -z "$SETTINGS" ]; then
    echo "Error: The -s or --settings option is required."
    exit 1
fi

echo "Running the BindCraft pipeline with Flux"
flux run -N 1 -n 1 -g 1 python -u "${SCRIPT_DIR}/bindcraft.py" --settings "${SETTINGS}" --filters "${FILTERS}" --advanced "${ADVANCED}"
