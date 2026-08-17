#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <install_dir>" >&2
    exit 1
fi

export WORKSPACE_ROOT_DIR=$(realpath $1)
export PRORED_RELEASE=v2.0.0
export SYSTEM_NAME=tuolumne
CC=/opt/rocm-6.4.3/bin/amdclang
CXX=/opt/rocm-6.4.3/bin/amdclang++
MPICC=/opt/cray/pe/mpich/9.0.1/ofi/crayclang/20.0/bin/mpicc
MPICXX=/opt/cray/pe/mpich/9.0.1/ofi/crayclang/20.0/bin/mpicxx
export MODULES="StdEnv cmake/3.29.2 PrgEnv-amd/8.7.0 cray-mpich-abi/9.0.1 rocm/6.4.3 rccl python/3.12.2"
module purge --force
module load $MODULES
export PROJECT_WORKSPACE_DIR=$WORKSPACE_ROOT_DIR/builds
export PROJECT_APPS_DIR=$WORKSPACE_ROOT_DIR/apps
export PRORED_VENV=prored-$PRORED_RELEASE-$SYSTEM_NAME
export PRORED_VENV_DIR=$PROJECT_WORKSPACE_DIR/$PRORED_VENV

mkdir -p $PROJECT_WORKSPACE_DIR
export INSTALL_LOCK=$PROJECT_WORKSPACE_DIR/prored_install_tuolumne.lock
exec 9>$INSTALL_LOCK
if ! flock -n 9; then
    echo "Another prored install is already running for $WORKSPACE_ROOT_DIR" >&2
    exit 1
fi
pushd $PROJECT_WORKSPACE_DIR

rm -rf $PRORED_VENV_DIR
python3 -m venv $PRORED_VENV_DIR
pushd $PRORED_VENV_DIR
source ./bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

pip install mpi4py==4.1.2
pip install accelerate torchmetrics biopython==1.85
pip install flux-python maestrowf
pip install transformers@git+https://github.com/Biohub/transformers.git@main
pip install esm@git+https://github.com/Biohub/esm.git@main

pip install --upgrade 'jax==0.4.35' 'jax-rocm60-plugin==0.4.35' 'jax-rocm60-pjrt==0.4.35'
pip install pdbfixer seaborn 'dm-haiku<0.0.14' dm-tree ml-collections immutabledict flax==0.9.0 orbax-checkpoint==0.7.0 optax==0.2.4 chex==0.1.88 'jax==0.4.35'
python - <<'PY'
modules = (
    "jax",
    "jaxlib",
    "haiku",
    "tree",
    "ml_collections",
    "immutabledict",
    "flax",
    "orbax.checkpoint",
    "optax",
    "chex",
)
for module in modules:
    __import__(module)
PY

export LD_LIBRARY_PATH=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.4.3/install/lib:$LD_LIBRARY_PATH

popd #$PRORED_VENV_DIR

export ACTIVATE_SCRIPT=$PROJECT_WORKSPACE_DIR/$PRORED_VENV.sh
cat > $ACTIVATE_SCRIPT << EOF
#!/bin/bash

export WORKSPACE_ROOT_DIR="$WORKSPACE_ROOT_DIR"
export SYSTEM_NAME="$SYSTEM_NAME"
export PRORED_RELEASE="$PRORED_RELEASE"
module purge --force
module load $MODULES
export PROJECT_WORKSPACE_DIR=\$WORKSPACE_ROOT_DIR/builds
export PRORED_VENV=prored-\$PRORED_RELEASE-\$SYSTEM_NAME
export PRORED_VENV_DIR=\$PROJECT_WORKSPACE_DIR/\$PRORED_VENV
source \$PRORED_VENV_DIR/bin/activate
export LD_LIBRARY_PATH=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-7.2.0/install/lib:$LD_LIBRARY_PATH
EOF
chmod +x $ACTIVATE_SCRIPT

deactivate
popd #$PROJECT_WORKSPACE_DIR
