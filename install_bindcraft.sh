#!/bin/bash
################## BindCraft installation script
################## specify conda/mamba folder, and installation folder for git repositories, and whether to use mamba or $pkg_manager
# Default value for pkg_manager
pkg_manager='conda'
backend='rocm'
rocm=''
cuda=''
python_env=''

# Define the short and long options
OPTIONS=p:c:b:r:e:
LONGOPTIONS=pkg_manager:,cuda:,backend:,rocm:,python_env:,python-env:

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    -b|--backend)
      backend="$2"
      shift 2
      ;;
    -r|--rocm)
      rocm="$2"
      backend='rocm'
      shift 2
      ;;
    -e|--python_env|--python-env)
      python_env="$2"
      shift 2
      ;;
    -c|--cuda)
      cuda="$2"
      backend='cuda'
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

case "$backend" in
  rocm|hip|cuda|cpu)
    ;;
  *)
    echo -e "Error: Unsupported backend '$backend'. Use one of: rocm, hip, cuda, cpu." >&2
    exit 1
    ;;
esac

if [ "$backend" = "hip" ]; then
  backend='rocm'
fi

# Example usage of the parsed variables
echo -e "Package manager: $pkg_manager"
echo -e "Backend: $backend"
if [ -n "$python_env" ]; then
  echo -e "Python environment: $python_env"
fi
if [ "$backend" = "cuda" ]; then
  echo -e "CUDA: $cuda"
elif [ "$backend" = "rocm" ] && [ -n "$rocm" ]; then
  echo -e "ROCm: $rocm"
fi

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths needed for installation and check for conda installation
install_dir=$(pwd)
env_activate_cmd="$pkg_manager activate BindCraft"

if [ -n "$python_env" ]; then
  echo -e "Loading existing Python environment\n"
  if [ -f "${python_env}/bin/activate" ]; then
    source "${python_env}/bin/activate" || { echo -e "Error: Failed to activate Python environment at '$python_env'."; exit 1; }
    env_activate_cmd="source ${python_env}/bin/activate"
  else
    CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
    echo -e "Conda is installed at: $CONDA_BASE"
    source ${CONDA_BASE}/bin/activate "$python_env" || { echo -e "Error: Failed to activate existing Conda environment '$python_env'."; exit 1; }
    env_activate_cmd="$pkg_manager activate $python_env"
  fi
  echo -e "Using existing Python environment at $(python -c 'import sys; print(sys.prefix)')"
else
  CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
  echo -e "Conda is installed at: $CONDA_BASE"

  ### BindCraft install begin, create base environment
  echo -e "Installing BindCraft environment\n"
  $pkg_manager create --name BindCraft python=3.10 -y || { echo -e "Error: Failed to create BindCraft conda environment"; exit 1; }
  conda env list | grep -w 'BindCraft' >/dev/null 2>&1 || { echo -e "Error: Conda environment 'BindCraft' does not exist after creation."; exit 1; }

  # Load newly created BindCraft environment
  echo -e "Loading BindCraft environment\n"
  source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindCraft || { echo -e "Error: Failed to activate the BindCraft environment."; exit 1; }
  [ "$CONDA_DEFAULT_ENV" = "BindCraft" ] || { echo -e "Error: The BindCraft environment is not active."; exit 1; }
  echo -e "BindCraft environment activated at ${CONDA_BASE}/envs/BindCraft"
fi

# install required conda packages
if [ -z "$python_env" ]; then
  echo -e "Installing conda requirements\n"
  common_conda_packages=(
    pip pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg fsspec py3dmol
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax
  )

  if [ "$backend" = "cuda" ]; then
    if [ -n "$cuda" ]; then
      CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install \
        "${common_conda_packages[@]}" \
        'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0=*cuda*' cuda-nvcc cudnn \
        -c conda-forge -c nvidia -y \
      || { echo -e "Error: Failed to install conda packages."; exit 1; }
    else
      $pkg_manager install \
        "${common_conda_packages[@]}" \
        'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0=*cuda*' cuda-nvcc cudnn \
        -c conda-forge -c nvidia -y \
      || { echo -e "Error: Failed to install conda packages."; exit 1; }
    fi
  elif [ "$backend" = "rocm" ]; then
    $pkg_manager install \
      "${common_conda_packages[@]}" \
      -c conda-forge -y \
    || { echo -e "Error: Failed to install conda packages."; exit 1; }

    echo -e "Installing JAX ROCm backend\n"
    if [ -n "$rocm" ]; then
      rocm_major="${rocm%%.*}"
      case "$rocm_major" in
        6)
          pip install --upgrade \
            'jax>=0.4,<=0.6.0' \
            'jax-rocm60-plugin>=0.4,<=0.6.0' \
            'jax-rocm60-pjrt>=0.4,<=0.6.0' \
          || { echo -e "Error: Failed to install JAX ROCm backend for ROCm ${rocm}."; exit 1; }
          ;;
        *)
          echo -e "Error: Unsupported ROCm version '$rocm'. This installer currently supports ROCm 6.x through the JAX rocm60 plugin packages." >&2
          exit 1
          ;;
      esac
    else
      pip install --upgrade 'jax[rocm]>=0.4,<=0.6.0' \
      || { echo -e "Error: Failed to install JAX ROCm backend."; exit 1; }
    fi
  else
    $pkg_manager install \
      "${common_conda_packages[@]}" \
      'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0' \
      -c conda-forge -y \
    || { echo -e "Error: Failed to install conda packages."; exit 1; }
  fi
else
  echo -e "Skipping conda dependency installation because --python-env was provided\n"
fi

# make sure all required packages were installed
if [ -z "$python_env" ]; then
  required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer seaborn tqdm jupyter ffmpeg fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib jax)

  if [ "$backend" = "cuda" ]; then
    required_packages+=(cuda-nvcc cudnn)
  fi

  missing_packages=()

  # Check each package
  for pkg in "${required_packages[@]}"; do
    conda list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
  done

  # If any packages are missing, output error and exit
  if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
      echo -e " - $pkg"
    done
    exit 1
  fi
else
  required_python_modules=(
    jax:jax
    jaxlib:jaxlib
    numpy:numpy
    pandas:pandas
    matplotlib:matplotlib
    biopython:Bio
    scipy:scipy
    pdbfixer:pdbfixer
    seaborn:seaborn
    tqdm:tqdm
    fsspec:fsspec
    py3dmol:py3Dmol
    chex:chex
    dm-haiku:haiku
    flax:flax
    dm-tree:tree
    joblib:joblib
    ml-collections:ml_collections
    immutabledict:immutabledict
    optax:optax
  )
  missing_modules=()

  for item in "${required_python_modules[@]}"; do
    pkg="${item%%:*}"
    module="${item#*:}"
    python -c "import ${module}" >/dev/null 2>&1 || missing_modules+=("$pkg")
  done

  if [ ${#missing_modules[@]} -ne 0 ]; then
    echo -e "Error: The following Python modules are missing from the existing environment:"
    for pkg in "${missing_modules[@]}"; do
      echo -e " - $pkg"
    done
    exit 1
  fi
fi

python -c "import jax, jaxlib; print('JAX', jax.__version__, 'JAXLIB', jaxlib.__version__)" >/dev/null 2>&1 || { echo -e "Error: JAX is not importable after installation"; exit 1; }

# install ColabDesign
if python -c "import colabdesign" >/dev/null 2>&1; then
  echo -e "ColabDesign already installed\n"
else
  echo -e "Installing ColabDesign\n"
  pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps || { echo -e "Error: Failed to install ColabDesign"; exit 1; }
fi
python -c "import colabdesign" >/dev/null 2>&1 || { echo -e "Error: colabdesign module not found after installation"; exit 1; }

# install PyRosetta
if python -c "import pyrosetta" >/dev/null 2>&1; then
  echo -e "PyRosetta already installed\n"
else
  echo -e "Installing PyRosetta\n"
  pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release.cxx11thread.serialization || { echo -e "Error: Failed to install PyRosetta"; exit 1; }
fi
python -c "import pyrosetta" >/dev/null 2>&1 || { echo -e "Error: pyrosetta module not found after installation"; exit 1; }

# AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

# download AF2 weights
mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
[ -s "${params_file}" ] || { echo -e "Error: Could not locate downloaded AlphaFold2 weights"; exit 1; }

# extract AF2 weights
tar tf "${params_file}" >/dev/null 2>&1 || { echo -e "Error: Corrupt AlphaFold2 weights download"; exit 1; }
tar -xvf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2weights"; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo -e "Error: Could not locate extracted AlphaFold2 weights"; exit 1; }
rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }

# chmod executables
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# finish
if [ -n "$CONDA_PREFIX" ]; then
  conda deactivate
elif [ -n "$VIRTUAL_ENV" ]; then
  deactivate
fi
echo -e "BindCraft environment set up\n"

############################################################################################################
############################################################################################################
################## cleanup
if [ -z "$python_env" ]; then
  echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
  $pkg_manager clean -a -y
  echo -e "$pkg_manager cleaned up\n"
else
  echo -e "Skipping ${pkg_manager} cleanup because --python-env was provided\n"
fi

################## finish script
t=$SECONDS 
echo -e "Successfully finished BindCraft installation!\n"
if [ -n "$python_env" ]; then
  echo -e "Activate environment using command: \"$env_activate_cmd\""
else
  echo -e "Activate environment using command: \"$pkg_manager activate BindCraft\""
fi
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
