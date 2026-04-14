#!/bin/bash
#SBATCH -J focusgs-sanity
#SBATCH -p gigabyte_a6000
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/woongohcho/logs/%x-%j.out

set -euo pipefail

ENV_NAME="${ENV_NAME:-4dgs}"
CONFIG_PATH="${CONFIG_PATH:-configs/dynerf/cook_spinach_remote_focusgs_random_sanity.yaml}"
REPO_DIR="${REPO_DIR:-$HOME/4d-gaussian-splatting}"
CUDA_HOME_PATH="${CUDA_HOME_PATH:-/opt/ohpc/pub/apps/cuda/11.8}"
TORCH_ARCH="${TORCH_ARCH:-8.6}"

module purge
ml cuda/11.8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export CUDA_HOME="${CUDA_HOME_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"

mkdir -p /scratch/woongohcho/logs
mkdir -p /scratch/woongohcho/outputs

cd "${REPO_DIR}"

echo "hostname=$(hostname)"
echo "cwd=$(pwd)"
echo "config=${CONFIG_PATH}"
echo "env=${ENV_NAME}"
echo "cuda_home=${CUDA_HOME}"
echo "torch_arch=${TORCH_CUDA_ARCH_LIST}"

python -c "import torch; print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); print('device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'))"
python train.py --config "${CONFIG_PATH}"
