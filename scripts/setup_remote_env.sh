#!/bin/bash
set -euo pipefail

ENV_NAME="${1:-4dgs}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8}"
CUDA_HOME_DEFAULT="${CUDA_HOME_DEFAULT:-/opt/ohpc/pub/apps/cuda/11.8}"
TORCH_CUDA_ARCH_LIST_DEFAULT="${TORCH_CUDA_ARCH_LIST_DEFAULT:-8.6}"

module purge
ml "${CUDA_MODULE}"
source ~/anaconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip setuptools=69.5.1 wheel
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install setuptools==69.5.1 "numpy<2" plyfile==0.8.1 tqdm==4.66.1 torchmetrics==0.11.4 imagesize==1.4.1 kornia==0.6.12 omegaconf==2.3.0 tensorboard ninja scikit-image lpips opencv-python==4.8.1.78

export CUDA_HOME="${CUDA_HOME_DEFAULT}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_DEFAULT}"

python -m pip install --no-build-isolation ./simple-knn
python -m pip install --no-build-isolation ./pointops2

echo "Environment ready: ${ENV_NAME}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
