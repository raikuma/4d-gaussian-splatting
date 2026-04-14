#!/bin/bash
set -euo pipefail

PARTITION="${PARTITION:-gigabyte_a6000}"
GPU_TYPE="${GPU_TYPE:-A6000}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
ENV_NAME="${ENV_NAME:-4dgs}"
CONFIG_PATH="${CONFIG_PATH:-configs/dynerf/cook_spinach_remote_baseline_sanity.yaml}"
RESET_ENV="${RESET_ENV:-1}"

srun -p "${PARTITION}" \
  --gres="gpu:${GPU_TYPE}:1" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  bash -lc "
set -euo pipefail
module purge
ml cuda/11.8
source ~/anaconda3/etc/profile.d/conda.sh

if [ \"${RESET_ENV}\" = \"1\" ] && conda env list | awk '{print \$1}' | grep -qx \"${ENV_NAME}\"; then
  conda env remove -y -n \"${ENV_NAME}\"
fi

cd ~/4d-gaussian-splatting
chmod +x scripts/setup_remote_env.sh
./scripts/setup_remote_env.sh \"${ENV_NAME}\"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate \"${ENV_NAME}\"
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/11.8
export TORCH_CUDA_ARCH_LIST=8.6

mkdir -p /scratch/woongohcho/outputs

python -c \"import torch; print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); print('device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'))\"
python -c \"import diff_gaussian_rasterization; print('diff_gaussian_rasterization import ok')\"
python train.py --config \"${CONFIG_PATH}\"
"
