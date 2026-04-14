#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

declare -a JOB_SPECS=(
  "4dgs-batcha configs/dynerf/cook_spinach_remote_baseline_batch_a.yaml"
  "4dgs-focusgs-r025-batcha configs/dynerf/cook_spinach_remote_focusgs_random_r025_batch_a.yaml"
  "4dgs-focusgs-r050-batcha configs/dynerf/cook_spinach_remote_focusgs_random_r050_batch_a.yaml"
  "4dgs-focusgs-eth005-batcha configs/dynerf/cook_spinach_remote_focusgs_error_threshold_t005_batch_a.yaml"
)

cd "${REPO_DIR}"
mkdir -p /scratch/woongohcho/logs

for spec in "${JOB_SPECS[@]}"; do
  job_name="${spec%% *}"
  config_path="${spec#* }"
  job_id="$(sbatch --parsable --time="${TIME_LIMIT}" --job-name="${job_name}" --export=ALL,ENV_NAME=4dgs,CONFIG_PATH="${config_path}" scripts/slurm/run_a6000_train.sbatch)"
  echo "${job_name},${job_id},${config_path}"
done
