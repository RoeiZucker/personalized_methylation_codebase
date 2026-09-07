#!/usr/bin/env bash
set -euo pipefail

REPO=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code
TASK_FILE="$REPO/configs/downsampled_runs/save_steps_4000_resume_array_tasks.txt"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set" >&2
  exit 2
fi

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
CONFIG_PATH=$(sed -n "${LINE_NUMBER}p" "$TASK_FILE")

if [[ -z "$CONFIG_PATH" || ! -f "$CONFIG_PATH" ]]; then
  echo "No config found for array task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

echo "Array task $SLURM_ARRAY_TASK_ID is training with $CONFIG_PATH"
exec "$REPO/scripts/run_training_params_phoenix.sh" "$CONFIG_PATH"

# Example, submit original array tasks 10-20 with at most ten running concurrently:
# sbatch --array=0-10%10 --mem=90g -c15 --gres=gg:g4:3 --time=5-23 --killable --requeue --constraint="firefoot|cyril" --job-name="ret10-save4k-10-20" /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_save_steps_4000_training_array_task.sh
