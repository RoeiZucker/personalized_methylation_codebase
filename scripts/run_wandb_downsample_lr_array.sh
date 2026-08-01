#!/usr/bin/env bash
set -euo pipefail

. /sci/nosnap/michall/roeizucker/new_python_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID

TASK_FILE="$1"
PROJECT="${PROJECT:-/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code}"
BASE_CONFIG="${BASE_CONFIG:-$PROJECT/configs/wandb_classification_full_training_r3_10k_limited_run_no_start_eval.yaml}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-personalized-methylation-r3-downsampled-bs14}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-r3-downsampled-bs14}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set" >&2
  exit 2
fi

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")"
if [[ -z "$line" ]]; then
  echo "No task found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

read -r ratio learning_rate train_dataset_path sample_name wandb_run_id <<< "$line"
if [[ -z "$ratio" || -z "$learning_rate" || -z "$train_dataset_path" || -z "$sample_name" || -z "$wandb_run_id" ]]; then
  echo "Bad task line ${SLURM_ARRAY_TASK_ID}: $line" >&2
  exit 2
fi

echo "Running W&B task ${SLURM_ARRAY_TASK_ID}"
echo "ratio=$ratio"
echo "learning_rate=$learning_rate"
echo "train_dataset_path=$train_dataset_path"
echo "wandb_run_id=$wandb_run_id"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python3 "$PROJECT/src/wandb_sweep_training.py" \
  --base-config "$BASE_CONFIG" \
  --sample-name "$sample_name" \
  --wandb-run-id "$wandb_run_id" \
  --project "$WANDB_PROJECT_NAME" \
  --group "$WANDB_GROUP_NAME" \
  --learning-rate "$learning_rate" \
  --batch-size 14 \
  --eval-steps "${EVAL_STEPS:-1000}" \
  --logging-steps "${LOGGING_STEPS:-333}" \
  --no-eval-on-start \
  --train-dataset-path "$train_dataset_path"

# Example, create the task file:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_wandb_downsample_lr_tasks.py --output-task-file /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/wandb_downsample_lr_tasks.tsv --ratios 1.0 0.5 0.25 --learning-rates 1e-6 1e-5 5e-5
#
# Example, submit all generated tasks with at most one running at once:
# TASK_FILE=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/wandb_downsample_lr_tasks.tsv
# N=$(wc -l < "$TASK_FILE")
# sbatch -A tomhope --mem=45g -c15 --gres=gg:g4:3 --time=5-23 --killable --requeue --array=1-${N}%1 scripts/run_wandb_downsample_lr_array.sh "$TASK_FILE"
