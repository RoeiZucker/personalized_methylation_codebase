#!/usr/bin/env bash
set -euo pipefail

# Example task file creation:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_downsample_dataset_tasks.py --ratio 0.25 --task-filter retrain --skip-existing --output-task-file /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/downsample_ratio_0p25_retrain_tasks.tsv
#
# Example Slurm array run:
# TASK_FILE=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/downsample_ratio_0p25_retrain_tasks.tsv
# N=$(wc -l < "$TASK_FILE")
# sbatch -A tomhope --mem=30g -c4 --time=1-23 --array=1-${N}%4 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_downsample_dataset_task.sh "$TASK_FILE"

. /sci/nosnap/michall/roeizucker/new_python_env/bin/activate

PROJECT="${PROJECT:-/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code}"
TASK_FILE="$1"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set" >&2
  exit 2
fi

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")"
if [[ -z "$line" ]]; then
  echo "No task found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

read -r input_dataset_path output_dataset_path ratio <<< "$line"
if [[ -z "$input_dataset_path" || -z "$output_dataset_path" || -z "$ratio" ]]; then
  echo "Bad task line ${SLURM_ARRAY_TASK_ID}: $line" >&2
  exit 2
fi

echo "Downsampling task ${SLURM_ARRAY_TASK_ID}"
echo "input_dataset_path=$input_dataset_path"
echo "output_dataset_path=$output_dataset_path"
echo "ratio=$ratio"

python3 "$PROJECT/scripts/downsample_token_classification_dataset.py" \
  --input-dataset-path "$input_dataset_path" \
  --output-dataset-path "$output_dataset_path" \
  --minority-to-majority-ratio "$ratio"
