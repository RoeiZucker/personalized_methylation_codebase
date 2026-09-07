#!/usr/bin/env bash
#SBATCH --mem=45g
#SBATCH -c 15
#SBATCH --gres=gg:g4:1
#SBATCH --time=5-23
#SBATCH --killable
#SBATCH --requeue
#SBATCH --constraint=firefoot|cyril
#SBATCH --job-name=chr12-pred

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 ARRAY_TASKS.tsv" >&2
    exit 2
fi
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "SLURM_ARRAY_TASK_ID must be set to a non-negative integer" >&2
    exit 2
fi

TASK_FILE=$1
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
TASK_ROW=$(sed -n "${LINE_NUMBER}p" "$TASK_FILE")
if [[ -z "$TASK_ROW" ]]; then
    echo "No task row for array index $SLURM_ARRAY_TASK_ID in $TASK_FILE" >&2
    exit 2
fi

IFS=$'\t' read -r TASK_ID JOB_LABEL CONFIG_PATH CHECKPOINT_PATH OUTPUT_PATH <<< "$TASK_ROW"
if [[ "$TASK_ID" != "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Task table index mismatch: expected $SLURM_ARRAY_TASK_ID, found $TASK_ID" >&2
    exit 2
fi

WRAPPER=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_downsample_checkpoint_prediction_params.sh
echo "array_task=$TASK_ID job=$JOB_LABEL checkpoint=$CHECKPOINT_PATH" >&2
exec "$WRAPPER" \
    --config "$CONFIG_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --output-path "$OUTPUT_PATH" \
    --overwrite

# Example, submit every pending task using the generated one-line command:
# bash /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/downsampled_prediction_chr12_standardized_v1_sbatch_commands.sh
#
# Example, retry only array task 17 with the same resources:
# sbatch --array=17 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_downsampled_prediction_chr12_standardized_v1_array.sh /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/downsampled_prediction_chr12_standardized_v1_array_tasks.txt
