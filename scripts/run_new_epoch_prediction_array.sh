#!/usr/bin/env bash
#SBATCH --mem=45g
#SBATCH -c 15
#SBATCH --gres=gg:g4:1
#SBATCH --time=5-23
#SBATCH --killable
#SBATCH --requeue
#SBATCH --constraint=firefoot|cyril
#SBATCH --job-name=new-epoch-pred

set -euo pipefail

REPO=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code
TASK_FILE="$REPO/scripts/downsampled_new_epoch_prediction_tasks.tsv"
WRAPPER="$REPO/scripts/run_downsample_checkpoint_prediction_params.sh"

if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "SLURM_ARRAY_TASK_ID must be set to a non-negative integer" >&2
    exit 2
fi

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
TASK_ROW=$(sed -n "${LINE_NUMBER}p" "$TASK_FILE")
if [[ -z "$TASK_ROW" ]]; then
    echo "No task row for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

IFS=$'\t' read -r TASK_ID JOB_LABEL CONFIG_PATH CHECKPOINT_PATH OUTPUT_PATH <<< "$TASK_ROW"
if [[ "$TASK_ID" != "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Task index mismatch: expected $SLURM_ARRAY_TASK_ID, found $TASK_ID" >&2
    exit 2
fi

if [[ -s "$OUTPUT_PATH" ]]; then
    echo "Prediction already exists; skipping: $OUTPUT_PATH" >&2
    exit 0
fi

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
TEMP_OUTPUT="${OUTPUT_PATH}.partial.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RESTART_COUNT}"
trap 'rm -f "$TEMP_OUTPUT"' EXIT INT TERM

echo "task=$TASK_ID label=$JOB_LABEL model=$CHECKPOINT_PATH" >&2
"$WRAPPER" \
    --config "$CONFIG_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --output-path "$TEMP_OUTPUT" \
    --overwrite

if [[ ! -s "$TEMP_OUTPUT" ]]; then
    echo "Prediction command produced an empty output: $TEMP_OUTPUT" >&2
    exit 1
fi
mv "$TEMP_OUTPUT" "$OUTPUT_PATH"
echo "Prediction complete: $OUTPUT_PATH" >&2

# Example, submit every generated prediction task, with at most 10 running:
# sbatch --array=0-99%10 scripts/run_new_epoch_prediction_array.sh
#
# Example, retry one failed prediction task:
# sbatch --array=17 scripts/run_new_epoch_prediction_array.sh
