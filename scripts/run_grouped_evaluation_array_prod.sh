#!/usr/bin/env bash
set -euo pipefail

. /sci/nosnap/michall/roeizucker/new_python_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID

TASK_FILE="$1"
MODE="${2:-variability_bins}"
PROJECT="${PROJECT:-/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code}"
RESULTS_BASE="${RESULTS_BASE:-/sci/labs/michall/roeizucker/token_cls_models2}"
COMPARISON_DICTS_BASE="${COMPARISON_DICTS_BASE:-/sci/labs/michall/roeizucker/comparison_dicts}"
CHROMS="${CHROMS:-chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set" >&2
  exit 2
fi

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")"
if [[ -z "$line" ]]; then
  echo "No task found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

read -r tissue group <<< "$line"
if [[ -z "$tissue" || -z "$group" ]]; then
  echo "Bad task line ${SLURM_ARRAY_TASK_ID}: $line" >&2
  exit 2
fi

output_root="$PROJECT/scripts/results/all_tissues_variability_bins/$tissue"
if [[ "$MODE" == "variability_free" ]]; then
  output_root="$PROJECT/scripts/results/all_tissues_grouped_evaluations/$tissue"
fi

echo "Running task ${SLURM_ARRAY_TASK_ID}: tissue=${tissue} group=${group} mode=${MODE}"
python3 "$PROJECT/scripts/evaluate_grouped_predictions.py" \
  --mode "$MODE" \
  --pretraining-group "$group" \
  --results-dir "$RESULTS_BASE/_token_cls_${tissue}_kmer" \
  --tissue-name "$tissue" \
  --comparison-dicts-dir "$COMPARISON_DICTS_BASE/$tissue" \
  --output-root "$output_root" \
  --chroms "$CHROMS" \
  --update-existing

# Example, submit five variability-bin tasks with at most two running at once:
# TASK_FILE=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/variability_bins_tasks_5.tsv
# sbatch -A tomhope --mem=32g -c2 --time=24:00:00 --array=1-5%2 \
#   --job-name=varbins_array \
#   --output=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/varbins_array_%A_%a.out \
#   --error=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/varbins_array_%A_%a.err \
#   /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_grouped_evaluation_array_prod.sh "$TASK_FILE" variability_bins
#
# Example, submit all tasks with at most five running at once:
# TASK_FILE=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/variability_bins_tasks.tsv
# N=$(wc -l < "$TASK_FILE")
# sbatch -A tomhope --mem=32g -c2 --time=24:00:00 --array=1-${N}%5 \
#   --job-name=varbins_array \
#   --output=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/varbins_array_%A_%a.out \
#   --error=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/varbins_array_%A_%a.err \
#   /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_grouped_evaluation_array_prod.sh "$TASK_FILE" variability_bins
