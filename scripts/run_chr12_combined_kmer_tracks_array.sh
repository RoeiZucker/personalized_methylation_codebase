#!/usr/bin/env bash
#SBATCH --mem=120g
#SBATCH -c 4
#SBATCH --time=2-00
#SBATCH --killable
#SBATCH --requeue
#SBATCH --job-name=chr12-kmer
#SBATCH --output=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/chr12_kmer_tracks_%A_%a.out
#SBATCH --error=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/chr12_kmer_tracks_%A_%a.err

set -euo pipefail

# Example, submit atlas and pretrained bundles as a two-task array:
# bash /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/chr12_combined_kmer_tracks_sbatch_commands.sh
#
# Example, retry only the atlas task:
# sbatch --array=0 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_chr12_combined_kmer_tracks_array.sh /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/chr12_combined_kmer_tracks_array_tasks.txt

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 ARRAY_TASKS.txt" >&2
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

IFS=$'\t' read -r TASK_ID BASELINE OUTPUT_DIR <<< "$TASK_ROW"
if [[ "$TASK_ID" != "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Task table index mismatch: expected $SLURM_ARRAY_TASK_ID, found $TASK_ID" >&2
    exit 2
fi
if [[ "$BASELINE" != "atlas" && "$BASELINE" != "pretrained" ]]; then
    echo "Unsupported baseline in task table: $BASELINE" >&2
    exit 2
fi

PROJECT=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code
RESULTS_ROOT=$PROJECT/scripts/results/downsampled_standard_ratio_0p25_lr_3e-05_chr12_standardized_split_v1
VARIABILITY_ROOT=/sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1

. /sci/nosnap/michall/roeizucker/new_python_env/bin/activate
export PYTHONUNBUFFERED=1

echo "array_task=$TASK_ID baseline=$BASELINE output=$OUTPUT_DIR" >&2
python3 "$PROJECT/scripts/generate_methylation_kmer_tracks.py" \
    --results-root "$RESULTS_ROOT" \
    --split combined \
    --variability-root "$VARIABILITY_ROOT" \
    --chrom chr12 \
    --output-dir "$OUTPUT_DIR" \
    --baseline "$BASELINE" \
    --combined-samples-track \
    --min-participants-per-kmer 3 \
    --workers 4 \
    --ucsc-item-details

python3 "$PROJECT/scripts/validate_methylation_kmer_track_bundle.py" \
    "$OUTPUT_DIR" \
    --baseline "$BASELINE" \
    --expected-participants 31 \
    --expected-split-rows 62 \
    --chrom chr12
