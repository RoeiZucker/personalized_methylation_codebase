#!/usr/bin/env bash
set -euo pipefail

# Submit the uncapped combined eval+test chr12 k-mer track bundles.
# Example:
#   bash /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/chr12_combined_kmer_tracks_sbatch_commands.sh

PROJECT=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code
sbatch --array=0-1%2 \
    "$PROJECT/scripts/run_chr12_combined_kmer_tracks_array.sh" \
    "$PROJECT/scripts/chr12_combined_kmer_tracks_array_tasks.txt"
