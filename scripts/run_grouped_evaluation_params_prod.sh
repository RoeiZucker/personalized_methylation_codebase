#!/usr/bin/env bash
set -euo pipefail

. /sci/nosnap/michall/roeizucker/new_python_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/evaluate_grouped_predictions.py" "$@"

# how to run variability-free:
# sbatch -A tomhope --mem=24g -c2 --time=12:00:00 --killable --requeue --wrap="/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_grouped_evaluation_params_prod.sh --mode variability_free --pretraining-group epoch_3_pretraining --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer --tissue-name Liver-Hepatocytes --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes --precomputed-scores-dir /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/results/precomputed_scores/Liver-Hepatocytes --output-root /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/results/all_tissues_grouped_evaluations/Liver-Hepatocytes --chroms chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22 --skip-mean-label --update-existing"
#
# how to run variability-bins:
# sbatch -A tomhope --mem=32g -c2 --time=24:00:00 --killable --requeue --wrap="/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_grouped_evaluation_params_prod.sh --mode variability_bins --pretraining-group epoch_3_pretraining --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer --tissue-name Liver-Hepatocytes --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes --mean-label-cache-dir /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/results/mean_label_caches/Liver-Hepatocytes --output-root /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/results/all_tissues_variability_bins/Liver-Hepatocytes --chroms chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22 --update-existing"
