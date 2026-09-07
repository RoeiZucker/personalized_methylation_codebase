#!/usr/bin/env bash
set -euo pipefail

. /cs/usr/roeizucker/new_storage/new_python_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID

REPO=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code
python3 "$REPO/data_extraction_script.py" "$1"

# Example, create one chr12-only prediction dataset:
# /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_chr12_prediction_dataset_creation.sh /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/downsampled_runs/chr12_prediction_dataset_v1/Z0000044H_chr12_prediction_v1_data_extraction.yaml
#
# Example, submit one chr12-only dataset build to Phoenix:
# sbatch --mem=64g -c8 --time=1-00 --killable --requeue --wrap="/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_chr12_prediction_dataset_creation.sh CONFIG.yaml"
