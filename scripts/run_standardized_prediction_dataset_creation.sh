#!/usr/bin/env bash
set -euo pipefail

. /cs/usr/roeizucker/new_storage/new_python_env/bin/activate

REPO=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code
python3 "$REPO/scripts/create_standardized_prediction_datasets.py" "$@"

# Example, inspect the common chr12 split on a Phoenix compute node:
# sbatch --mem=32g -c4 --time=12:00:00 --killable --requeue --wrap="/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_standardized_prediction_dataset_creation.sh --dry-run"
#
# Example, create or validate the versioned chr12 datasets on Phoenix:
# sbatch --mem=64g -c8 --time=1-00 --killable --requeue --wrap="/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_standardized_prediction_dataset_creation.sh --skip-existing"
#
# Example, replace an incomplete versioned build on Phoenix:
# sbatch --mem=64g -c8 --time=1-00 --killable --requeue --wrap="/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_standardized_prediction_dataset_creation.sh --overwrite"
