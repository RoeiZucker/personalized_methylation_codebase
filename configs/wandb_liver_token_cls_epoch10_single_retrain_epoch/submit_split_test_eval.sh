#!/usr/bin/env bash
set -euo pipefail

sbatch -A tomhope --mem=32g -c4 --time=12:00:00 --wrap="/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_split_test_eval_params.sh --skip-existing --overwrite"
