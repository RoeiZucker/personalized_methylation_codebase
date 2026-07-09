#!/usr/bin/env bash
set -euo pipefail

. /cs/usr/roeizucker/new_storage/new_python_env/bin/activate

cd /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code
python3 scripts/split_test_datasets_to_eval_and_test.py "$@"

# how to run:
# sbatch -A tomhope --mem=32g -c4 --time=12:00:00 --wrap="/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_split_test_eval_params.sh"
#
# dry run:
# sbatch -A tomhope --mem=8g -c1 --time=00:30:00 --wrap="/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_split_test_eval_params.sh --dry-run"
