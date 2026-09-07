#!/usr/bin/env bash
#SBATCH --account=tomhope
#SBATCH --partition=medium
#SBATCH --mem=45g
#SBATCH -c 15
#SBATCH --gres=gg:g4:1
#SBATCH --time=5-23
#SBATCH --constraint=firefoot|cyril
#SBATCH --job-name=new-epoch-pred-th

set -euo pipefail

WORKER=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_new_epoch_prediction_array.sh
exec bash "$WORKER"

# Example, run five selected prediction tasks concurrently under tomhope:
# sbatch --array=0-63%5 scripts/run_new_epoch_prediction_array_tomhope.sh
