#!/usr/bin/env bash
set -euo pipefail

. /sci/nosnap/michall/roeizucker/new_python_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID

REPO=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code
python3 "$REPO/training_script.py" "$1"

# Example, submit an epoch-4 resume training job on Phoenix:
# sbatch --mem=90g -c15 --gres=gg:g4:1 --time=5-23 --killable --requeue --constraint="firefoot|cyril" --job-name="ret-epoch4:Z0000044H:Liver" --wrap="/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_training_params_phoenix.sh /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/downsampled_runs/epoch4_resume_phoenix/_token_cls_Liver-Hepatocytes_kmer/Z0000044H_epoch-10-step-849740_downsampled_ratio_0p25_retrain_token_cls_Liver-Hepatocytes_kmer_lr_3e-05_bs_14_seq_5400.yaml"
