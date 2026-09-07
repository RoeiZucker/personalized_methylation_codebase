#!/usr/bin/env bash
set -euo pipefail

. /cs/usr/roeizucker/new_storage/new_python_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID

REPO=/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code

python3 "$REPO/scripts/run_pretrain_checkpoint_prediction.py" "$@"

# Example, predict one checkpoint on eval:
# /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_downsample_checkpoint_prediction_params.sh --config /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/downsampled_runs/prediction_chr12_standardized_v1/Z0000044M_downsampled_ratio_0p25_lr_3e-05_chr12_standardized_split_v1_predict_eval.yaml --checkpoint-path /sci/labs/michall/roeizucker/roei_test/token_cls_models_downsampled/_token_cls_Liver-Hepatocytes_kmer/Z0000044M_epoch-10-step-849900_downsampled_ratio_0p25_retrain_token_cls_Liver-Hepatocytes_kmer_lr_3e-05_bs_14_seq_5400/checkpoint-15676 --output-path /sci/labs/michall/roeizucker/roei_test/token_cls_models_downsampled/_token_cls_Liver-Hepatocytes_kmer/Z0000044M_epoch-10-step-849900_downsampled_ratio_0p25_retrain_token_cls_Liver-Hepatocytes_kmer_lr_3e-05_bs_14_seq_5400/checkpoint-15676/chr12_standardized_split_v1/eval_predictions.csv.gitbackup --overwrite
#
# Example, submit one checkpoint prediction to Slurm:
# sbatch --mem=45g -c15 --gres=gg:g4:1 --time=5-23 --killable --requeue --constraint="firefoot|cyril" --wrap="/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_downsample_checkpoint_prediction_params.sh --config CONFIG.yaml --checkpoint-path CHECKPOINT_DIR --output-path OUTPUT.csv.gitbackup --overwrite"
