#!/usr/bin/env bash
set -euo pipefail

. /sci/nosnap/michall/roeizucker/new_python_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID

PROJECT="${PROJECT:-/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code}"
BASE_CONFIG="${BASE_CONFIG:-$PROJECT/configs/wandb_classification_full_training_r3_10k_limited_run_no_start_eval.yaml}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-/sci/labs/michall/roeizucker/roei_test/token_cls_datasets_downsampled/_token_cls_Liver-Hepatocytes_kmer/Z000000R3_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train_ratio_1p0}"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-unknown}}"
MAX_STEPS="${MAX_STEPS:-20}"

echo "GPU probe for batch_size=14"
echo "GPU_COUNT=$GPU_COUNT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python3 "$PROJECT/src/wandb_sweep_training.py" \
  --base-config "$BASE_CONFIG" \
  --sample-name "R3-bs14-gpu${GPU_COUNT}-probe" \
  --wandb-run-id "r3-bs14-gpu${GPU_COUNT}-probe" \
  --project "${WANDB_PROJECT:-personalized-methylation-r3-gpu-probe}" \
  --group "${WANDB_GROUP:-r3-bs14-gpu-probe}" \
  --mode "${WANDB_MODE:-online}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --batch-size 14 \
  --max-steps "$MAX_STEPS" \
  --eval-steps "${EVAL_STEPS:-1000000}" \
  --logging-steps "${LOGGING_STEPS:-1}" \
  --no-eval-on-start \
  --train-dataset-path "$TRAIN_DATASET_PATH"

# Example, probe one GPU:
# sbatch -A tomhope --mem=45g -c15 --gres=gg:g4:1 --time=02:00:00 --killable --requeue scripts/run_wandb_bs14_gpu_probe.sh
#
# Example, probe two GPUs:
# sbatch -A tomhope --mem=45g -c15 --gres=gg:g4:2 --time=02:00:00 --killable --requeue scripts/run_wandb_bs14_gpu_probe.sh
#
# Example, probe three GPUs:
# sbatch -A tomhope --mem=45g -c15 --gres=gg:g4:3 --time=02:00:00 --killable --requeue scripts/run_wandb_bs14_gpu_probe.sh
