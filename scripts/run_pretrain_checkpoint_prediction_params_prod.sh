#!/usr/bin/env bash
set -euo pipefail

. /home/users/roeizucker/tests/my_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/run_pretrain_checkpoint_prediction.py" "$@"
