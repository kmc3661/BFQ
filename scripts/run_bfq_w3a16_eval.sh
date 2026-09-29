#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

MODEL=${MODEL:-internvl2}
MODEL_ARGS=${MODEL_ARGS:-pretrained=OpenGVLab/InternVL2-8B}
TASKS=${TASKS:-mmmu_val}
BATCH_SIZE=${BATCH_SIZE:-1}
W_GROUP=${W_GROUP:-128}
SCALE_PATH=${SCALE_PATH:-outputs/bfq/${MODEL}_w3a16.pt}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/eval/${MODEL}_w3a16}
LOG_SAMPLES_SUFFIX=${LOG_SAMPLES_SUFFIX:-w3a16}
EXTRA_ARGS=${EXTRA_ARGS:-}

${PYTHON:-python} -W ignore main.py   --model "$MODEL"   --model_args "$MODEL_ARGS"   --tasks "$TASKS"   --batch_size "$BATCH_SIZE"   --method bfq   --pseudo_quant   --w_bit 3   --a_bit 16   --w_group "$W_GROUP"   --log_samples   --log_samples_suffix "$LOG_SAMPLES_SUFFIX"   --output_path "$OUTPUT_PATH"   --scale_path "$SCALE_PATH"   $EXTRA_ARGS
