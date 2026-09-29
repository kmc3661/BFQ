#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

MODEL=${MODEL:-internvl2}
MODEL_ARGS=${MODEL_ARGS:-pretrained=OpenGVLab/InternVL2-8B}
CALIB_DATA=${CALIB_DATA:-coco}
DATA_PATH=${DATA_PATH:-data/path/calibration.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/path/images}
TEXT_DATA_PATH=${TEXT_DATA_PATH:-}
N_SAMPLES=${N_SAMPLES:-64}
W_GROUP=${W_GROUP:-128}
W_BIT=${W_BIT:?set W_BIT}
A_BIT=${A_BIT:?set A_BIT}
USE_DISTORT=${USE_DISTORT:-0}
BFQ_CALIB_NO_SHUFFLE=${BFQ_CALIB_NO_SHUFFLE:-0}
SCALE_PATH=${SCALE_PATH:?set SCALE_PATH}
BFQ_POLICY_OVERRIDE_PATH=${BFQ_POLICY_OVERRIDE_PATH:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

CMD=(
  "${PYTHON:-python}" -W ignore main_quant.py
  --model "$MODEL"
  --model_args "$MODEL_ARGS"
  --calib_data "$CALIB_DATA"
  --data_path "$DATA_PATH"
  --image_folder "$IMAGE_FOLDER"
  --n_samples "$N_SAMPLES"
  --method bfq
  --run_process
  --w_bit "$W_BIT"
  --a_bit "$A_BIT"
  --w_group "$W_GROUP"
  --scale_path "$SCALE_PATH"
)
[[ -n "$TEXT_DATA_PATH" ]] && CMD+=(--text_data_path "$TEXT_DATA_PATH")
[[ -n "$BFQ_POLICY_OVERRIDE_PATH" ]] && CMD+=(--bfq_policy_override_path "$BFQ_POLICY_OVERRIDE_PATH")
[[ "$BFQ_CALIB_NO_SHUFFLE" == "1" ]] && CMD+=(--calib_no_shuffle)
[[ "$USE_DISTORT" == "1" ]] && CMD+=(--distort)
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=( $EXTRA_ARGS )
  CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}"
