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
BFQ_POLICY_SAMPLES=${BFQ_POLICY_SAMPLES:-$N_SAMPLES}
CALIB_SEED=${CALIB_SEED:-42}
ANALYSIS_MICRO_BATCH_SIZE=${ANALYSIS_MICRO_BATCH_SIZE:-4}
W_GROUP=${W_GROUP:-128}
SCALE_PATH=${SCALE_PATH:-outputs/bfq/${MODEL}_w4a8.pt}
BFQ_POLICY_DIR=${BFQ_POLICY_DIR:-$(dirname "$SCALE_PATH")/$(basename "${SCALE_PATH%.*}")_auto_rule}
EXTRA_ARGS=${EXTRA_ARGS:-}

ANALYSIS_DIR="$BFQ_POLICY_DIR/quant_effect"
CALIB_MANIFEST="$BFQ_POLICY_DIR/calibration_samples.json"
[[ "$CALIB_DATA" == "coco" ]] || { echo "BFQ calibration wrappers require CALIB_DATA=coco" >&2; exit 1; }
[[ "$BFQ_POLICY_SAMPLES" == "$N_SAMPLES" ]] || { echo "QE and CWE sample counts must match" >&2; exit 1; }
[[ ! -e "$SCALE_PATH" ]] || { echo "Scale cache already exists; choose a new SCALE_PATH: $SCALE_PATH" >&2; exit 1; }
mkdir -p "$BFQ_POLICY_DIR"

"${PYTHON:-python}" "$REPO_ROOT/scripts/prepare_bfq_calibration_manifest.py" \
  --source "$DATA_PATH" --output "$CALIB_MANIFEST" \
  --n_samples "$N_SAMPLES" --seed "$CALIB_SEED"

"${PYTHON:-python}" "$REPO_ROOT/scripts/analyze_calib_quantization_effect.py" \
  --model "$MODEL" --model_args "$MODEL_ARGS" \
  --data_json "$CALIB_MANIFEST" --image_root "$IMAGE_FOLDER" \
  --n_samples "$N_SAMPLES" --no_shuffle --micro_batch_size "$ANALYSIS_MICRO_BATCH_SIZE" --output_dir "$ANALYSIS_DIR" \
  --w_bit 4 --act_a_bit 8 --w_group "$W_GROUP" --components weight,vision,text

QUANT_MODE=w4a8 RUNNER="$REPO_ROOT/scripts/run_bfq_calibrate_core.sh" \
QUANT_EFFECT_SUMMARY="$ANALYSIS_DIR/summary.csv" LOG_ROOT="$BFQ_POLICY_DIR" \
MODEL="$MODEL" MODEL_ARGS="$MODEL_ARGS" CALIB_DATA="$CALIB_DATA" \
DATA_PATH="$CALIB_MANIFEST" IMAGE_FOLDER="$IMAGE_FOLDER" TEXT_DATA_PATH="$TEXT_DATA_PATH" \
N_SAMPLES="$N_SAMPLES" BFQ_CALIB_NO_SHUFFLE=1 W_GROUP="$W_GROUP" \
W_BIT=4 A_BIT=8 USE_DISTORT=0 SCALE_PATH="$SCALE_PATH" EXTRA_ARGS="$EXTRA_ARGS" \
bash "$REPO_ROOT/scripts/run_bfq_auto_rule_existing_runner.sh"
