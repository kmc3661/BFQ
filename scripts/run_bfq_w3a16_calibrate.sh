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
BFQ_POLICY_SAMPLES=${BFQ_POLICY_SAMPLES:-64}
W_GROUP=${W_GROUP:-128}
SCALE_PATH=${SCALE_PATH:-outputs/bfq/${MODEL}_w3a16.pt}
BFQ_POLICY_DIR=${BFQ_POLICY_DIR:-$(dirname "$SCALE_PATH")/$(basename "${SCALE_PATH%.*}")_auto_rule}
EXTRA_ARGS=${EXTRA_ARGS:-}

ANALYSIS_DIR="$BFQ_POLICY_DIR/quant_effect"
mkdir -p "$BFQ_POLICY_DIR"

"${PYTHON:-python}" "$REPO_ROOT/scripts/analyze_calib_quantization_effect.py"   --model "$MODEL"   --model_args "$MODEL_ARGS"   --data_json "$DATA_PATH"   --image_root "$IMAGE_FOLDER"   --n_samples "$BFQ_POLICY_SAMPLES"   --output_dir "$ANALYSIS_DIR"   --w_bit 3   --act_a_bit 8   --w_group "$W_GROUP"   --components weight

QUANT_MODE=w3a16 RUNNER="$REPO_ROOT/scripts/run_bfq_calibrate_core.sh" QUANT_EFFECT_SUMMARY="$ANALYSIS_DIR/summary.csv" LOG_ROOT="$BFQ_POLICY_DIR" MODEL="$MODEL" MODEL_ARGS="$MODEL_ARGS" CALIB_DATA="$CALIB_DATA" DATA_PATH="$DATA_PATH" IMAGE_FOLDER="$IMAGE_FOLDER" TEXT_DATA_PATH="$TEXT_DATA_PATH" N_SAMPLES="$N_SAMPLES" W_GROUP="$W_GROUP" W_BIT=3 A_BIT=16 USE_DISTORT=0 SCALE_PATH="$SCALE_PATH" EXTRA_ARGS="$EXTRA_ARGS" bash "$REPO_ROOT/scripts/run_bfq_auto_rule_existing_runner.sh"
