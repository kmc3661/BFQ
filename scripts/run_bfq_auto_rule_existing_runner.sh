#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

RUNNER="${RUNNER:?set RUNNER}"
QUANT_EFFECT_SUMMARY="${QUANT_EFFECT_SUMMARY:?set QUANT_EFFECT_SUMMARY}"
QUANT_MODE="${QUANT_MODE:?set QUANT_MODE to w3a16|w4a8}"
LOG_ROOT="${LOG_ROOT:?set LOG_ROOT}"
TASKS_CSV="${TASKS_CSV:-mmmu_val}"
FORCE_RECALIB="${FORCE_RECALIB:-0}"

mkdir -p "$LOG_ROOT"

AUTO_CASE="auto_rule"
CASE_DIR="$LOG_ROOT/$AUTO_CASE"
mkdir -p "$CASE_DIR"

AUTO_ENV_FILE="$CASE_DIR/auto_rule_env.txt"
AUTO_JSON_FILE="$CASE_DIR/auto_rule_stats.json"
POLICY_JSON="$CASE_DIR/policy_override.json"
CASE_SUMMARY="$CASE_DIR/summary.csv"
CASE_SETTINGS="$CASE_DIR/settings.txt"
CASE_SCALE="${SCALE_PATH:-$CASE_DIR/scale.pt}"

[[ -f "$QUANT_EFFECT_SUMMARY" ]] || { echo "[error] QUANT_EFFECT_SUMMARY not found: $QUANT_EFFECT_SUMMARY" >&2; exit 1; }

"$PYTHON_BIN" "$ROOT/scripts/infer_bfq_rule_hparams.py" \
  --summary_csv "$QUANT_EFFECT_SUMMARY" \
  --quant_mode "$QUANT_MODE" \
  --format env > "$AUTO_ENV_FILE"

"$PYTHON_BIN" "$ROOT/scripts/infer_bfq_rule_hparams.py" \
  --summary_csv "$QUANT_EFFECT_SUMMARY" \
  --quant_mode "$QUANT_MODE" \
  --format json > "$AUTO_JSON_FILE"

set -a
source "$AUTO_ENV_FILE"
set +a

export GLMI_BETA="${GLMI_BETA:-0.5}"
export GLMI_N_GRID_MAX="${GLMI_N_GRID_MAX:-48}"
export GLMI_BUDGET_CONFLICT_ALPHA="${GLMI_BUDGET_CONFLICT_ALPHA:-0.5}"
export GLMI_BUDGET_TEMPERATURE="${GLMI_BUDGET_TEMPERATURE:-8.0}"
export GLMI_AGGRESSIVENESS_TEMPERATURE="${GLMI_AGGRESSIVENESS_TEMPERATURE:-3.0}"
export GLMI_RATIO_UPPER_MIN="${GLMI_RATIO_UPPER_MIN:-0.5}"
export GLMI_RATIO_UPPER_MAX="${GLMI_RATIO_UPPER_MAX:-1.0}"
export GLMI_WEIGHT_SIGNED_CLIP_PERCENTILE="${GLMI_WEIGHT_SIGNED_CLIP_PERCENTILE:-100}"
export GLMI_BUDGET_BONUS_LOW_PERCENTILE="${GLMI_BUDGET_BONUS_LOW_PERCENTILE:-50}"
export GLMI_BUDGET_BONUS_THRESHOLD_MODE="${GLMI_BUDGET_BONUS_THRESHOLD_MODE:-percentile}"
export GLMI_BUDGET_BONUS_THRESHOLD_ALPHA="${GLMI_BUDGET_BONUS_THRESHOLD_ALPHA:-1.0}"
export GLMI_BUDGET_PENALTY_LOW_PERCENTILE="${GLMI_BUDGET_PENALTY_LOW_PERCENTILE:-5.0}"
export GLMI_BUDGET_PENALTY_HIGH_PERCENTILE="${GLMI_BUDGET_PENALTY_HIGH_PERCENTILE:-20.0}"
export GLMI_BUDGET_PENALTY_GAMMA="${GLMI_BUDGET_PENALTY_GAMMA:-2.0}"
export GLMI_CONFLICT_MODE="${GLMI_CONFLICT_MODE:-shared}"
export GLMI_BUDGET_TARGET_MODE="${GLMI_BUDGET_TARGET_MODE:-both}"
export GLMI_FIXED_BUDGET="${GLMI_FIXED_BUDGET:-0}"
export GLMI_FIXED_AGGRESSIVENESS="${GLMI_FIXED_AGGRESSIVENESS:-1}"
export GLMI_UPPER_TAIL_BUDGET="${GLMI_UPPER_TAIL_BUDGET:-1}"
export GLMI_LOWER_TAIL_PENALTY="${GLMI_LOWER_TAIL_PENALTY:-0}"
export GLMI_POLICY_N_SAMPLES="${GLMI_POLICY_N_SAMPLES:-64}"

if [[ "$QUANT_MODE" == "w3a16" ]]; then
  export GLMI_MODALITY_RATIO_MIN=1.0
  export GLMI_MODALITY_RATIO_MAX=1.0
else
  export GLMI_MODALITY_RATIO_MIN="${GLMI_MODALITY_RATIO_MIN:-0.25}"
  export GLMI_MODALITY_RATIO_MAX="${GLMI_MODALITY_RATIO_MAX:-4.0}"
fi

(
  cd "$ROOT"
  policy_cmd=(
    "$PYTHON_BIN" "$ROOT/scripts/build_bfq_policy_from_summary.py"
    --summary_csv "$QUANT_EFFECT_SUMMARY" \
    --quant_mode "$QUANT_MODE" \
    --output_json "$POLICY_JSON" \
    --glmi_beta "$GLMI_BETA" \
    --glmi_n_grid_max "$GLMI_N_GRID_MAX" \
    --glmi_n_grid_target_mean "$GLMI_N_GRID_TARGET_MEAN" \
    --glmi_budget_conflict_alpha "$GLMI_BUDGET_CONFLICT_ALPHA" \
    --glmi_budget_temperature "$GLMI_BUDGET_TEMPERATURE" \
    --glmi_aggressiveness_temperature "$GLMI_AGGRESSIVENESS_TEMPERATURE" \
    --glmi_ratio_upper_min "$GLMI_RATIO_UPPER_MIN" \
    --glmi_ratio_upper_max "$GLMI_RATIO_UPPER_MAX" \
    --glmi_modality_ratio_min "$GLMI_MODALITY_RATIO_MIN" \
    --glmi_modality_ratio_max "$GLMI_MODALITY_RATIO_MAX" \
    --glmi_weight_signed_clip_percentile "$GLMI_WEIGHT_SIGNED_CLIP_PERCENTILE" \
    --glmi_budget_base_grid "$GLMI_BUDGET_BASE_GRID" \
    --glmi_budget_bonus_low_percentile "$GLMI_BUDGET_BONUS_LOW_PERCENTILE" \
    --glmi_budget_bonus_high_percentile "$GLMI_BUDGET_BONUS_HIGH_PERCENTILE" \
    --glmi_budget_bonus_gamma "$GLMI_BUDGET_BONUS_GAMMA" \
    --glmi_budget_bonus_threshold_mode "$GLMI_BUDGET_BONUS_THRESHOLD_MODE" \
    --glmi_budget_bonus_threshold_alpha "$GLMI_BUDGET_BONUS_THRESHOLD_ALPHA" \
    --glmi_budget_penalty_low_percentile "$GLMI_BUDGET_PENALTY_LOW_PERCENTILE" \
    --glmi_budget_penalty_high_percentile "$GLMI_BUDGET_PENALTY_HIGH_PERCENTILE" \
    --glmi_budget_penalty_gamma "$GLMI_BUDGET_PENALTY_GAMMA" \
    --glmi_conflict_mode "$GLMI_CONFLICT_MODE" \
    --glmi_budget_target_mode "$GLMI_BUDGET_TARGET_MODE"
  )
  [[ "$GLMI_FIXED_BUDGET" == "1" ]] && policy_cmd+=(--glmi_fixed_budget)
  [[ "$GLMI_FIXED_AGGRESSIVENESS" == "1" ]] && policy_cmd+=(--glmi_fixed_aggressiveness)
  [[ "$GLMI_UPPER_TAIL_BUDGET" == "1" ]] && policy_cmd+=(--glmi_upper_tail_budget)
  [[ "$GLMI_LOWER_TAIL_PENALTY" == "1" ]] && policy_cmd+=(--glmi_lower_tail_penalty)
  PYTHONPATH="$ROOT:${PYTHONPATH:-}" "${policy_cmd[@]}"
)

echo "[bfq-auto-rule] summary: $QUANT_EFFECT_SUMMARY"
echo "[bfq-auto-rule] env: $AUTO_ENV_FILE"
echo "[bfq-auto-rule] json: $AUTO_JSON_FILE"
echo "[bfq-auto-rule] policy: $POLICY_JSON"

TASKS_CSV="$TASKS_CSV" \
LOG_DIR="$CASE_DIR" \
SUMMARY_CSV="$CASE_SUMMARY" \
SETTINGS_TXT="$CASE_SETTINGS" \
SCALE_PATH="$CASE_SCALE" \
FORCE_RECALIB="$FORCE_RECALIB" \
GLMI_POLICY_OVERRIDE_PATH="$POLICY_JSON" \
bash "$RUNNER"
