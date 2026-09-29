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

export BFQ_BETA="${BFQ_BETA:-0.5}"
export BFQ_N_GRID_MAX="${BFQ_N_GRID_MAX:-48}"
export BFQ_BUDGET_CONFLICT_ALPHA="${BFQ_BUDGET_CONFLICT_ALPHA:-0.5}"
export BFQ_BUDGET_TEMPERATURE="${BFQ_BUDGET_TEMPERATURE:-8.0}"
export BFQ_AGGRESSIVENESS_TEMPERATURE="${BFQ_AGGRESSIVENESS_TEMPERATURE:-3.0}"
export BFQ_RATIO_UPPER_MIN="${BFQ_RATIO_UPPER_MIN:-0.5}"
export BFQ_RATIO_UPPER_MAX="${BFQ_RATIO_UPPER_MAX:-1.0}"
export BFQ_WEIGHT_SIGNED_CLIP_PERCENTILE="${BFQ_WEIGHT_SIGNED_CLIP_PERCENTILE:-100}"
export BFQ_BUDGET_BONUS_LOW_PERCENTILE="${BFQ_BUDGET_BONUS_LOW_PERCENTILE:-50}"
export BFQ_BUDGET_BONUS_THRESHOLD_MODE="${BFQ_BUDGET_BONUS_THRESHOLD_MODE:-percentile}"
export BFQ_BUDGET_BONUS_THRESHOLD_ALPHA="${BFQ_BUDGET_BONUS_THRESHOLD_ALPHA:-1.0}"
export BFQ_BUDGET_PENALTY_LOW_PERCENTILE="${BFQ_BUDGET_PENALTY_LOW_PERCENTILE:-5.0}"
export BFQ_BUDGET_PENALTY_HIGH_PERCENTILE="${BFQ_BUDGET_PENALTY_HIGH_PERCENTILE:-20.0}"
export BFQ_BUDGET_PENALTY_GAMMA="${BFQ_BUDGET_PENALTY_GAMMA:-2.0}"
export BFQ_CONFLICT_MODE="${BFQ_CONFLICT_MODE:-shared}"
export BFQ_BUDGET_TARGET_MODE="${BFQ_BUDGET_TARGET_MODE:-both}"
export BFQ_FIXED_BUDGET="${BFQ_FIXED_BUDGET:-0}"
export BFQ_FIXED_AGGRESSIVENESS="${BFQ_FIXED_AGGRESSIVENESS:-1}"
export BFQ_UPPER_TAIL_BUDGET="${BFQ_UPPER_TAIL_BUDGET:-1}"
export BFQ_LOWER_TAIL_PENALTY="${BFQ_LOWER_TAIL_PENALTY:-0}"
export BFQ_POLICY_N_SAMPLES="${BFQ_POLICY_N_SAMPLES:-64}"

if [[ "$QUANT_MODE" == "w3a16" ]]; then
  export BFQ_MODALITY_RATIO_MIN=1.0
  export BFQ_MODALITY_RATIO_MAX=1.0
else
  export BFQ_MODALITY_RATIO_MIN="${BFQ_MODALITY_RATIO_MIN:-0.25}"
  export BFQ_MODALITY_RATIO_MAX="${BFQ_MODALITY_RATIO_MAX:-4.0}"
fi

(
  cd "$ROOT"
  policy_cmd=(
    "$PYTHON_BIN" "$ROOT/scripts/build_bfq_policy_from_summary.py"
    --summary_csv "$QUANT_EFFECT_SUMMARY" \
    --quant_mode "$QUANT_MODE" \
    --output_json "$POLICY_JSON" \
    --bfq_beta "$BFQ_BETA" \
    --bfq_n_grid_max "$BFQ_N_GRID_MAX" \
    --bfq_n_grid_target_mean "$BFQ_N_GRID_TARGET_MEAN" \
    --bfq_budget_conflict_alpha "$BFQ_BUDGET_CONFLICT_ALPHA" \
    --bfq_budget_temperature "$BFQ_BUDGET_TEMPERATURE" \
    --bfq_aggressiveness_temperature "$BFQ_AGGRESSIVENESS_TEMPERATURE" \
    --bfq_ratio_upper_min "$BFQ_RATIO_UPPER_MIN" \
    --bfq_ratio_upper_max "$BFQ_RATIO_UPPER_MAX" \
    --bfq_modality_ratio_min "$BFQ_MODALITY_RATIO_MIN" \
    --bfq_modality_ratio_max "$BFQ_MODALITY_RATIO_MAX" \
    --bfq_weight_signed_clip_percentile "$BFQ_WEIGHT_SIGNED_CLIP_PERCENTILE" \
    --bfq_budget_base_grid "$BFQ_BUDGET_BASE_GRID" \
    --bfq_budget_bonus_low_percentile "$BFQ_BUDGET_BONUS_LOW_PERCENTILE" \
    --bfq_budget_bonus_high_percentile "$BFQ_BUDGET_BONUS_HIGH_PERCENTILE" \
    --bfq_budget_bonus_gamma "$BFQ_BUDGET_BONUS_GAMMA" \
    --bfq_budget_bonus_threshold_mode "$BFQ_BUDGET_BONUS_THRESHOLD_MODE" \
    --bfq_budget_bonus_threshold_alpha "$BFQ_BUDGET_BONUS_THRESHOLD_ALPHA" \
    --bfq_budget_penalty_low_percentile "$BFQ_BUDGET_PENALTY_LOW_PERCENTILE" \
    --bfq_budget_penalty_high_percentile "$BFQ_BUDGET_PENALTY_HIGH_PERCENTILE" \
    --bfq_budget_penalty_gamma "$BFQ_BUDGET_PENALTY_GAMMA" \
    --bfq_conflict_mode "$BFQ_CONFLICT_MODE" \
    --bfq_budget_target_mode "$BFQ_BUDGET_TARGET_MODE"
  )
  [[ "$BFQ_FIXED_BUDGET" == "1" ]] && policy_cmd+=(--bfq_fixed_budget)
  [[ "$BFQ_FIXED_AGGRESSIVENESS" == "1" ]] && policy_cmd+=(--bfq_fixed_aggressiveness)
  [[ "$BFQ_UPPER_TAIL_BUDGET" == "1" ]] && policy_cmd+=(--bfq_upper_tail_budget)
  [[ "$BFQ_LOWER_TAIL_PENALTY" == "1" ]] && policy_cmd+=(--bfq_lower_tail_penalty)
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
BFQ_POLICY_OVERRIDE_PATH="$POLICY_JSON" \
bash "$RUNNER"
