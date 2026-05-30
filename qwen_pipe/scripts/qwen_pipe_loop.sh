#!/usr/bin/env bash
set -euo pipefail

PIPE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEBUG_ROOT="${DEBUG_DISTRIBUTION_ROOT:-$PIPE_ROOT/../debug_distribution}"
DEBUG_ROOT="$(cd "$DEBUG_ROOT" && pwd)"

ACTION="${1:-all}"
if [[ "$ACTION" == "smoke" ]]; then
  MAX_SAMPLES="${MAX_SAMPLES:-3}"
  ACTION="all"
elif [[ "$ACTION" == "decision" ]]; then
  MAX_SAMPLES="${MAX_SAMPLES:-60}"
  ACTION="all"
elif [[ "$ACTION" == "full" ]]; then
  MAX_SAMPLES="${MAX_SAMPLES:-0}"
  ACTION="all"
else
  MAX_SAMPLES="${MAX_SAMPLES:-3}"
fi

PYTHON_BIN="${PYTHON:-python3}"
PIPELINE="${PIPELINE:-staged}"
MODEL="${MODEL:-qwen3.6-35b-a3b}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
EXP_NAME="${EXP_NAME:-qwen_pipe_$(date +%Y%m%d_%H%M%S)}"
LEDGER="${LEDGER:-outputs/tuning/experiments.jsonl}"
LEDGER_SCRIPT="${TUNING_LEDGER_SCRIPT:-/Users/penpen/.codex/skills/paper-guided-algorithm-tuning/scripts/tuning_ledger.py}"
DECISION="${DECISION:-candidate}"
CHANGES="${CHANGES:-Qwen-pipe tuning iteration}"
NOTES="${NOTES:-}"
PAPER_URL="${PAPER_URL:-DocShield/CCT proxy}"
PRIMARY_METRIC="${PRIMARY_METRIC:-S_Fin_proxy}"

if [[ "$MAX_SAMPLES" == "0" ]]; then
  TAG="300"
else
  TAG="$MAX_SAMPLES"
fi

RAW_JSONL="${RAW_JSONL:-outputs/raw/${EXP_NAME}_${TAG}.jsonl}"
OUT_JSON="${OUT_JSON:-outputs/eval/${EXP_NAME}_${TAG}.json}"
OUT_CSV="${OUT_CSV:-outputs/eval/${EXP_NAME}_${TAG}.csv}"

abs_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$PIPE_ROOT" "$path"
  fi
}

RAW_JSONL_ABS="$(abs_path "$RAW_JSONL")"
OUT_JSON_ABS="$(abs_path "$OUT_JSON")"
OUT_CSV_ABS="$(abs_path "$OUT_CSV")"
LEDGER_ABS="$(abs_path "$LEDGER")"

run_command=$(
  printf 'cd %q && PYTHON=%q PIPELINE=%q MODEL=%q MAX_SAMPLES=%q NUM_WORKERS=%q MAX_TOKENS=%q OUTPUT_JSONL=%q ./run_api_debug.sh && PYTHON=%q PIPELINE=%q MAX_SAMPLES=%q RAW_JSONL=%q OUT_JSON=%q OUT_CSV=%q ./eval_debug.sh' \
    "$DEBUG_ROOT" "$PYTHON_BIN" "$PIPELINE" "$MODEL" "$MAX_SAMPLES" "$NUM_WORKERS" "$MAX_TOKENS" "$RAW_JSONL_ABS" \
    "$PYTHON_BIN" "$PIPELINE" "$MAX_SAMPLES" "$RAW_JSONL_ABS" "$OUT_JSON_ABS" "$OUT_CSV_ABS"
)
RUN_COMMAND="${RUN_COMMAND:-$run_command}"

show_config() {
  cat <<EOF
Qwen-pipe config:
  pipe_root:    $PIPE_ROOT
  debug_root:   $DEBUG_ROOT
  action:       $ACTION
  experiment:   $EXP_NAME
  pipeline:     $PIPELINE
  model:        $MODEL
  max_samples:  $MAX_SAMPLES
  workers:      $NUM_WORKERS
  raw_jsonl:    $RAW_JSONL_ABS
  eval_json:    $OUT_JSON_ABS
  eval_csv:     $OUT_CSV_ABS
  ledger:       $LEDGER_ABS
EOF
}

preflight() {
  [[ -f "$DEBUG_ROOT/run_api_debug.sh" ]] || { echo "missing $DEBUG_ROOT/run_api_debug.sh" >&2; exit 2; }
  [[ -f "$DEBUG_ROOT/eval_debug.sh" ]] || { echo "missing $DEBUG_ROOT/eval_debug.sh" >&2; exit 2; }
  [[ -f "$LEDGER_SCRIPT" ]] || { echo "missing tuning ledger script: $LEDGER_SCRIPT" >&2; exit 2; }
}

run_inference() {
  preflight
  mkdir -p "$(dirname "$RAW_JSONL_ABS")"
  show_config
  (
    cd "$DEBUG_ROOT"
    PYTHON="$PYTHON_BIN" \
    PIPELINE="$PIPELINE" \
    MODEL="$MODEL" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    NUM_WORKERS="$NUM_WORKERS" \
    MAX_TOKENS="$MAX_TOKENS" \
    OUTPUT_JSONL="$RAW_JSONL_ABS" \
      ./run_api_debug.sh
  )
}

run_eval() {
  preflight
  [[ -f "$RAW_JSONL_ABS" ]] || { echo "raw output not found: $RAW_JSONL_ABS" >&2; exit 2; }
  mkdir -p "$(dirname "$OUT_JSON_ABS")" "$(dirname "$OUT_CSV_ABS")"
  (
    cd "$DEBUG_ROOT"
    PYTHON="$PYTHON_BIN" \
    PIPELINE="$PIPELINE" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    RAW_JSONL="$RAW_JSONL_ABS" \
    OUT_JSON="$OUT_JSON_ABS" \
    OUT_CSV="$OUT_CSV_ABS" \
      ./eval_debug.sh
  )
}

record_run() {
  preflight
  [[ -f "$OUT_JSON_ABS" ]] || { echo "eval json not found: $OUT_JSON_ABS" >&2; exit 2; }
  mkdir -p "$(dirname "$LEDGER_ABS")"
  "$PYTHON_BIN" "$LEDGER_SCRIPT" record \
    --ledger "$LEDGER_ABS" \
    --name "$EXP_NAME" \
    --eval-json "$OUT_JSON_ABS" \
    --paper-url "$PAPER_URL" \
    --command "$RUN_COMMAND" \
    --changes "$CHANGES" \
    --notes "$NOTES" \
    --decision "$DECISION"
}

compare_runs() {
  preflight
  "$PYTHON_BIN" "$LEDGER_SCRIPT" compare \
    --ledger "$LEDGER_ABS" \
    --primary "$PRIMARY_METRIC" \
    --last "${LAST:-0}"
}

list_errors() {
  preflight
  local issue="${2:-${ISSUE:-forged_without_boxes}}"
  [[ -f "$OUT_JSON_ABS" ]] || { echo "eval json not found: $OUT_JSON_ABS" >&2; exit 2; }
  "$PYTHON_BIN" "$LEDGER_SCRIPT" errors \
    --eval-json "$OUT_JSON_ABS" \
    --issue "$issue" \
    --limit "${LIMIT:-12}"
}

record_baseline_once() {
  local name="$1"
  local eval_json="$2"
  local notes="$3"
  if [[ -f "$LEDGER_ABS" ]] && grep -q "\"name\": \"$name\"" "$LEDGER_ABS"; then
    echo "baseline already recorded: $name"
    return
  fi
  EXP_NAME="$name" \
  OUT_JSON="$eval_json" \
  RAW_JSONL="reference-only" \
  LEDGER="$LEDGER" \
  DEBUG_DISTRIBUTION_ROOT="$DEBUG_ROOT" \
  CHANGES="Baseline lock from debug_distribution." \
  NOTES="$notes" \
  DECISION="keep" \
  RUN_COMMAND="reference eval from $eval_json; original history is in debug_distribution/outputs/tuning/experiments.jsonl" \
    "$0" record
}

seed_baseline() {
  preflight
  mkdir -p "$(dirname "$LEDGER_ABS")"
  record_baseline_once \
    "qwen_pipe_baseline_v28_60" \
    "$DEBUG_ROOT/outputs/eval/staged_cct_v28_visual_artifact_reviewer_60.json" \
    "Existing local eval: S_Fin_proxy=0.6532, S_Det=0.9500, S_Loc=0.0434; remaining issue forged_without_boxes=3."
  record_baseline_once \
    "qwen_pipe_baseline_v28_300" \
    "$DEBUG_ROOT/outputs/eval/staged_cct_v28_visual_artifact_reviewer_300.json" \
    "Existing local eval: S_Fin_proxy=0.6012, S_Det=0.8167, S_Loc=0.0303; remaining issues authentic_predicted_with_boxes=37 and forged_without_boxes=18."
}

case "$ACTION" in
  run)
    run_inference
    ;;
  eval)
    run_eval
    ;;
  record)
    record_run
    ;;
  compare)
    compare_runs
    ;;
  errors)
    list_errors "$@"
    ;;
  config)
    show_config
    ;;
  seed-baseline)
    seed_baseline
    ;;
  all)
    run_inference
    run_eval
    record_run
    compare_runs
    ;;
  *)
    cat >&2 <<'EOF'
Usage:
  scripts/qwen_pipe_loop.sh [all|smoke|decision|full|run|eval|record|compare|errors|config|seed-baseline]

Common environment variables:
  EXP_NAME, MAX_SAMPLES, MODEL, NUM_WORKERS, PIPELINE, RAW_JSONL, OUT_JSON,
  OUT_CSV, LEDGER, CHANGES, NOTES, DECISION, ISSUE, LIMIT,
  DEBUG_DISTRIBUTION_ROOT.
EOF
    exit 2
    ;;
esac
