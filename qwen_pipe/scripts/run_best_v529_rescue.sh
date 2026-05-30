#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "$ROOT/../scripts/eval_competition_aligned.py" && -d "$ROOT/../data" ]]; then
  DEBUG_ROOT="${DEBUG_DISTRIBUTION_ROOT:-$ROOT/..}"
else
  DEBUG_ROOT="${DEBUG_DISTRIBUTION_ROOT:-$ROOT/../debug_distribution}"
fi
DEBUG_ROOT="$(cd "$DEBUG_ROOT" && pwd)"

PYTHON_BIN="${PYTHON:-$DEBUG_ROOT/.venv/bin/python}"

resolve_qwen_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$ROOT/$1" ;;
  esac
}

BASE_RAW="${BASE_RAW:-outputs/raw/qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.jsonl}"
BASE_EVAL="${BASE_EVAL:-outputs/eval/qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.json}"
OBS_CACHE="${OBS_CACHE:-outputs/cache/pair_observations/v518_v507_lowloc_t010_context_cl32.json}"
EMPTY_SELECTED="${EMPTY_SELECTED:-outputs/diagnostics/qwen_pipe_v521_empty_selected_context_diag.jsonl}"

V528_RAW="${V528_RAW:-outputs/raw/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_on_v507_300.jsonl}"
V528_SUMMARY="${V528_SUMMARY:-outputs/diagnostics/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_summary.json}"
V528_DIAG="${V528_DIAG:-outputs/diagnostics/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_diag.jsonl}"
V528_EVAL="${V528_EVAL:-outputs/eval/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_on_v507_300.json}"
V528_CSV="${V528_CSV:-outputs/eval/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_on_v507_300.csv}"

V529_RAW="${V529_RAW:-outputs/raw/qwen_pipe_v529_knn_append_precision_lowloc_t010_on_v528_300.jsonl}"
V529_SUMMARY="${V529_SUMMARY:-outputs/diagnostics/qwen_pipe_v529_knn_append_precision_lowloc_t010_summary.json}"
V529_DIAG="${V529_DIAG:-outputs/diagnostics/qwen_pipe_v529_knn_append_precision_lowloc_t010_diag.jsonl}"
V529_EVAL="${V529_EVAL:-outputs/eval/qwen_pipe_v529_knn_append_precision_lowloc_t010_on_v528_300.json}"
V529_CSV="${V529_CSV:-outputs/eval/qwen_pipe_v529_knn_append_precision_lowloc_t010_on_v528_300.csv}"

BASE_RAW_PATH="$(resolve_qwen_path "$BASE_RAW")"
BASE_EVAL_PATH="$(resolve_qwen_path "$BASE_EVAL")"
OBS_CACHE_PATH="$(resolve_qwen_path "$OBS_CACHE")"
EMPTY_SELECTED_PATH="$(resolve_qwen_path "$EMPTY_SELECTED")"
V528_RAW_PATH="$(resolve_qwen_path "$V528_RAW")"
V528_SUMMARY_PATH="$(resolve_qwen_path "$V528_SUMMARY")"
V528_DIAG_PATH="$(resolve_qwen_path "$V528_DIAG")"
V528_EVAL_PATH="$(resolve_qwen_path "$V528_EVAL")"
V528_CSV_PATH="$(resolve_qwen_path "$V528_CSV")"
V529_RAW_PATH="$(resolve_qwen_path "$V529_RAW")"
V529_SUMMARY_PATH="$(resolve_qwen_path "$V529_SUMMARY")"
V529_DIAG_PATH="$(resolve_qwen_path "$V529_DIAG")"
V529_EVAL_PATH="$(resolve_qwen_path "$V529_EVAL")"
V529_CSV_PATH="$(resolve_qwen_path "$V529_CSV")"

for required in "$BASE_RAW_PATH" "$BASE_EVAL_PATH" "$OBS_CACHE_PATH" "$EMPTY_SELECTED_PATH"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required cached artifact: $required" >&2
    echo "This script reproduces the final GT-blind rescue stage from local cached Qwen-pipe artifacts." >&2
    exit 2
  fi
done

mkdir -p "$ROOT/outputs/raw" "$ROOT/outputs/eval" "$ROOT/outputs/diagnostics"

cd "$ROOT"

"$PYTHON_BIN" scripts/qwen_hard_positive_knn_rescue.py \
  --base-raw-jsonl "$BASE_RAW_PATH" \
  --eval-json "$BASE_EVAL_PATH" \
  --observation-cache "$OBS_CACHE_PATH" \
  --previous-pair-jsonl "$EMPTY_SELECTED_PATH" \
  --output-jsonl "$V528_RAW_PATH" \
  --summary-json "$V528_SUMMARY_PATH" \
  --diag-jsonl "$V528_DIAG_PATH" \
  --debug-root "$DEBUG_ROOT" \
  --loc-threshold 0.10 \
  --min-positive-delta 0.0005 \
  --k 7 \
  --neg-weight 0.35 \
  --threshold-mode precision \
  --family-allowlist ocr,linegrid,evidence,row \
  --max-per-sample 1 \
  --sample-risk-gate

(
  cd "$DEBUG_ROOT"
  "$PYTHON_BIN" scripts/eval_competition_aligned.py \
    --gt-jsonl data/val_300.jsonl \
    --raw-jsonl "$V528_RAW_PATH" \
    --output "$V528_EVAL_PATH" \
    --output-csv "$V528_CSV_PATH" \
    --skip-bertscore \
    --allow-empty-loc \
    --include-samples
)

"$PYTHON_BIN" scripts/qwen_hard_positive_knn_rescue.py \
  --base-raw-jsonl "$V528_RAW_PATH" \
  --eval-json "$V528_EVAL_PATH" \
  --observation-cache "$OBS_CACHE_PATH" \
  --previous-pair-jsonl "$EMPTY_SELECTED_PATH" \
  --output-jsonl "$V529_RAW_PATH" \
  --summary-json "$V529_SUMMARY_PATH" \
  --diag-jsonl "$V529_DIAG_PATH" \
  --debug-root "$DEBUG_ROOT" \
  --loc-threshold 0.10 \
  --min-positive-delta 0.0005 \
  --k 7 \
  --neg-weight 0.35 \
  --threshold-mode precision \
  --action-mode append \
  --family-allowlist ocr,linegrid,evidence,row,token \
  --max-per-sample 1 \
  --sample-risk-gate

(
  cd "$DEBUG_ROOT"
  "$PYTHON_BIN" scripts/eval_competition_aligned.py \
    --gt-jsonl data/val_300.jsonl \
    --raw-jsonl "$V529_RAW_PATH" \
    --output "$V529_EVAL_PATH" \
    --output-csv "$V529_CSV_PATH" \
    --skip-bertscore \
    --allow-empty-loc \
    --include-samples
)

echo "Best raw: $V529_RAW_PATH"
echo "Best eval: $V529_EVAL_PATH"
