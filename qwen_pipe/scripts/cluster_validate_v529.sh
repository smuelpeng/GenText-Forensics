#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$QWEN_ROOT/../scripts/eval_competition_aligned.py" && -d "$QWEN_ROOT/../data" ]]; then
  DEBUG_ROOT="${DEBUG_DISTRIBUTION_ROOT:-$QWEN_ROOT/..}"
else
  DEBUG_ROOT="${DEBUG_DISTRIBUTION_ROOT:-$QWEN_ROOT/../debug_distribution}"
fi
DEBUG_ROOT="$(cd "$DEBUG_ROOT" && pwd)"
PYTHON_BIN="${PYTHON:-$DEBUG_ROOT/.venv/bin/python}"

ARTIFACT_ROOT="${QWEN_PIPE_ARTIFACT_ROOT:-$QWEN_ROOT/outputs}"
OUTPUT_ROOT="${QWEN_PIPE_OUTPUT_ROOT:-$QWEN_ROOT/outputs/cluster_v529}"

if [[ ! -d "$ARTIFACT_ROOT" ]]; then
  echo "missing QWEN_PIPE_ARTIFACT_ROOT directory: $ARTIFACT_ROOT" >&2
  echo "Set QWEN_PIPE_ARTIFACT_ROOT to the directory containing raw/, eval/, cache/, and diagnostics/." >&2
  exit 2
fi

ARTIFACT_ROOT="$(cd "$ARTIFACT_ROOT" && pwd)"
mkdir -p "$OUTPUT_ROOT/raw" "$OUTPUT_ROOT/eval" "$OUTPUT_ROOT/diagnostics"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

BASE_RAW="$ARTIFACT_ROOT/raw/qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.jsonl"
BASE_EVAL="$ARTIFACT_ROOT/eval/qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.json"
OBS_CACHE="$ARTIFACT_ROOT/cache/pair_observations/v518_v507_lowloc_t010_context_cl32.json"
EMPTY_SELECTED="$ARTIFACT_ROOT/diagnostics/qwen_pipe_v521_empty_selected_context_diag.jsonl"

for required in "$BASE_RAW" "$BASE_EVAL" "$OBS_CACHE" "$EMPTY_SELECTED"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 2
  fi
done

export DEBUG_DISTRIBUTION_ROOT="$DEBUG_ROOT"
export PYTHON="$PYTHON_BIN"
export BASE_RAW
export BASE_EVAL
export OBS_CACHE
export EMPTY_SELECTED

export V528_RAW="$OUTPUT_ROOT/raw/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_on_v507_300.jsonl"
export V528_SUMMARY="$OUTPUT_ROOT/diagnostics/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_summary.json"
export V528_DIAG="$OUTPUT_ROOT/diagnostics/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_diag.jsonl"
export V528_EVAL="$OUTPUT_ROOT/eval/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_on_v507_300.json"
export V528_CSV="$OUTPUT_ROOT/eval/qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_on_v507_300.csv"

export V529_RAW="$OUTPUT_ROOT/raw/qwen_pipe_v529_knn_append_precision_lowloc_t010_on_v528_300.jsonl"
export V529_SUMMARY="$OUTPUT_ROOT/diagnostics/qwen_pipe_v529_knn_append_precision_lowloc_t010_summary.json"
export V529_DIAG="$OUTPUT_ROOT/diagnostics/qwen_pipe_v529_knn_append_precision_lowloc_t010_diag.jsonl"
export V529_EVAL="$OUTPUT_ROOT/eval/qwen_pipe_v529_knn_append_precision_lowloc_t010_on_v528_300.json"
export V529_CSV="$OUTPUT_ROOT/eval/qwen_pipe_v529_knn_append_precision_lowloc_t010_on_v528_300.csv"

"$QWEN_ROOT/scripts/run_best_v529_rescue.sh"

"$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ["V529_EVAL"]
with open(path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

scores = data["scores"]
print(f"cluster_validation_eval={path}")
for key in ["S_Fin_proxy", "S_Det", "S_Loc", "S_Exp", "S_Rep_proxy"]:
    print(f"{key}={scores.get(key):.12f}")
PY
