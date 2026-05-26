#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PIPELINE="${PIPELINE:-staged}"
MAX_SAMPLES="${MAX_SAMPLES:-60}"
PYTHON_BIN="${PYTHON:-python3}"
if [[ "$MAX_SAMPLES" == "0" ]]; then
  TAG="300"
  EVAL_LIMIT=0
else
  TAG="$MAX_SAMPLES"
  EVAL_LIMIT="$MAX_SAMPLES"
fi

if [[ "$PIPELINE" == "prompt" ]]; then
  RAW_JSONL="outputs/raw/docshield_api_val_${TAG}.jsonl"
  OUT_JSON="outputs/eval/docshield_api_val_${TAG}.json"
  OUT_CSV="outputs/eval/docshield_api_val_${TAG}.csv"
else
  RAW_JSONL="outputs/raw/staged_docshield_api_val_${TAG}.jsonl"
  OUT_JSON="outputs/eval/staged_docshield_api_val_${TAG}.json"
  OUT_CSV="outputs/eval/staged_docshield_api_val_${TAG}.csv"
fi

"$PYTHON_BIN" scripts/eval_competition_aligned.py \
  --gt-jsonl data/val_300.jsonl \
  --raw-jsonl "$RAW_JSONL" \
  --output "$OUT_JSON" \
  --output-csv "$OUT_CSV" \
  --max-samples "$EVAL_LIMIT" \
  --skip-bertscore \
  --include-samples \
  --allow-empty-loc
