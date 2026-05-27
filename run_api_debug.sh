#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PIPELINE="${PIPELINE:-staged}"
MAX_SAMPLES="${MAX_SAMPLES:-60}"
PYTHON_BIN="${PYTHON:-python3}"
if [[ "$MAX_SAMPLES" == "0" ]]; then
  TAG="300"
else
  TAG="$MAX_SAMPLES"
fi

if [[ "$PIPELINE" == "prompt" ]]; then
  RAW_JSONL="${OUTPUT_JSONL:-outputs/raw/docshield_api_val_${TAG}.jsonl}"
  "$PYTHON_BIN" baselines/DocShield/run_docshield_api.py \
    --model "${MODEL:-qwen3-vl-flash}" \
    --prompt-mode "${PROMPT_MODE:-cct}" \
    --input-jsonl data/val_300.jsonl \
    --output-jsonl "$RAW_JSONL" \
    --api-key-file "${API_KEY_FILE:-/Users/penpen/Desktop/api-key.txt}" \
    --max-tokens "${MAX_TOKENS:-8192}" \
    --max-samples "$MAX_SAMPLES" \
    --num-workers "${NUM_WORKERS:-1}" \
    --resume
else
  RAW_JSONL="${OUTPUT_JSONL:-outputs/raw/staged_docshield_api_val_${TAG}.jsonl}"
  "$PYTHON_BIN" baselines/DocShield/run_staged_docshield_api.py \
    --model "${MODEL:-qwen3.6-35b-a3b}" \
    --input-jsonl data/val_300.jsonl \
    --output-jsonl "$RAW_JSONL" \
    --api-key-file "${API_KEY_FILE:-/Users/penpen/Desktop/api-key.txt}" \
    --max-tokens "${MAX_TOKENS:-8192}" \
    --max-samples "$MAX_SAMPLES" \
    --num-workers "${NUM_WORKERS:-1}" \
    --forged-risk-threshold "${FORGED_RISK_THRESHOLD:-80}" \
    --forged-risk-thresholds "${FORGED_RISK_THRESHOLDS:-ar=70,id=75}" \
    --resume
fi
