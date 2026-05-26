#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python baselines/DocShield/run_docshield_api.py \
  --model "${MODEL:-qwen3-vl-flash}" \
  --prompt-mode "${PROMPT_MODE:-cct}" \
  --input-jsonl data/val_300.jsonl \
  --output-jsonl outputs/raw/docshield_api_val_300.jsonl \
  --api-key-file "${API_KEY_FILE:-api-key.txt}" \
  --max-tokens "${MAX_TOKENS:-8192}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --resume
