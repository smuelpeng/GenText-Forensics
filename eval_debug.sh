#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/eval_competition_aligned.py \
  --gt-jsonl data/val_300.jsonl \
  --raw-jsonl outputs/raw/docshield_api_val_300.jsonl \
  --output outputs/eval/docshield_api_val_300.json \
  --output-csv outputs/eval/docshield_api_val_300.csv \
  --skip-bertscore \
  --include-samples \
  --allow-empty-loc
