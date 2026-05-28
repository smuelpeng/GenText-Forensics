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
  BENIGN_REVIEWER_ARGS=()
  if [[ "${DISABLE_BENIGN_REVIEWER:-0}" == "1" ]]; then
    BENIGN_REVIEWER_ARGS+=(--disable-benign-reviewer)
  fi
  TAXONOMY_PROMPT_ARGS=()
  if [[ "${ENABLE_TAXONOMY_PROMPTS:-0}" == "1" ]]; then
    TAXONOMY_PROMPT_ARGS+=(--enable-taxonomy-prompts)
  fi
  OCR_TRANSCRIPT_ARGS=()
  if [[ "${OCR_TRANSCRIPT_TO_EVIDENCE:-0}" == "1" ]]; then
    OCR_TRANSCRIPT_ARGS+=(--ocr-transcript-to-evidence)
  fi
  if [[ "${OCR_TRANSCRIPT_TO_GROUNDING:-0}" == "1" ]]; then
    OCR_TRANSCRIPT_ARGS+=(--ocr-transcript-to-grounding)
  fi
  OCR_LAYOUT_ARGS=()
  if [[ "${REQUIRE_OCR_LAYOUT_CACHE:-0}" == "1" ]]; then
    OCR_LAYOUT_ARGS+=(--require-ocr-layout-cache)
  fi
  if [[ "${OCR_LAYOUT_TO_STAGE1:-0}" == "1" ]]; then
    OCR_LAYOUT_ARGS+=(--ocr-layout-to-stage1)
  fi
  "$PYTHON_BIN" baselines/DocShield/run_staged_docshield_api.py \
    --model "${MODEL:-qwen3.6-35b-a3b}" \
    --ocr-model "${OCR_MODEL:-}" \
    --ocr-transcript-model "${OCR_TRANSCRIPT_MODEL:-}" \
    --ocr-transcript-api-key-file "${OCR_TRANSCRIPT_API_KEY_FILE:-}" \
    --ocr-transcript-max-chars "${OCR_TRANSCRIPT_MAX_CHARS:-6000}" \
    --ocr-transcript-cache-dir "${OCR_TRANSCRIPT_CACHE_DIR:-outputs/cache/ocr_transcripts}" \
    --ocr-layout-cache-model "${OCR_LAYOUT_CACHE_MODEL:-qwen-vl-ocr}" \
    --ocr-layout-cache-dir "${OCR_LAYOUT_CACHE_DIR:-outputs/cache/ocr_layouts}" \
    --ocr-layout-max-spans "${OCR_LAYOUT_MAX_SPANS:-0}" \
    --ocr-layout-max-chars "${OCR_LAYOUT_MAX_CHARS:-16000}" \
    --input-jsonl data/val_300.jsonl \
    --output-jsonl "$RAW_JSONL" \
    --api-key-file "${API_KEY_FILE:-/Users/penpen/Desktop/api-key.txt}" \
    --max-tokens "${MAX_TOKENS:-8192}" \
    --max-samples "$MAX_SAMPLES" \
    --num-workers "${NUM_WORKERS:-1}" \
    --forged-risk-threshold "${FORGED_RISK_THRESHOLD:-80}" \
    --forged-risk-thresholds "${FORGED_RISK_THRESHOLDS:-ar=70,id=75}" \
    --grounding-box-scale-x "${GROUNDING_BOX_SCALE_X:-3.5}" \
    --grounding-box-scale-y "${GROUNDING_BOX_SCALE_Y:-4.0}" \
    --benign-reviewer-max-risk "${BENIGN_REVIEWER_MAX_RISK:-95}" \
    --benign-reviewer-min-hits "${BENIGN_REVIEWER_MIN_HITS:-1}" \
    --benign-reviewer-max-strong-hits "${BENIGN_REVIEWER_MAX_STRONG_HITS:-1}" \
    --benign-reviewer-max-anomalies "${BENIGN_REVIEWER_MAX_ANOMALIES:-1}" \
    ${BENIGN_REVIEWER_ARGS[@]+"${BENIGN_REVIEWER_ARGS[@]}"} \
    ${TAXONOMY_PROMPT_ARGS[@]+"${TAXONOMY_PROMPT_ARGS[@]}"} \
    ${OCR_TRANSCRIPT_ARGS[@]+"${OCR_TRANSCRIPT_ARGS[@]}"} \
    ${OCR_LAYOUT_ARGS[@]+"${OCR_LAYOUT_ARGS[@]}"} \
    --resume
fi
