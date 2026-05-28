#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

RAW_ARGS=()
EVAL_ARGS=()

add_result() {
  local label="$1"
  local raw_path="$2"
  local eval_path="$3"
  if [[ -f "$raw_path" ]]; then
    RAW_ARGS+=(--raw-jsonl "${label}=${raw_path}")
    if [[ -f "$eval_path" ]]; then
      EVAL_ARGS+=(--eval-json "${label}=${eval_path}")
    fi
  fi
}

add_result "v9-benign" \
  "outputs/raw/staged_cct_v9_offline_benign_reviewer_60.jsonl" \
  "outputs/eval/staged_cct_v9_offline_benign_reviewer_60.json"

add_result "v13-best-box-graft" \
  "outputs/raw/staged_cct_v13_ocr_boxes_on_v9_60.jsonl" \
  "outputs/eval/staged_cct_v13_ocr_boxes_on_v9_60.json"

add_result "v16-layout-cct" \
  "outputs/raw/staged_cct_v16_qwen_ocr_layout_cct_60.jsonl" \
  "outputs/eval/staged_cct_v16_qwen_ocr_layout_cct_60.json"

add_result "v17-stage1-grounding" \
  "outputs/raw/staged_cct_v17_qwen_ocr_stage1_grounding_60.jsonl" \
  "outputs/eval/staged_cct_v17_qwen_ocr_stage1_grounding_60.json"

add_result "v18-grounding-only" \
  "outputs/raw/staged_cct_v18_qwen_ocr_grounding_only_60.jsonl" \
  "outputs/eval/staged_cct_v18_qwen_ocr_grounding_only_60.json"

add_result "v19-multilan-stage1" \
  "outputs/raw/staged_cct_v19_multilan_stage1_60.jsonl" \
  "outputs/eval/staged_cct_v19_multilan_stage1_60.json"

add_result "v20-formal-v13-grounder" \
  "outputs/raw/staged_cct_v20_formal_v13_grounder_60.jsonl" \
  "outputs/eval/staged_cct_v20_formal_v13_grounder_60.json"

add_result "v21-fp-reviewer-v2" \
  "outputs/raw/staged_cct_v21_fp_reviewer_v2_tuned_60.jsonl" \
  "outputs/eval/staged_cct_v21_fp_reviewer_v2_tuned_60.json"

add_result "v28-full-visual-fp-review" \
  "outputs/raw/staged_cct_v28_visual_artifact_reviewer_300.jsonl" \
  "outputs/eval/staged_cct_v28_visual_artifact_reviewer_300.json"

add_result "v31-full-trigger-rescue" \
  "outputs/raw/staged_cct_v31_strong_trigger_rescue_300.jsonl" \
  "outputs/eval/staged_cct_v31_strong_trigger_rescue_300.json"

add_result "v35-full-plus-targeted" \
  "outputs/raw/staged_cct_v35_plus_targeted_rescue_300.jsonl" \
  "outputs/eval/staged_cct_v35_plus_targeted_rescue_300.json"

add_result "v37-full-zoom-crop-best" \
  "outputs/raw/staged_cct_v37_zoom_crop_rescue_300.jsonl" \
  "outputs/eval/staged_cct_v37_zoom_crop_rescue_300.json"

if [[ "${#RAW_ARGS[@]}" -eq 0 ]]; then
  echo "No result JSONL files found under outputs/raw." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/build_ocr_viewer_data.py \
  "${RAW_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  --gt-jsonl data/val_300.jsonl \
  --cache-model qwen-vl-ocr \
  --cache-dir outputs/cache/ocr_transcripts_multilan \
  --ocr-layout-model qwen-vl-ocr \
  --ocr-layout-cache-dir outputs/cache/ocr_layouts \
  --output outputs/ocr_viewer/data.json
