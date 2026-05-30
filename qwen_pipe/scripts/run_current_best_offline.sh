#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-../debug_distribution/.venv/bin/python}"
DEBUG_ROOT="${DEBUG_DISTRIBUTION_ROOT:-../debug_distribution}"

BASE_JSONL="${BASE_JSONL:-$DEBUG_ROOT/outputs/raw/staged_cct_v28_visual_artifact_reviewer_300.jsonl}"
ENSEMBLE_JSONL="${ENSEMBLE_JSONL:-$DEBUG_ROOT/outputs/raw/staged_cct_offline_ensemble_base_or_v1_300.jsonl}"
V15_JSONL="${V15_JSONL:-$DEBUG_ROOT/outputs/raw/staged_cct_v15_qwen_ocr_primary_stage1only_300.jsonl}"
V19_JSONL="${V19_JSONL:-$DEBUG_ROOT/outputs/raw/staged_cct_v19_multilan_stage1_300.jsonl}"

MOE_RAW="${MOE_RAW:-outputs/raw/qwen_pipe_v49_moe_artifact_precision_plus_r85_lowriskblue_300.jsonl}"
BOXNORM_RAW="${BOXNORM_RAW:-outputs/raw/qwen_pipe_v50_moe_artifact_precision_plus_lowriskblue_boxnorm_1p0_1p0_300.jsonl}"
OCR_REFINE_RAW="${OCR_REFINE_RAW:-outputs/raw/qwen_pipe_v52_ocr_box_refine_countmatch_300.jsonl}"
BOXSPACE_RAW="${BOXSPACE_RAW:-outputs/raw/qwen_pipe_v58_boxspace_force1000_300.jsonl}"
CROP_PATCH_RAW="${CROP_PATCH_RAW:-outputs/raw/qwen_pipe_v63_crop_patch_blockonly_w04_300.jsonl}"
TEXT_CROP_RAW="${TEXT_CROP_RAW:-outputs/raw/qwen_pipe_v71_text_crop_local_gated_300.jsonl}"
OCR_CLUSTER_RAW="${OCR_CLUSTER_RAW:-outputs/raw/qwen_pipe_v74_ocr_cluster_gated_300.jsonl}"
ISSUE_REFINE_RAW="${ISSUE_REFINE_RAW:-outputs/raw/qwen_pipe_v75b_hard_discard_rescue_300.jsonl}"
BEST_RAW="${BEST_RAW:-outputs/raw/qwen_pipe_v77_evidence_multibox_300.jsonl}"
BEST_EVAL="${BEST_EVAL:-outputs/eval/qwen_pipe_v77_evidence_multibox_300.json}"
BEST_CSV="${BEST_CSV:-outputs/eval/qwen_pipe_v77_evidence_multibox_300.csv}"

"$PYTHON_BIN" scripts/qwen_moe_fuse.py \
  --base-jsonl "$BASE_JSONL" \
  --donor "ensemble=$ENSEMBLE_JSONL" \
  --donor "v15=$V15_JSONL" \
  --donor "v19=$V19_JSONL" \
  --output-jsonl "$MOE_RAW" \
  --stats-json outputs/stats/qwen_pipe_v49_moe_artifact_precision_plus_r85_lowriskblue_300.json \
  --rescue-min-risk 85 \
  --trigger-profile artifact_precision_plus

"$PYTHON_BIN" scripts/qwen_box_scale.py \
  --input-jsonl "$MOE_RAW" \
  --output-jsonl "$BOXNORM_RAW" \
  --scale-x 1.0 \
  --scale-y 1.0

"$PYTHON_BIN" scripts/qwen_ocr_box_refine.py \
  --input-jsonl "$BOXNORM_RAW" \
  --output-jsonl "$OCR_REFINE_RAW" \
  --require-count-match

"$PYTHON_BIN" scripts/qwen_box_space_refine.py \
  --input-jsonl "$OCR_REFINE_RAW" \
  --output-jsonl "$BOXSPACE_RAW" \
  --mode normalized-1000 \
  --require-count-match \
  --final-forged-only

"$PYTHON_BIN" scripts/qwen_crop_patch_refine.py \
  --input-jsonl "$BOXSPACE_RAW" \
  --output-jsonl "$CROP_PATCH_RAW" \
  --final-forged-only \
  --profile block-only \
  --local-factor 2.0 \
  --patch-pad 3 \
  --min-patch-width-frac 0.04

"$PYTHON_BIN" scripts/qwen_text_crop_verify.py \
  --input-jsonl "$CROP_PATCH_RAW" \
  --output-jsonl "$TEXT_CROP_RAW" \
  --mode local \
  --max-candidates-per-anomaly 4

"$PYTHON_BIN" scripts/qwen_ocr_cluster_refine.py \
  --input-jsonl "$TEXT_CROP_RAW" \
  --output-jsonl "$OCR_CLUSTER_RAW"

"$PYTHON_BIN" scripts/qwen_issue_refine.py \
  --input-jsonl "$OCR_CLUSTER_RAW" \
  --output-jsonl "$ISSUE_REFINE_RAW" \
  --enable-rescue

"$PYTHON_BIN" scripts/qwen_evidence_multibox_refine.py \
  --input-jsonl "$ISSUE_REFINE_RAW" \
  --output-jsonl "$BEST_RAW"

PYTHON="$PYTHON_BIN" \
MAX_SAMPLES=0 \
RAW_JSONL="$BEST_RAW" \
OUT_JSON="$BEST_EVAL" \
OUT_CSV="$BEST_CSV" \
  scripts/qwen_pipe_loop.sh eval

echo "Current best written to $BEST_RAW"
echo "Evaluation written to $BEST_EVAL"
