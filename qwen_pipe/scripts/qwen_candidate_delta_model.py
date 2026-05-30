#!/usr/bin/env python3
"""Candidate-level delta model for localization recall.

This is a diagnostic learning experiment: train a small ridge regressor on
GT-derived candidate deltas, using only inference-available candidate/OCR/image
features as inputs.  Predictions are made in sample-level cross-validation, so
each sample's selected boxes are scored by a model that did not train on that
sample.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v85_embedding_v4_basekey_stablecache_k8_300.jsonl"
DEFAULT_EVAL = PIPE_ROOT / "outputs/eval/qwen_pipe_v85_embedding_v4_basekey_stablecache_k8_300.json"
DEFAULT_GT = DEFAULT_DEBUG_ROOT / "data/val_300.jsonl"
DEFAULT_OCR_LAYOUT_CACHE = DEFAULT_DEBUG_ROOT / "outputs/cache/ocr_layouts"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import (  # noqa: E402
    box_iou,
    choose_topk_v89token,
    generate_candidates,
    insert_extra_anomalies,
    mask_stats,
    page_area_ratio,
    read_gt_mask,
    read_jsonl,
    sample_id_from_row,
    select_apply_extras,
)
from qwen_text_crop_verify import area, clamp_box, resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


FAMILIES = ["token", "linegrid", "evidence", "ocr", "row", "patch", "grid", "paragraph"]
LANGUAGES = ["ar", "en", "id", "ms", "th", "zh", ""]
GROUNDING_RE = re.compile(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", re.IGNORECASE)
GROUNDING_BOX_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[([^\[\]]+)\]", re.IGNORECASE)
ANOMALY_HEADER_RE = re.compile(r"^###\s+(ANOMALY[^\n]*)", re.IGNORECASE | re.MULTILINE)
TEXT_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
FAMILY_QUOTA_ORDER = ["token", "linegrid", "ocr", "row", "evidence", "patch", "paragraph", "grid", "existing"]
REPORT_RELATION_FEATURE_COUNT = 20


def conclusion_is_forged(row: dict[str, Any]) -> bool:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper() == "FORGED"


def stable_fold(sample_id: str, folds: int) -> int:
    return sum(sample_id.encode("utf-8")) % folds


def load_eval_samples(path: str) -> dict[str, dict[str, Any]]:
    data = json.loads(resolve_pipe_path(path).read_text(encoding="utf-8"))
    return {str(sample.get("sample_id") or ""): sample for sample in data.get("samples") or []}


def sample_is_lowloc(sample_id: str, eval_samples: dict[str, dict[str, Any]], args: argparse.Namespace) -> bool:
    sample = eval_samples.get(sample_id) or {}
    loc = float(sample.get("loc_score") or 0.0)
    return (
        sample.get("gt_label") == "FORGED"
        and sample.get("pred_label") == "FORGED"
        and loc < args.lowloc_weight_threshold
    )


def sample_is_loc0(sample_id: str, eval_samples: dict[str, dict[str, Any]]) -> bool:
    sample = eval_samples.get(sample_id) or {}
    return (
        sample.get("gt_label") == "FORGED"
        and sample.get("pred_label") == "FORGED"
        and float(sample.get("loc_score") or 0.0) == 0.0
    )


def sample_in_candidate_scope(
    *,
    sample_id: str,
    row: dict[str, Any],
    eval_samples: dict[str, dict[str, Any]],
    is_lowloc: bool,
    args: argparse.Namespace,
) -> bool:
    sample = eval_samples.get(sample_id) or {}
    lang = str(sample.get("language_code") or row.get("language_code") or "")
    allowed_langs = parse_family_filter(args.candidate_scope_languages)
    if allowed_langs and lang not in allowed_langs:
        return False
    if args.candidate_scope == "all":
        return True
    if args.candidate_scope == "lowloc":
        return bool(is_lowloc)
    if args.candidate_scope == "loc0":
        return sample_is_loc0(sample_id, eval_samples)
    return True


def candidate_training_weight(*, is_lowloc: bool, target_delta: float, args: argparse.Namespace) -> float:
    if args.sample_weight_mode == "lowloc" and is_lowloc:
        return float(args.lowloc_weight)
    if args.sample_weight_mode == "lowloc_positive" and is_lowloc and target_delta > 0:
        return float(args.lowloc_weight)
    return 1.0


def refresh_observation_target(obs: dict[str, Any], args: argparse.Namespace) -> None:
    append_delta = float(obs.get("append_delta") or 0.0)
    append_f1 = float(obs.get("append_f1") or obs.get("base_f1") or 0.0)
    replace_delta = float(obs.get("replace_delta") or 0.0)
    replace_f1 = float(obs.get("replace_f1") or obs.get("base_f1") or 0.0)
    if args.target_mode == "replace":
        target_delta = replace_delta
        target_new_f1 = replace_f1
    elif args.target_mode == "best":
        if replace_delta >= append_delta:
            target_delta = replace_delta
            target_new_f1 = replace_f1
        else:
            target_delta = append_delta
            target_new_f1 = append_f1
    else:
        target_delta = append_delta
        target_new_f1 = append_f1
    obs["delta"] = target_delta
    obs["new_f1"] = target_new_f1
    obs["sample_weight"] = candidate_training_weight(
        is_lowloc=bool(obs.get("is_lowloc")),
        target_delta=target_delta,
        args=args,
    )


def candidate_language(candidate: dict[str, Any], sample_row: dict[str, Any]) -> str:
    meta = candidate.get("meta") or {}
    row = sample_row.get("row") or sample_row
    return str(
        meta.get("language")
        or meta.get("language_code")
        or sample_row.get("language_code")
        or row.get("language_code")
        or ((row.get("metadata") or {}).get("language_code"))
        or ""
    ).lower()


def reject_candidate_by_guard(obs: dict[str, Any], sample: dict[str, Any], args: argparse.Namespace) -> bool:
    """GT-free candidate guard for failure modes found after local diagnosis."""
    guard = str(args.linegrid_guard or "none")
    cand = obs["candidate"]
    family = str(cand.get("family") or "")
    source = str(cand.get("source") or "")
    is_linegrid = family == "linegrid" and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
    if is_linegrid and float(args.linegrid_min_match_score or 0.0) > 0.0:
        meta = cand.get("meta") or {}
        if float(meta.get("match_score") or 0.0) < float(args.linegrid_min_match_score):
            return True
    if guard == "none":
        return False
    lang = candidate_language(cand, sample)
    if not is_linegrid:
        return False
    if guard == "no-th-linegrid" and lang == "th":
        return True
    return False


def reject_candidate_by_shape(obs: dict[str, Any], args: argparse.Namespace) -> bool:
    """GT-free candidate shape/score guard learned from append over-generation."""
    pred_delta = float(obs.get("pred_delta") or 0.0)
    cand = obs["candidate"]
    cand_area = float(area(cand.get("box") or [0, 0, 0, 0]))
    if args.pred_delta_max is not None and pred_delta > float(args.pred_delta_max):
        return True
    if float(args.candidate_min_area or 0.0) > 0.0 and cand_area < float(args.candidate_min_area):
        return True
    if float(args.candidate_max_area or 0.0) > 0.0 and cand_area > float(args.candidate_max_area):
        return True
    return False


def parse_family_filter(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def parse_text_filter(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def parse_box_text(value: str) -> list[int] | None:
    nums = [int(float(part)) for part in re.findall(r"-?\d+(?:\.\d+)?", value or "")]
    if len(nums) < 4:
        return None
    return nums[:4]


def compact_text(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def text_tokens(value: str) -> set[str]:
    tokens = {tok.lower() for tok in TEXT_TOKEN_RE.findall(str(value or "")) if len(tok) >= 2}
    # For scripts where OCR text has little whitespace, char-level overlap is a useful fallback.
    compact = compact_text(value)
    if len(tokens) <= 2 and compact:
        tokens.update(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
    return tokens


def digit_chars(value: str) -> set[str]:
    return {ch for ch in str(value or "") if ch.isdigit()}


def split_candidate_text(candidate: dict[str, Any]) -> tuple[str, str, str]:
    text = str(candidate.get("text") or "")
    meta = candidate.get("meta") or {}
    term = str(meta.get("term") or "")
    if "::" in text:
        left, right = text.split("::", 1)
        term = term or left.strip()
        line = right.strip()
    else:
        line = text
    return text, term, line


def parse_report_blocks(report: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    matches = list(ANOMALY_HEADER_RE.finditer(report or ""))
    if not matches:
        return blocks
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        header = match.group(1).strip()
        block = report[start:end]
        lower = (header + "\n" + block).lower()
        if "anomaly_v" in lower or "exhaustive recall candidate" in lower or "gt-blind" in lower:
            continue
        box = None
        box_match = GROUNDING_BOX_RE.search(block)
        if box_match:
            box = parse_box_text(box_match.group(1))
        text = re.sub(GROUNDING_BOX_RE, " ", block)
        text = re.sub(r"\[(GROUNDING|REASON)\]\s*:?", " ", text, flags=re.IGNORECASE)
        blocks.append(
            {
                "header": header,
                "box": box,
                "text": text,
                "tokens": text_tokens(text),
                "compact": compact_text(text),
                "digits": digit_chars(text),
            }
        )
    return blocks


def center_distance(a: list[int], b: list[int], width: int, height: int) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    diag = max(1.0, math.hypot(width, height))
    return min(1.0, math.hypot(ax - bx, ay - by) / diag)


def axis_overlap_ratio(a1: int, a2: int, b1: int, b2: int) -> float:
    span = max(1, min(a2, b2) - max(a1, b1))
    denom = max(1, min(a2 - a1, b2 - b1))
    return max(0.0, span / denom)


def report_relation_features(
    *,
    candidate: dict[str, Any],
    row: dict[str, Any],
    width: int,
    height: int,
    existing: list[list[int]],
) -> list[float]:
    report = str(row.get("raw_output") or "")
    blocks = parse_report_blocks(report)
    if not blocks:
        return [0.0] * REPORT_RELATION_FEATURE_COUNT
    box = [int(v) for v in candidate.get("box") or [0, 0, 0, 0]]
    full_text, term_text, line_text = split_candidate_text(candidate)
    cand_text = " ".join(part for part in (full_text, term_text, line_text) if part)
    cand_tokens = text_tokens(cand_text)
    cand_compact = compact_text(cand_text)
    term_compact = compact_text(term_text)
    cand_digits = digit_chars(cand_text)
    best_jaccard = best_cand_recall = best_block_recall = 0.0
    best_char_jaccard = best_substring = best_term_substring = 0.0
    best_digit_recall = best_any_digit = 0.0
    best_text_box_iou = best_text_contain = best_text_center = 0.0
    best_text_x_overlap = best_text_y_overlap = 0.0
    visual_words = {"blur", "blurry", "artifact", "edge", "font", "overlap", "misalign", "missing", "tamper", "color", "highlight", "模糊", "错位", "重叠", "缺失"}
    best_visual_cue = 0.0
    for block in blocks:
        btokens = block["tokens"]
        inter = cand_tokens & btokens
        union = cand_tokens | btokens
        jaccard = len(inter) / max(1, len(union))
        cand_recall = len(inter) / max(1, len(cand_tokens))
        block_recall = len(inter) / max(1, len(btokens))
        bcompact = str(block["compact"])
        cchars = set(cand_compact)
        bchars = set(bcompact)
        char_jaccard = len(cchars & bchars) / max(1, len(cchars | bchars))
        substring = 1.0 if cand_compact and len(cand_compact) <= 120 and cand_compact in bcompact else 0.0
        term_substring = 1.0 if term_compact and len(term_compact) <= 80 and term_compact in bcompact else 0.0
        bdigits = block["digits"]
        digit_recall = len(cand_digits & bdigits) / max(1, len(cand_digits)) if cand_digits else 0.0
        any_digit = 1.0 if cand_digits and cand_digits & bdigits else 0.0
        text_score = max(jaccard, cand_recall, char_jaccard, substring, term_substring, digit_recall)
        bbox = block.get("box")
        box_iou_score = contain = center = x_overlap = y_overlap = 0.0
        if bbox:
            box_iou_score = box_iou(box, bbox)
            contain = max_contain_overlap(box, [bbox])
            center = 1.0 - center_distance(box, bbox, width, height)
            x_overlap = axis_overlap_ratio(box[0], box[2], bbox[0], bbox[2])
            y_overlap = axis_overlap_ratio(box[1], box[3], bbox[1], bbox[3])
        if text_score > max(best_jaccard, best_cand_recall, best_char_jaccard, best_substring, best_digit_recall):
            best_text_box_iou = box_iou_score
            best_text_contain = contain
            best_text_center = center
            best_text_x_overlap = x_overlap
            best_text_y_overlap = y_overlap
            best_visual_cue = 1.0 if any(word in str(block["text"]).lower() for word in visual_words) else 0.0
        best_jaccard = max(best_jaccard, jaccard)
        best_cand_recall = max(best_cand_recall, cand_recall)
        best_block_recall = max(best_block_recall, block_recall)
        best_char_jaccard = max(best_char_jaccard, char_jaccard)
        best_substring = max(best_substring, substring)
        best_term_substring = max(best_term_substring, term_substring)
        best_digit_recall = max(best_digit_recall, digit_recall)
        best_any_digit = max(best_any_digit, any_digit)
    existing_iou = max((box_iou(box, b) for b in existing), default=0.0)
    novelty = 1.0 - existing_iou
    return [
        min(1.0, len(blocks) / 8.0),
        min(1.0, len(cand_tokens) / 40.0),
        min(1.0, len(cand_compact) / 240.0),
        best_jaccard,
        best_cand_recall,
        best_block_recall,
        best_char_jaccard,
        best_substring,
        best_term_substring,
        best_digit_recall,
        best_any_digit,
        best_text_box_iou,
        best_text_contain,
        best_text_center,
        best_text_x_overlap,
        best_text_y_overlap,
        novelty,
        max(best_cand_recall, best_digit_recall, best_term_substring) * novelty,
        best_visual_cue,
        1.0 if cand_tokens else 0.0,
    ]


def max_contain_overlap(box: list[int], boxes: list[list[int]]) -> float:
    bx_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    best = 0.0
    for other in boxes:
        x1 = max(box[0], other[0])
        y1 = max(box[1], other[1])
        x2 = min(box[2], other[2])
        y2 = min(box[3], other[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        best = max(best, inter / bx_area)
    return best


def region_stats(image: Image.Image, box: list[int]) -> dict[str, float]:
    arr = np.asarray(image.crop(tuple(box)).convert("RGB")).astype(np.float32)
    if arr.size == 0:
        return {k: 0.0 for k in ("mean", "std", "edge", "sat", "red", "yellow", "dark", "bright")}
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gy = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
    gx = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    red = ((arr[..., 0] > 150) & (arr[..., 0] > arr[..., 1] * 1.25) & (arr[..., 0] > arr[..., 2] * 1.25)).mean()
    yellow = ((arr[..., 0] > 140) & (arr[..., 1] > 120) & (arr[..., 2] < 120)).mean()
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "edge": gx + gy,
        "sat": float(sat.mean()),
        "red": float(red),
        "yellow": float(yellow),
        "dark": float((gray < 70).mean()),
        "bright": float((gray > 230).mean()),
    }


def ring_box(box: list[int], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = box
    pad_x = max(8, int((x2 - x1) * 1.2))
    pad_y = max(6, int((y2 - y1) * 1.6))
    return clamp_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def candidate_features(
    *,
    candidate: dict[str, Any],
    image: Image.Image,
    width: int,
    height: int,
    existing: list[list[int]],
) -> list[float]:
    box = candidate["box"]
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area = page_area_ratio(box, width, height)
    rank = (candidate.get("meta") or {}).get("v87b_rank") or (candidate.get("meta") or {}).get("v87_rank") or {}
    inner = region_stats(image, box)
    outer = region_stats(image, ring_box(box, width, height) or box)
    source = str(candidate.get("source") or "")
    text = str(candidate.get("text") or "")
    family = str(candidate.get("family") or "")
    meta = candidate.get("meta") or {}
    span_box = meta.get("span_box") or box
    if not isinstance(span_box, list) or len(span_box) < 4:
        span_box = box
    span_box = [int(v) for v in span_box[:4]]
    sx1, sy1, sx2, sy2 = span_box
    span_w = max(1, sx2 - sx1)
    span_h = max(1, sy2 - sy1)
    span_area = max(1.0, float(span_w * span_h))
    start_frac = float(meta.get("start_frac") if meta.get("start_frac") is not None else meta.get("physical_start_frac") or 0.0)
    end_frac = float(meta.get("end_frac") if meta.get("end_frac") is not None else meta.get("physical_end_frac") or 0.0)
    match_score = float(meta.get("match_score") or 0.0)
    linegrid_window = 1.0 if source.startswith(("ocr_linegrid", "ocr_scriptgrid")) else 0.0
    horizontal_offset = ((x1 + x2) / 2.0 - sx1) / span_w
    vertical_offset = ((y1 + y2) / 2.0 - sy1) / span_h
    line_edge_distance = min(start_frac, max(0.0, 1.0 - end_frac))
    one_hot = [1.0 if family == fam else 0.0 for fam in FAMILIES]
    existing_area = sum(page_area_ratio(b, width, height) for b in existing)
    overlaps = [(box_iou(box, b), max_contain_overlap(box, [b]), page_area_ratio(b, width, height)) for b in existing]
    best_iou, best_contain, best_existing_area = max(overlaps, default=(0.0, 0.0, 0.0), key=lambda t: (t[1], t[0]))
    return [
        *one_hot,
        area,
        bw / max(1, width),
        bh / max(1, height),
        math.log1p(bw / max(1, bh)),
        float(candidate.get("score") or 0.0),
        float(rank.get("score") or 0.0),
        float(rank.get("query_score") or 0.0),
        float(rank.get("compact_adjustment") or 0.0),
        1.0 - max_contain_overlap(box, existing),
        max((box_iou(box, b) for b in existing), default=0.0),
        best_iou,
        best_contain,
        best_existing_area,
        area / max(1e-6, best_existing_area) if best_existing_area else 0.0,
        float(len(existing)),
        existing_area,
        inner["mean"] / 255.0,
        inner["std"] / 128.0,
        inner["edge"] / 64.0,
        inner["sat"],
        inner["red"],
        inner["yellow"],
        inner["dark"],
        inner["bright"],
        abs(inner["mean"] - outer["mean"]) / 255.0,
        abs(inner["edge"] - outer["edge"]) / 64.0,
        1.0 if "span_id" in source else 0.0,
        1.0 if source.startswith("ocr_token_subspan") else 0.0,
        1.0 if "exact" in source else 0.0,
        1.0 if any(ch.isdigit() for ch in text) else 0.0,
        min(1.0, len(text) / 240.0),
        linegrid_window,
        start_frac,
        end_frac,
        max(0.0, end_frac - start_frac),
        line_edge_distance,
        match_score,
        page_area_ratio(span_box, width, height),
        bw / span_w,
        bh / span_h,
        (bw * bh) / span_area,
        horizontal_offset,
        vertical_offset,
        1.0 if 0.25 <= horizontal_offset <= 0.75 else 0.0,
        1.0 if line_edge_distance <= 0.05 else 0.0,
    ]


def replacement_context_features(
    *,
    candidate_box: list[int],
    existing: list[list[int]],
    replace_index: int,
    width: int,
    height: int,
    language_code: str,
) -> list[float]:
    """GT-free context for a candidate replacement action.

    Earlier low-S_Loc diagnostics showed that candidate coverage is high but
    action selection is brittle.  The base crop features describe the candidate
    alone; this context describes the actual old grounding it would replace.
    """
    box = [int(v) for v in candidate_box[:4]]
    c_area = max(1.0, float(max(0, box[2] - box[0]) * max(0, box[3] - box[1])))
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    c_w = max(1.0, float(box[2] - box[0]))
    c_h = max(1.0, float(box[3] - box[1]))
    page_diag = max(1.0, math.hypot(width, height))
    lang = language_code if language_code in LANGUAGES else ""
    base = [
        *[1.0 if lang == item else 0.0 for item in LANGUAGES],
        1.0 if replace_index >= 0 else 0.0,
        float(replace_index + 1) / max(1.0, float(len(existing))) if replace_index >= 0 and existing else 0.0,
    ]
    if replace_index < 0 or replace_index >= len(existing):
        return base + [0.0] * 17
    old = [int(v) for v in existing[replace_index][:4]]
    o_area = max(1.0, float(max(0, old[2] - old[0]) * max(0, old[3] - old[1])))
    ox = (old[0] + old[2]) / 2.0
    oy = (old[1] + old[3]) / 2.0
    o_w = max(1.0, float(old[2] - old[0]))
    o_h = max(1.0, float(old[3] - old[1]))
    x_overlap = axis_overlap_ratio(box[0], box[2], old[0], old[2])
    y_overlap = axis_overlap_ratio(box[1], box[3], old[1], old[3])
    center_dist = min(1.0, math.hypot(cx - ox, cy - oy) / page_diag)
    old_diag = max(1.0, math.hypot(o_w, o_h))
    local_center_dist = min(2.0, math.hypot(cx - ox, cy - oy) / old_diag)
    cand_inside_old = max_contain_overlap(box, [old])
    old_inside_cand = max_contain_overlap(old, [box])
    iou = box_iou(box, old)
    area_ratio = c_area / o_area
    return base + [
        page_area_ratio(old, width, height),
        o_w / max(1.0, width),
        o_h / max(1.0, height),
        math.log1p(o_w / max(1.0, o_h)),
        iou,
        cand_inside_old,
        old_inside_cand,
        x_overlap,
        y_overlap,
        center_dist,
        local_center_dist,
        area_ratio,
        math.log1p(area_ratio),
        abs(math.log(max(1e-6, area_ratio))),
        c_w / o_w,
        c_h / o_h,
        1.0 if 0.15 <= area_ratio <= 1.25 and local_center_dist <= 1.0 else 0.0,
    ]


def candidate_to_dict(candidate: Any) -> dict[str, Any]:
    return {
        "label": candidate.label,
        "box": candidate.box,
        "source": candidate.source,
        "family": candidate.family,
        "score": candidate.score,
        "text": candidate.text,
        "meta": candidate.meta,
    }


def family_quota_preselect(candidates: list[Any], family_quota: int, fill_limit: int) -> list[dict[str, Any]]:
    """Build a broad GT-free pool without letting the old ranker erase families.

    v165 showed that all-candidate coverage is high but old top-k coverage is
    weak.  This preselector keeps the best candidates from each family before
    the learned reranker scores them, so line/row/ocr anchors can compete with
    token anchors instead of being filtered out upstream.
    """
    by_family: dict[str, list[Any]] = defaultdict(list)
    for cand in sorted(candidates, key=lambda c: (float(c.score), -area(c.box)), reverse=True):
        by_family[str(cand.family)].append(cand)

    selected: list[Any] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()

    def add(cand: Any) -> None:
        key = (str(cand.family), tuple(int(v) for v in cand.box))
        if key in seen:
            return
        seen.add(key)
        selected.append(cand)

    for family in FAMILY_QUOTA_ORDER:
        for cand in by_family.get(family, [])[:family_quota]:
            add(cand)
    if fill_limit > 0 and len(selected) < fill_limit:
        for cand in sorted(candidates, key=lambda c: (float(c.score), -area(c.box)), reverse=True):
            add(cand)
            if len(selected) >= fill_limit:
                break
    return [candidate_to_dict(cand) for cand in selected]


def preselect_candidates(row: dict[str, Any], candidates: list[Any], args: argparse.Namespace, width: int, height: int) -> list[dict[str, Any]]:
    if args.candidate_preselect == "family_quota":
        return family_quota_preselect(candidates, args.family_quota, args.preselect_limit)
    if args.candidate_preselect == "all":
        return [candidate_to_dict(cand) for cand in sorted(candidates, key=lambda c: (float(c.score), -area(c.box)), reverse=True)]
    return [candidate_to_dict(cand) for cand in choose_topk_v89token(row, candidates, args.top_k, width, height)]


def best_replace_delta(
    *,
    gt_mask: np.ndarray | None,
    existing: list[list[int]],
    candidate_box: list[int],
    width: int,
    height: int,
    base_f1: float,
) -> tuple[float, int, float]:
    if gt_mask is None:
        return 0.0, -1, 0.0
    if not existing:
        new_f1 = float(mask_stats(gt_mask, [candidate_box], width, height).get("mask_f1") or 0.0)
        return new_f1 - base_f1, -1, new_f1
    best_delta = -1e9
    best_index = -1
    best_f1 = 0.0
    for idx, _old in enumerate(existing):
        replaced = list(existing)
        replaced[idx] = candidate_box
        new_f1 = float(mask_stats(gt_mask, replaced, width, height).get("mask_f1") or 0.0)
        delta = new_f1 - base_f1
        if delta > best_delta:
            best_delta = delta
            best_index = idx
            best_f1 = new_f1
    return best_delta, best_index, best_f1


def infer_replace_index(candidate_box: list[int], existing: list[list[int]]) -> int:
    """Choose a replacement target without using GT.

    The replacement policy prefers an existing wide box that spatially contains
    the candidate.  If no box contains it, fall back to IoU and then center
    distance.  This keeps training/evaluation aligned with deployable inference:
    GT may score the result after the fact, but it never chooses the index.
    """
    if not existing:
        return -1
    cx = (candidate_box[0] + candidate_box[2]) / 2.0
    cy = (candidate_box[1] + candidate_box[3]) / 2.0
    c_area = max(1, (candidate_box[2] - candidate_box[0]) * (candidate_box[3] - candidate_box[1]))
    ranked: list[tuple[float, int]] = []
    for idx, old in enumerate(existing):
        old_area = max(1, (old[2] - old[0]) * (old[3] - old[1]))
        contain = max_contain_overlap(candidate_box, [old])
        iou = box_iou(candidate_box, old)
        ox = (old[0] + old[2]) / 2.0
        oy = (old[1] + old[3]) / 2.0
        diag = max(1.0, ((old[2] - old[0]) ** 2 + (old[3] - old[1]) ** 2) ** 0.5)
        center_score = 1.0 / (1.0 + (((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 / diag))
        compact_gain = max(0.0, 1.0 - min(1.0, c_area / old_area))
        score = contain * 4.0 + iou * 2.0 + center_score * 0.35 + compact_gain * 0.25
        # Avoid replacing a very small precise box with another unrelated small
        # box unless they genuinely overlap.
        if contain < 0.05 and iou < 0.02:
            score -= 1.5
        ranked.append((score, idx))
    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] < -0.5:
        return -1
    return ranked[0][1]


def replace_delta_for_index(
    *,
    gt_mask: np.ndarray | None,
    existing: list[list[int]],
    candidate_box: list[int],
    replace_index: int,
    width: int,
    height: int,
    base_f1: float,
) -> tuple[float, float]:
    if gt_mask is None or replace_index < 0 or replace_index >= len(existing):
        return 0.0, base_f1
    replaced = list(existing)
    replaced[replace_index] = candidate_box
    new_f1 = float(mask_stats(gt_mask, replaced, width, height).get("mask_f1") or 0.0)
    return new_f1 - base_f1, new_f1


def build_observations(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    raw_rows = read_jsonl(resolve_pipe_path(args.input_jsonl))
    gt_rows = {
        str(r.get("sample_id") or Path(str(r.get("image_file") or "")).stem): r
        for r in read_jsonl(Path(args.gt_jsonl).expanduser() if Path(args.gt_jsonl).expanduser().is_absolute() else debug_root / args.gt_jsonl)
    }
    eval_samples = load_eval_samples(args.eval_json)
    observations: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        sid = sample_id_from_row(row)
        sample_eval = eval_samples.get(sid) or {}
        language_code = str(sample_eval.get("language_code") or "")
        is_lowloc = sample_is_lowloc(sid, eval_samples, args)
        if not conclusion_is_forged(row):
            sample_rows.append({"sample_id": sid, "row": row, "language_code": language_code, "selected": []})
            continue
        if not sample_in_candidate_scope(
            sample_id=sid,
            row=row,
            eval_samples=eval_samples,
            is_lowloc=is_lowloc,
            args=args,
        ):
            sample_rows.append({"sample_id": sid, "row": row, "language_code": language_code, "selected": [], "candidate_indices": []})
            continue
        image_path = resolve_image_path(row, debug_root)
        if not image_path:
            sample_rows.append({"sample_id": sid, "row": row, "language_code": language_code, "selected": []})
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        candidates = generate_candidates(
            row,
            image,
            debug_root,
            Path(args.ocr_layout_cache_dir).expanduser().resolve(),
            args.ocr_layout_model,
            args.coord_mode,
            args.max_candidates,
            enable_token_candidates=True,
            enable_linegrid_candidates=args.enable_linegrid_candidates,
            enable_scriptgrid_candidates=args.enable_scriptgrid_candidates,
            language_code=str(row.get("language_code") or ((row.get("metadata") or {}).get("language_code")) or ""),
        )
        top_dicts = preselect_candidates(row, candidates, args, width, height)
        existing = report_boxes(str(row.get("raw_output") or ""))
        extras = select_apply_extras(
            top_dicts,
            existing,
            width=width,
            height=height,
            limit=args.candidate_limit,
            max_area_ratio=args.apply_max_area_ratio,
            duplicate_iou=args.apply_duplicate_iou,
        )
        gt_mask = None
        gt_row = gt_rows.get(sid)
        if gt_row:
            gt_mask = read_gt_mask(gt_row, width, height, debug_root)
        base_f1 = float(mask_stats(gt_mask, existing, width, height).get("mask_f1") or 0.0) if gt_mask is not None else 0.0
        sample_obs: list[int] = []
        for idx, cand in enumerate(extras):
            inferred_replace_index = infer_replace_index(cand["box"], existing)
            fvec = candidate_features(candidate=cand, image=image, width=width, height=height, existing=existing)
            fvec += replacement_context_features(
                candidate_box=cand["box"],
                existing=existing,
                replace_index=inferred_replace_index,
                width=width,
                height=height,
                language_code=language_code,
            )
            append_f1 = float(mask_stats(gt_mask, existing + [cand["box"]], width, height).get("mask_f1") or 0.0) if gt_mask is not None else 0.0
            oracle_replace_delta, oracle_replace_index, oracle_replace_f1 = best_replace_delta(
                gt_mask=gt_mask,
                existing=existing,
                candidate_box=cand["box"],
                width=width,
                height=height,
                base_f1=base_f1,
            )
            replace_delta, replace_f1 = replace_delta_for_index(
                gt_mask=gt_mask,
                existing=existing,
                candidate_box=cand["box"],
                replace_index=inferred_replace_index,
                width=width,
                height=height,
                base_f1=base_f1,
            )
            append_delta = append_f1 - base_f1
            if args.target_mode == "replace":
                target_delta = replace_delta
                target_new_f1 = replace_f1
            elif args.target_mode == "best":
                if replace_delta >= append_delta:
                    target_delta = replace_delta
                    target_new_f1 = replace_f1
                else:
                    target_delta = append_delta
                    target_new_f1 = append_f1
            else:
                target_delta = append_delta
                target_new_f1 = append_f1
            obs = {
                "sample_id": sid,
                "candidate_index": idx,
                "candidate": cand,
                "features": (
                    fvec
                    + report_relation_features(candidate=cand, row=row, width=width, height=height, existing=existing)
                    if args.enable_report_relation_features
                    else fvec
                ),
                "report_relation_features_added": bool(args.enable_report_relation_features),
                "sample_weight": candidate_training_weight(is_lowloc=is_lowloc, target_delta=target_delta, args=args),
                "base_f1": base_f1,
                "append_f1": append_f1,
                "append_delta": append_delta,
                "replace_f1": replace_f1,
                "replace_delta": replace_delta,
                "replace_index": inferred_replace_index,
                "oracle_replace_f1": oracle_replace_f1,
                "oracle_replace_delta": oracle_replace_delta,
                "oracle_replace_index": oracle_replace_index,
                "new_f1": target_new_f1,
                "delta": target_delta,
                "is_lowloc": bool(is_lowloc),
                "fold": stable_fold(sid, args.folds),
            }
            sample_obs.append(len(observations))
            observations.append(obs)
        sample_rows.append({"sample_id": sid, "row": row, "language_code": language_code, "selected": [], "candidate_indices": sample_obs})
    return observations, sample_rows


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    y_fit = y
    if weights is not None:
        w_sqrt = np.sqrt(np.maximum(weights, 1e-6)).reshape(-1, 1)
        xb = xb * w_sqrt
        y_fit = y * w_sqrt.reshape(-1)
    reg = np.eye(xb.shape[1]) * alpha
    reg[0, 0] = 0.0
    w = np.linalg.pinv(xb.T @ xb + reg) @ xb.T @ y_fit
    return w, mean, std


def predict_ridge(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    w, mean, std = model
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    return xb @ w


def fit_knn(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    return xs, y, np.maximum(weights, 1e-6), mean, std


def predict_knn(
    x: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    k: int,
    distance_weight: bool,
    chunk_size: int = 256,
) -> np.ndarray:
    train_x, train_y, train_weights, mean, std = model
    xs = (x - mean) / std
    k = max(1, min(int(k), len(train_y)))
    out = np.zeros(xs.shape[0], dtype=np.float64)
    for start in range(0, xs.shape[0], chunk_size):
        chunk = xs[start : start + chunk_size]
        dist2 = ((chunk[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
        nn = np.argpartition(dist2, k - 1, axis=1)[:, :k]
        nn_dist = np.take_along_axis(dist2, nn, axis=1)
        nn_y = train_y[nn]
        nn_w = train_weights[nn]
        if distance_weight:
            nn_w = nn_w / (np.sqrt(nn_dist) + 1e-6)
        out[start : start + len(chunk)] = (nn_y * nn_w).sum(axis=1) / np.maximum(nn_w.sum(axis=1), 1e-6)
    return out


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
    iterations: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    y = y.astype(np.float64)
    weights = np.maximum(weights.astype(np.float64), 1e-6)
    pos = float((y > 0.5).sum())
    neg = float((y <= 0.5).sum())
    if pos > 0 and neg > 0:
        class_weight = np.where(y > 0.5, (pos + neg) / (2.0 * pos), (pos + neg) / (2.0 * neg))
        weights = weights * class_weight
    weights = weights / max(1e-6, weights.mean())
    coef = np.zeros(xb.shape[1], dtype=np.float64)
    lr = float(learning_rate)
    for _ in range(max(1, int(iterations))):
        pred = sigmoid(xb @ coef)
        grad = (xb.T @ ((pred - y) * weights)) / max(1, len(y))
        grad[1:] += float(l2) * coef[1:] / max(1, len(y))
        coef -= lr * grad
    return coef, mean, std


def predict_logistic(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    coef, mean, std = model
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    return sigmoid(xb @ coef)


def has_binary_targets(y: np.ndarray) -> bool:
    return bool((y > 0.5).any() and (y <= 0.5).any())


def predict_family_ridge(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    family_train: np.ndarray,
    x_test: np.ndarray,
    family_test: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Train one delta regressor per candidate family, with a global fallback."""
    global_model = fit_ridge(x_train, y_train, args.ridge_alpha, w_train)
    out = predict_ridge(x_test, global_model)
    for family in sorted(set(family_test.tolist())):
        train_mask = family_train == family
        test_mask = family_test == family
        if int(train_mask.sum()) < args.family_model_min_train:
            continue
        model = fit_ridge(x_train[train_mask], y_train[train_mask], args.ridge_alpha, w_train[train_mask])
        out[test_mask] = predict_ridge(x_test[test_mask], model)
    return out


def predict_family_logistic(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    family_train: np.ndarray,
    x_test: np.ndarray,
    family_test: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Train one positive-candidate gate per family, with a global fallback."""
    global_model = fit_logistic(
        x_train,
        y_train,
        w_train,
        l2=args.risk_l2,
        iterations=args.risk_iters,
        learning_rate=args.risk_lr,
    )
    out = predict_logistic(x_test, global_model)
    for family in sorted(set(family_test.tolist())):
        train_mask = family_train == family
        test_mask = family_test == family
        if int(train_mask.sum()) < args.family_model_min_train:
            continue
        if not has_binary_targets(y_train[train_mask]):
            continue
        model = fit_logistic(
            x_train[train_mask],
            y_train[train_mask],
            w_train[train_mask],
            l2=args.risk_l2,
            iterations=args.risk_iters,
            learning_rate=args.risk_lr,
        )
        out[test_mask] = predict_logistic(x_test[test_mask], model)
    return out


def containment_ratio(inner: list[int], outer: list[int]) -> float:
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / inner_area


def replacement_risk_features(obs: dict[str, Any], sample_map: dict[str, dict[str, Any]]) -> list[float]:
    sample = sample_map.get(str(obs.get("sample_id") or "")) or {}
    row = sample.get("row") or sample
    cand = obs["candidate"]
    box = cand["box"]
    width = int(row.get("width") or row.get("image_width") or 1)
    height = int(row.get("height") or row.get("image_height") or 1)
    existing = report_boxes(str(row.get("raw_output") or ""))
    replace_index = int(obs.get("replace_index") if obs.get("replace_index") is not None else -1)
    old = existing[replace_index] if 0 <= replace_index < len(existing) else None
    cand_area = float(area(box))
    old_area = float(area(old)) if old else 1.0
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    if old:
        ox = (old[0] + old[2]) / 2.0
        oy = (old[1] + old[3]) / 2.0
        old_w = max(1.0, float(old[2] - old[0]))
        old_h = max(1.0, float(old[3] - old[1]))
        old_diag = max(1.0, (old_w * old_w + old_h * old_h) ** 0.5)
        center_dist = (((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5) / old_diag
        cand_in_old = containment_ratio(box, old)
        old_in_cand = containment_ratio(old, box)
        iou = box_iou(box, old)
        old_page_area = page_area_ratio(old, width, height)
    else:
        old_w = old_h = old_diag = center_dist = cand_in_old = old_in_cand = iou = old_page_area = 0.0
    meta = cand.get("meta") or {}
    family = str(cand.get("family") or "")
    source = str(cand.get("source") or "")
    area_ratio = cand_area / max(1.0, old_area)
    one_hot = [1.0 if family == fam else 0.0 for fam in FAMILIES]
    return [
        *one_hot,
        page_area_ratio(box, width, height),
        old_page_area,
        area_ratio,
        math.log1p(area_ratio),
        abs(math.log(max(area_ratio, 1e-6))),
        iou,
        cand_in_old,
        old_in_cand,
        center_dist,
        float(replace_index >= 0),
        float(len(existing)),
        float(cand.get("score") or 0.0),
        float(meta.get("match_score") or 0.0),
        float(meta.get("token_hit_count") or 0.0),
        float(len(meta.get("query_hits") or [])),
        float(len(meta.get("number_hits") or [])),
        float(len(meta.get("visual_hits") or [])),
        float(meta.get("start_frac") if meta.get("start_frac") is not None else meta.get("physical_start_frac") or 0.0),
        float(meta.get("end_frac") if meta.get("end_frac") is not None else meta.get("physical_end_frac") or 0.0),
        1.0 if source.startswith("ocr_token_subspan") else 0.0,
        1.0 if source.startswith(("ocr_linegrid", "ocr_scriptgrid")) else 0.0,
        1.0 if source.startswith("stage_evidence") else 0.0,
        1.0 if source.startswith("ocr_span") else 0.0,
    ]


def append_report_relation_features(
    observations: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if not args.enable_report_relation_features:
        return
    sample_map = {str(sample.get("sample_id") or ""): sample for sample in sample_rows}
    for obs in observations:
        if obs.get("report_relation_features_added"):
            continue
        sample = sample_map.get(str(obs.get("sample_id") or ""))
        if not sample:
            obs["features"] = list(obs.get("features") or []) + [0.0] * REPORT_RELATION_FEATURE_COUNT
            obs["report_relation_features_added"] = True
            continue
        row = sample.get("row") or sample
        width = int(row.get("width") or row.get("image_width") or 1)
        height = int(row.get("height") or row.get("image_height") or 1)
        existing = report_boxes(str(row.get("raw_output") or ""))
        obs["features"] = list(obs.get("features") or []) + report_relation_features(
            candidate=obs.get("candidate") or {},
            row=row,
            width=width,
            height=height,
            existing=existing,
        )
        obs["report_relation_features_added"] = True


def predict_fold(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    family_train: np.ndarray,
    x_test: np.ndarray,
    family_test: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.rank_model == "knn":
        model = fit_knn(x_train, y_train, w_train)
        return predict_knn(
            x_test,
            model,
            k=args.knn_k,
            distance_weight=bool(args.knn_distance_weight),
        )
    if args.rank_model == "family_ridge":
        return predict_family_ridge(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            family_train=family_train,
            x_test=x_test,
            family_test=family_test,
            args=args,
        )
    model = fit_ridge(x_train, y_train, args.ridge_alpha, w_train)
    return predict_ridge(x_test, model)


def crossval_select(observations: list[dict[str, Any]], sample_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not observations:
        return {"selected_samples": 0, "selected_candidates": 0}
    x_all = np.asarray([o["features"] for o in observations], dtype=np.float64)
    y_all = np.asarray([float(o["delta"]) for o in observations], dtype=np.float64)
    weights_all = np.asarray([float(o.get("sample_weight") or 1.0) for o in observations], dtype=np.float64)
    family_all = np.asarray([str((o.get("candidate") or {}).get("family") or "") for o in observations], dtype=object)
    sample_map = {str(sample.get("sample_id") or ""): sample for sample in sample_rows}
    risk_rows = []
    for o in observations:
        risk_features = replacement_risk_features(o, sample_map)
        if args.enable_report_relation_features:
            risk_features += list(o.get("features") or [])[-REPORT_RELATION_FEATURE_COUNT:]
        risk_rows.append(risk_features)
    risk_x_all = np.asarray(risk_rows, dtype=np.float64)
    risk_y_all = (y_all > 0.0).astype(np.float64)
    pred = np.zeros(len(observations), dtype=np.float64)
    risk_pred = np.ones(len(observations), dtype=np.float64)
    folds_used = []
    for fold in range(args.folds):
        train_idx = [i for i, o in enumerate(observations) if int(o["fold"]) != fold]
        test_idx = [i for i, o in enumerate(observations) if int(o["fold"]) == fold]
        if not train_idx or not test_idx:
            continue
        pred[test_idx] = predict_fold(
            x_train=x_all[train_idx],
            y_train=y_all[train_idx],
            w_train=weights_all[train_idx],
            family_train=family_all[train_idx],
            x_test=x_all[test_idx],
            family_test=family_all[test_idx],
            args=args,
        )
        if args.risk_gate_model == "logistic":
            risk_model = fit_logistic(
                risk_x_all[train_idx],
                risk_y_all[train_idx],
                weights_all[train_idx],
                l2=args.risk_l2,
                iterations=args.risk_iters,
                learning_rate=args.risk_lr,
            )
            risk_pred[test_idx] = predict_logistic(risk_x_all[test_idx], risk_model)
        elif args.risk_gate_model == "family_logistic":
            risk_pred[test_idx] = predict_family_logistic(
                x_train=risk_x_all[train_idx],
                y_train=risk_y_all[train_idx],
                w_train=weights_all[train_idx],
                family_train=family_all[train_idx],
                x_test=risk_x_all[test_idx],
                family_test=family_all[test_idx],
                args=args,
            )
        folds_used.append(fold)
    for idx, value in enumerate(pred):
        observations[idx]["pred_delta"] = float(value)
        observations[idx]["risk_score"] = float(risk_pred[idx])
    selected_samples = 0
    selected_candidates = 0
    guard_rejected = 0
    guard_rejected_by_language: Counter[str] = Counter()
    risk_rejected = 0
    family_rejected = 0
    source_rejected = 0
    shape_rejected = 0
    allowed_families = parse_family_filter(args.allowed_families)
    blocked_families = parse_family_filter(args.blocked_families)
    allowed_source_prefixes = parse_text_filter(args.allowed_source_prefixes)
    blocked_source_substrings = parse_text_filter(args.blocked_source_substrings)
    for sample in sample_rows:
        indices = list(sample.get("candidate_indices") or [])
        ranked = sorted((observations[i] for i in indices), key=lambda o: float(o.get("pred_delta") or 0.0), reverse=True)
        picked = []
        occupied = report_boxes(str(sample["row"].get("raw_output") or ""))
        used_replace_indices: set[int] = set()
        for obs in ranked:
            pred_delta = float(obs.get("pred_delta") or 0.0)
            if pred_delta < args.pred_delta_threshold:
                continue
            cand = obs["candidate"]
            if reject_candidate_by_guard(obs, sample, args):
                guard_rejected += 1
                guard_rejected_by_language[candidate_language(cand, sample) or "unknown"] += 1
                continue
            if args.shape_gate_stage == "pre" and reject_candidate_by_shape(obs, args):
                shape_rejected += 1
                continue
            cand_family = str(cand.get("family") or "")
            if allowed_families and cand_family not in allowed_families:
                family_rejected += 1
                continue
            if blocked_families and cand_family in blocked_families:
                family_rejected += 1
                continue
            cand_source = str(cand.get("source") or "")
            if allowed_source_prefixes and not any(cand_source.startswith(prefix) for prefix in allowed_source_prefixes):
                source_rejected += 1
                continue
            if blocked_source_substrings and any(part in cand_source for part in blocked_source_substrings):
                source_rejected += 1
                continue
            if args.risk_gate_model != "none" and float(obs.get("risk_score") or 0.0) < args.risk_threshold:
                risk_rejected += 1
                continue
            box = cand.get("box")
            replace_index = int(obs.get("replace_index") if obs.get("replace_index") is not None else -1)
            if args.apply_mode == "replace":
                if replace_index < 0 or replace_index in used_replace_indices:
                    continue
                if replace_index >= len(occupied):
                    continue
                used_replace_indices.add(replace_index)
            else:
                if any(box_iou(box, prev) >= args.apply_duplicate_iou for prev in occupied):
                    continue
            picked.append(obs)
            if args.apply_mode == "replace" and replace_index >= 0:
                occupied[replace_index] = box
            else:
                occupied.append(box)
            if len(picked) >= args.apply_top_n:
                break
        if args.shape_gate_stage == "post":
            kept = []
            for obs in picked:
                if reject_candidate_by_shape(obs, args):
                    shape_rejected += 1
                    continue
                kept.append(obs)
            picked = kept
        sample["selected"] = picked
        if picked:
            selected_samples += 1
            selected_candidates += len(picked)
    return {
        "folds": folds_used,
        "candidate_count": len(observations),
        "target_delta_mean": float(y_all.mean()),
        "target_delta_positive_rate": float((y_all > 0).mean()),
        "pred_delta_mean": float(pred.mean()),
        "pred_delta_max": float(pred.max()),
        "sample_weight_mode": args.sample_weight_mode,
        "weighted_candidate_count": int((weights_all > 1.0).sum()),
        "mean_sample_weight": float(weights_all.mean()),
        "linegrid_guard": args.linegrid_guard,
        "guard_rejected": guard_rejected,
        "guard_rejected_by_language": dict(guard_rejected_by_language),
        "rank_model": args.rank_model,
        "family_model_min_train": args.family_model_min_train,
        "knn_k": args.knn_k,
        "knn_distance_weight": bool(args.knn_distance_weight),
        "risk_gate_model": args.risk_gate_model,
        "risk_threshold": args.risk_threshold,
        "risk_rejected": risk_rejected,
        "risk_score_mean": float(risk_pred.mean()),
        "risk_score_max": float(risk_pred.max()),
        "allowed_families": sorted(allowed_families),
        "blocked_families": sorted(blocked_families),
        "family_rejected": family_rejected,
        "allowed_source_prefixes": allowed_source_prefixes,
        "blocked_source_substrings": blocked_source_substrings,
        "source_rejected": source_rejected,
        "pred_delta_max": args.pred_delta_max,
        "candidate_min_area": args.candidate_min_area,
        "candidate_max_area": args.candidate_max_area,
        "shape_gate_stage": args.shape_gate_stage,
        "shape_rejected": shape_rejected,
        "selected_samples": selected_samples,
        "selected_candidates": selected_candidates,
    }


def replace_groundings(report: str, replacements: dict[int, list[int]]) -> tuple[str, int]:
    if not replacements:
        return report, 0
    idx = 0
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx, count
        cur_idx = idx
        idx += 1
        box = replacements.get(cur_idx)
        if not box:
            return match.group(0)
        count += 1
        return f"{match.group(1)}{box}"

    return GROUNDING_RE.sub(repl, report), count


def write_outputs(observations: list[dict[str, Any]], sample_rows: list[dict[str, Any]], args: argparse.Namespace, cv_summary: dict[str, Any]) -> None:
    from postprocess import parse_cct_report  # type: ignore

    diag_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    raw_path = resolve_pipe_path(args.applied_raw_jsonl) if args.applied_raw_jsonl else None
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)

    obs_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        obs_by_sample[obs["sample_id"]].append(obs)
    with diag_path.open("w", encoding="utf-8") as fh:
        for sample in sample_rows:
            selected = sample.get("selected") or []
            fh.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "candidate_count": len(obs_by_sample.get(sample["sample_id"], [])),
                        "selected_count": len(selected),
                        "selected": [
                            {
                                "candidate": s["candidate"],
                                "pred_delta": s.get("pred_delta"),
                                "risk_score": s.get("risk_score"),
                                "target_delta": s.get("delta"),
                                "append_delta": s.get("append_delta"),
                                "replace_delta": s.get("replace_delta"),
                                "replace_index": s.get("replace_index"),
                                "base_f1": s.get("base_f1"),
                                "new_f1": s.get("new_f1"),
                            }
                            for s in selected
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if raw_path:
        with raw_path.open("w", encoding="utf-8") as fh:
            for sample in sample_rows:
                row = sample["row"]
                selected = sample.get("selected") or []
                if selected:
                    row = dict(row)
                    extras = [s["candidate"] for s in selected]
                    if args.apply_mode == "replace":
                        replacements = {
                            int(s["replace_index"]): s["candidate"]["box"]
                            for s in selected
                            if int(s.get("replace_index") if s.get("replace_index") is not None else -1) >= 0
                        }
                        report, replaced_count = replace_groundings(str(row.get("raw_output") or ""), replacements)
                        if replaced_count < len(extras):
                            # Fallback for rare malformed reports: preserve selected evidence.
                            remaining = [
                                s["candidate"]
                                for s in selected[replaced_count:]
                            ]
                            report = insert_extra_anomalies(report, remaining)
                    else:
                        report = insert_extra_anomalies(str(row.get("raw_output") or ""), extras)
                    row["raw_output"] = report
                    row["parsed"] = parse_cct_report(report)
                    stage_outputs = dict(row.get("stage_outputs") or {})
                    stage_outputs["qwen_pipe_v91_candidate_delta_model"] = {
                        "applied": True,
                        "boxes_added": len(extras),
                        "apply_mode": args.apply_mode,
                        "target_mode": args.target_mode,
                        "candidate_preselect": args.candidate_preselect,
                        "pred_delta_threshold": args.pred_delta_threshold,
                        "pred_delta_max": args.pred_delta_max,
                        "candidate_min_area": args.candidate_min_area,
                        "candidate_max_area": args.candidate_max_area,
                        "shape_gate_stage": args.shape_gate_stage,
                        "linegrid_guard": args.linegrid_guard,
                        "linegrid_min_match_score": args.linegrid_min_match_score,
                        "risk_gate_model": args.risk_gate_model,
                        "risk_threshold": args.risk_threshold,
                        "selected": [
                            {
                                "candidate": s["candidate"],
                                "pred_delta": s.get("pred_delta"),
                                "risk_score": s.get("risk_score"),
                                "replace_index": s.get("replace_index"),
                            }
                            for s in selected
                        ],
                        "policy": "Sample-level cross-validated ridge model predicts candidate mask-F1 delta from GT-free OCR/candidate/image features; candidate preselection is GT-free.",
                    }
                    row["stage_outputs"] = stage_outputs
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        **cv_summary,
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "applied_raw_jsonl": str(raw_path) if raw_path else "",
        "candidate_preselect": args.candidate_preselect,
        "candidate_scope": args.candidate_scope,
        "candidate_scope_languages": args.candidate_scope_languages,
        "family_quota": args.family_quota,
        "preselect_limit": args.preselect_limit,
        "enable_linegrid_candidates": bool(args.enable_linegrid_candidates),
        "enable_scriptgrid_candidates": bool(args.enable_scriptgrid_candidates),
        "sample_weight_mode": args.sample_weight_mode,
        "lowloc_weight": args.lowloc_weight,
        "lowloc_weight_threshold": args.lowloc_weight_threshold,
        "linegrid_guard": args.linegrid_guard,
        "linegrid_min_match_score": args.linegrid_min_match_score,
        "apply_top_n": args.apply_top_n,
        "apply_mode": args.apply_mode,
        "target_mode": args.target_mode,
        "pred_delta_threshold": args.pred_delta_threshold,
        "pred_delta_max": args.pred_delta_max,
        "candidate_min_area": args.candidate_min_area,
        "candidate_max_area": args.candidate_max_area,
        "shape_gate_stage": args.shape_gate_stage,
        "ridge_alpha": args.ridge_alpha,
        "rank_model": args.rank_model,
        "family_model_min_train": args.family_model_min_train,
        "knn_k": args.knn_k,
        "knn_distance_weight": bool(args.knn_distance_weight),
        "risk_gate_model": args.risk_gate_model,
        "risk_threshold": args.risk_threshold,
        "risk_l2": args.risk_l2,
        "risk_iters": args.risk_iters,
        "risk_lr": args.risk_lr,
        "enable_report_relation_features": bool(args.enable_report_relation_features),
        "allowed_families": args.allowed_families,
        "blocked_families": args.blocked_families,
        "allowed_source_prefixes": args.allowed_source_prefixes,
        "blocked_source_substrings": args.blocked_source_substrings,
        "folds": args.folds,
        "observations_cache_json": args.observations_cache_json,
        "reuse_observations_cache": bool(args.reuse_observations_cache),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL))
    parser.add_argument("--gt-jsonl", default=str(DEFAULT_GT))
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--ocr-layout-cache-dir", default=str(DEFAULT_OCR_LAYOUT_CACHE))
    parser.add_argument("--ocr-layout-model", default="qwen-vl-ocr")
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel", "auto"], default="normalized-1000")
    parser.add_argument("--max-candidates", type=int, default=900)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--candidate-preselect", choices=["ranked", "family_quota", "all"], default="ranked")
    parser.add_argument("--candidate-scope", choices=["all", "lowloc", "loc0"], default="all")
    parser.add_argument("--candidate-scope-languages", default="")
    parser.add_argument("--family-quota", type=int, default=8)
    parser.add_argument("--preselect-limit", type=int, default=96)
    parser.add_argument("--enable-linegrid-candidates", action="store_true")
    parser.add_argument("--enable-scriptgrid-candidates", action="store_true")
    parser.add_argument("--apply-top-n", type=int, default=2)
    parser.add_argument("--target-mode", choices=["append", "replace", "best"], default="append")
    parser.add_argument("--apply-mode", choices=["append", "replace"], default="append")
    parser.add_argument("--apply-max-area-ratio", type=float, default=0.035)
    parser.add_argument("--apply-duplicate-iou", type=float, default=0.50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rank-model", choices=["ridge", "knn", "family_ridge"], default="ridge")
    parser.add_argument("--ridge-alpha", type=float, default=3.0)
    parser.add_argument("--knn-k", type=int, default=25)
    parser.add_argument("--knn-distance-weight", action="store_true")
    parser.add_argument("--family-model-min-train", type=int, default=80)
    parser.add_argument("--risk-gate-model", choices=["none", "logistic", "family_logistic"], default="none")
    parser.add_argument("--risk-threshold", type=float, default=0.50)
    parser.add_argument("--risk-l2", type=float, default=1.0)
    parser.add_argument("--risk-iters", type=int, default=240)
    parser.add_argument("--risk-lr", type=float, default=0.15)
    parser.add_argument("--enable-report-relation-features", action="store_true")
    parser.add_argument("--allowed-families", default="")
    parser.add_argument("--blocked-families", default="")
    parser.add_argument("--allowed-source-prefixes", default="")
    parser.add_argument("--blocked-source-substrings", default="")
    parser.add_argument("--sample-weight-mode", choices=["none", "lowloc", "lowloc_positive"], default="none")
    parser.add_argument("--lowloc-weight", type=float, default=4.0)
    parser.add_argument("--lowloc-weight-threshold", type=float, default=0.02)
    parser.add_argument("--linegrid-guard", choices=["none", "no-th-linegrid"], default="none")
    parser.add_argument("--linegrid-min-match-score", type=float, default=0.0)
    parser.add_argument("--pred-delta-threshold", type=float, default=0.001)
    parser.add_argument("--pred-delta-max", type=float, default=None, help="Optional upper bound for selected candidate predicted delta.")
    parser.add_argument("--candidate-min-area", type=float, default=0.0, help="Optional selected candidate minimum pixel area.")
    parser.add_argument("--candidate-max-area", type=float, default=0.0, help="Optional selected candidate maximum pixel area; 0 disables.")
    parser.add_argument("--shape-gate-stage", choices=["pre", "post"], default="pre", help="Apply shape gate before selection with refill, or after selection without refill.")
    parser.add_argument("--observations-cache-json", default="")
    parser.add_argument("--reuse-observations-cache", action="store_true")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--applied-raw-jsonl", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    cache_path = resolve_pipe_path(args.observations_cache_json) if args.observations_cache_json else None
    if cache_path and args.reuse_observations_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        observations = payload["observations"]
        sample_rows = payload["sample_rows"]
        for obs in observations:
            refresh_observation_target(obs, args)
        for sample in sample_rows:
            sample["selected"] = []
        append_report_relation_features(observations, sample_rows, args)
        print(f"loaded observations cache: {cache_path}", file=sys.stderr)
    else:
        observations, sample_rows = build_observations(args)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "observations": observations,
                "sample_rows": sample_rows,
                "meta": {
                    "input_jsonl": args.input_jsonl,
                    "eval_json": args.eval_json,
                    "candidate_preselect": args.candidate_preselect,
                    "candidate_scope": args.candidate_scope,
                    "candidate_scope_languages": args.candidate_scope_languages,
                    "family_quota": args.family_quota,
                    "preselect_limit": args.preselect_limit,
                    "max_candidates": args.max_candidates,
                    "candidate_limit": args.candidate_limit,
                    "enable_linegrid_candidates": bool(args.enable_linegrid_candidates),
                    "enable_scriptgrid_candidates": bool(args.enable_scriptgrid_candidates),
                    "target_mode": args.target_mode,
                    "apply_mode": args.apply_mode,
                    "apply_max_area_ratio": args.apply_max_area_ratio,
                    "apply_duplicate_iou": args.apply_duplicate_iou,
                    "features_version": "v518_language_replace_context",
                },
            }
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print(f"wrote observations cache: {cache_path}", file=sys.stderr)
    append_report_relation_features(observations, sample_rows, args)
    cv_summary = crossval_select(observations, sample_rows, args)
    write_outputs(observations, sample_rows, args, cv_summary)


if __name__ == "__main__":
    main()
