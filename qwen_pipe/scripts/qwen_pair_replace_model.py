#!/usr/bin/env python3
"""Pair-level localization replacement model for Qwen-pipe.

This is the deployable follow-up to the v93 replacement upper bound.  Instead
of letting GT choose which final-report box a candidate should replace, this
script enumerates GT-blind actions:

    (candidate box, existing grounding index) -> replace
    (candidate box, no existing index) -> append

GT masks are used only after those actions exist, to train a small ridge model
in sample-level cross validation.  At application time each sample is scored by
a model that did not train on that sample, and selected actions are written back
using only inference-time candidate/OCR/image geometry features.
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
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v95_fp_rejector_t065_fixed_300.jsonl"
DEFAULT_EVAL = PIPE_ROOT / "outputs/eval/qwen_pipe_v95_fp_rejector_t065_fixed_300.json"
DEFAULT_GT = DEFAULT_DEBUG_ROOT / "data/val_300.jsonl"
DEFAULT_OCR_LAYOUT_CACHE = DEFAULT_DEBUG_ROOT / "outputs/cache/ocr_layouts"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_candidate_delta_model import (  # noqa: E402
    candidate_features,
    fit_ridge,
    max_contain_overlap,
    predict_ridge,
    region_stats,
    replace_groundings,
    ring_box,
    stable_fold,
)
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import (  # noqa: E402
    box_iou,
    choose_topk_v145ocranchor,
    choose_topk_v89token,
    choose_topk_v87mix,
    generate_candidates,
    insert_extra_anomalies,
    mask_stats,
    page_area_ratio,
    read_gt_mask,
    read_jsonl,
    sample_id_from_row,
    select_apply_extras,
)
from qwen_text_crop_verify import clamp_box, resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


def conclusion_is_forged(row: dict[str, Any]) -> bool:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper() == "FORGED"


def ocr_layout_language(row: dict[str, Any]) -> str:
    layout = (row.get("stage_outputs") or {}).get("ocr_layout") or {}
    raw = layout.get("raw")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return str(parsed.get("document_language") or "")
        except json.JSONDecodeError:
            return ""
    if isinstance(layout, dict):
        return str(layout.get("document_language") or "")
    return ""


def language_code(row: dict[str, Any]) -> str:
    return str(
        row.get("language_code")
        or ((row.get("metadata") or {}).get("language_code"))
        or ocr_layout_language(row)
        or ""
    )


def box_area(box: list[int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def inter_area(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def contain_fraction(inner: list[int], outer: list[int]) -> float:
    return inter_area(inner, outer) / max(1.0, box_area(inner))


def box_center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def text_pattern_types(text: str) -> set[str]:
    lower = text.lower()
    patterns: set[str] = set()
    if re.search(r"https?://|www\.|\.(com|org|net|edu|gov|co|id|th|cn|uk)\b", lower):
        patterns.add("url")
    if re.search(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
        r"|\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
        r"january|february|march|april|june|july|august|september|october|november|december)\b"
        r"|\b(19|20)\d{2}\b",
        lower,
    ):
        patterns.add("date")
    if re.search(r"[$€£¥₹]|\b(rs|usd|eur|gbp|idr|rp|thb|baht)\b|\b\d+[,.]\d{2}\b", lower):
        patterns.add("amount")
    if re.search(r"\b\d+(\.\d+)?\s*%", lower):
        patterns.add("percent")
    if re.search(r"\b[A-Z]{1,4}[- ]?\d{3,}\b", text):
        patterns.add("id")
    if re.search(r"\d", text):
        patterns.add("number")
    if re.search(r"[\u4e00-\u9fff]", text):
        patterns.add("cjk")
    if re.search(r"[\u0e00-\u0e7f]", text):
        patterns.add("thai")
    if re.search(r"[\u0600-\u06ff]", text):
        patterns.add("arabic")
    return patterns


HEADER_TOKEN_RE = re.compile(
    r"\b(issn|isbn|doi|vol\.?|volume|journal|proceedings|conference|page|pp\.)\b",
    re.IGNORECASE,
)
PAGE_FOOTER_RE = re.compile(
    r"\bpage\s*\d+\s*(?:of|/)\s*\d+\b|"
    r"\bpage\s*\d+\b|"
    r"^\s*\d+\s*(?:o|of)\s*::\s*page\s+\d+",
    re.IGNORECASE,
)
LINEGRID_SHORT_NUMERIC_HEADER_RE = re.compile(
    r"^\s*[\d٠-٩۰-۹]+(?:[.\-/]\d+)?\s*::\s*"
    r"[\d٠-٩۰-۹][\d٠-٩۰-۹\s.,/\-]{2,24}\s*$"
)


def candidate_rank_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    meta = candidate.get("meta") or {}
    return meta.get("v87b_rank") or meta.get("v87_rank") or {}


def should_reject_ocr_replace(obs: dict[str, Any], occupied: list[list[int]], args: argparse.Namespace) -> str:
    """GT-free guard for OCR-span replacements that repeatedly hurt localization.

    The pair model can over-trust standalone OCR spans when the text weakly
    matches the report.  This guard only blocks the narrow high-risk slice:
    low-query page headers and long plain prose OCR spans with little evidence.
    Numeric/date/value spans remain eligible because they are common true wins.
    """
    if args.ocr_replace_guard == "off":
        return ""
    action = str(obs.get("action") or "")
    # qwen_candidate_delta_model caches are replacement-only and may omit the
    # action field; treat a valid replace_index as a replacement action.
    if action and action != "replace":
        return ""
    candidate = obs.get("candidate") or {}
    if str(candidate.get("family") or "") not in {"ocr", "linegrid", "row"}:
        return ""
    source = str(candidate.get("source") or "")
    if not (source.startswith("ocr_span") or source.startswith("ocr_linegrid") or source.startswith("ocr_row")):
        return ""

    text = str(candidate.get("text") or "").strip()
    box = candidate.get("box") or []
    if len(box) < 4:
        return ""
    rank = candidate_rank_payload(candidate)
    query_score = float(rank.get("query_score") or 0.0)
    token_hit_count = int((candidate.get("meta") or {}).get("token_hit_count") or rank.get("token_hit_count") or 0)
    patterns = text_pattern_types(text)
    digit_count = sum(1 for ch in text if ch.isdigit())
    digit_ratio = digit_count / max(1, len(text))
    numeric_value_signal = bool(patterns & {"date", "amount", "percent", "number"}) and (
        len(text) <= args.ocr_guard_value_text_max_chars or digit_ratio >= args.ocr_guard_digit_ratio
    )
    text_has_value_signal = bool(patterns & {"url", "id"}) or numeric_value_signal

    page_h = max([float(box[3])] + [float(b[3]) for b in occupied if len(b) >= 4] + [1.0])
    y1_ratio = float(box[1]) / page_h
    y2_ratio = float(box[3]) / page_h
    if (
        HEADER_TOKEN_RE.search(text)
        and y1_ratio <= args.ocr_guard_header_y_ratio
        and query_score < args.ocr_guard_header_max_query
    ):
        return "header_low_query"
    if (
        source.startswith(("ocr_linegrid", "ocr_row"))
        and PAGE_FOOTER_RE.search(text)
    ):
        return "footer_page_noise"
    if (
        source.startswith(("ocr_linegrid", "ocr_row"))
        and y1_ratio <= args.ocr_guard_header_y_ratio
        and LINEGRID_SHORT_NUMERIC_HEADER_RE.search(text)
    ):
        return "numeric_header_linegrid_noise"

    if (
        len(text) >= args.ocr_guard_long_text_min_chars
        and query_score < args.ocr_guard_long_text_max_query
        and token_hit_count <= args.ocr_guard_long_text_max_token_hits
        and not text_has_value_signal
    ):
        return "long_plain_low_query"

    return ""


def old_box_features(
    *,
    old_box: list[int] | None,
    old_index: int,
    image: Image.Image,
    width: int,
    height: int,
    existing_count: int,
) -> list[float]:
    if old_box is None:
        return [0.0] * 20
    x1, y1, x2, y2 = old_box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    stats = region_stats(image, old_box)
    outer = region_stats(image, ring_box(old_box, width, height) or old_box)
    cx, cy = box_center(old_box)
    return [
        1.0,
        float(old_index) / max(1.0, float(existing_count - 1)),
        page_area_ratio(old_box, width, height),
        bw / max(1, width),
        bh / max(1, height),
        math.log1p(bw / max(1, bh)),
        cx / max(1, width),
        cy / max(1, height),
        stats["mean"] / 255.0,
        stats["std"] / 128.0,
        stats["edge"] / 64.0,
        stats["sat"],
        stats["red"],
        stats["yellow"],
        stats["dark"],
        stats["bright"],
        abs(stats["mean"] - outer["mean"]) / 255.0,
        abs(stats["edge"] - outer["edge"]) / 64.0,
        float(existing_count),
        max_contain_overlap(old_box, []),
    ]


def pair_features(
    *,
    candidate: dict[str, Any],
    old_box: list[int] | None,
    old_index: int,
    image: Image.Image,
    width: int,
    height: int,
    existing: list[list[int]],
) -> list[float]:
    cand_box = candidate["box"]
    cand_vec = candidate_features(candidate=candidate, image=image, width=width, height=height, existing=existing)
    old_vec = old_box_features(
        old_box=old_box,
        old_index=old_index,
        image=image,
        width=width,
        height=height,
        existing_count=len(existing),
    )
    cand_area = box_area(cand_box)
    old_area = box_area(old_box) if old_box is not None else 0.0
    cx, cy = box_center(cand_box)
    if old_box is None:
        pair_vec = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            cand_area / max(1.0, width * height),
            0.0,
            0.0,
            cx / max(1, width),
            cy / max(1, height),
        ]
    else:
        ox, oy = box_center(old_box)
        old_w = max(1, old_box[2] - old_box[0])
        old_h = max(1, old_box[3] - old_box[1])
        cand_w = max(1, cand_box[2] - cand_box[0])
        cand_h = max(1, cand_box[3] - cand_box[1])
        old_diag = max(1.0, float(old_w * old_w + old_h * old_h) ** 0.5)
        page_diag = max(1.0, float(width * width + height * height) ** 0.5)
        center_dist = float((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
        pair_vec = [
            0.0,
            box_iou(cand_box, old_box),
            contain_fraction(cand_box, old_box),
            contain_fraction(old_box, cand_box),
            center_dist / old_diag,
            center_dist / page_diag,
            cand_area / max(1.0, old_area),
            cand_area / max(1.0, width * height),
            cand_w / max(1, old_w),
            cand_h / max(1, old_h),
            (cx - ox) / max(1, width),
            (cy - oy) / max(1, height),
        ]
    meta = candidate.get("meta") or {}
    rank = meta.get("v87b_rank") or meta.get("v87_rank") or {}
    text = str(candidate.get("text") or "")
    pattern_types = text_pattern_types(text)
    number_hits = meta.get("number_hits") or rank.get("number_hits") or []
    query_hits = meta.get("query_hits") or rank.get("query_hits") or []
    visual_hits = meta.get("visual_hits") or rank.get("visual_hits") or []
    token_hit_count = int(meta.get("token_hit_count") or rank.get("token_hit_count") or 0)
    match_score = float(meta.get("match_score") or 0.0)
    span_box = meta.get("span_box") or cand_box
    if not isinstance(span_box, list) or len(span_box) < 4:
        span_box = cand_box
    span_h = max(1.0, float(span_box[3]) - float(span_box[1]))
    cand_h = max(1.0, float(cand_box[3]) - float(cand_box[1]))
    span_height_ratio = cand_h / span_h
    candidate_area = box_area(cand_box)
    id_route = "id" in pattern_types and "date" not in pattern_types and span_height_ratio >= 1.20 and match_score >= 8.0
    hnum_route = span_height_ratio >= 1.63 and match_score >= 5.0 and bool(number_hits)
    cjknum_route = (
        "cjk" in pattern_types
        and "number" in pattern_types
        and span_height_ratio >= 1.25
        and match_score >= 8.0
        and len(number_hits) >= 3
        and candidate_area <= 60000
    )
    scriptnum_route = (
        not cjknum_route
        and bool(pattern_types & {"thai", "arabic"})
        and "number" in pattern_types
        and span_height_ratio >= 1.25
        and match_score >= 8.0
        and len(number_hits) >= 2
        and candidate_area <= 60000
    )
    latintext_route = (
        not pattern_types
        and span_height_ratio >= 1.20
        and match_score >= 9.0
        and token_hit_count >= 4
        and candidate_area <= 40000
    )
    typed_vec = [
        1.0 if "url" in pattern_types else 0.0,
        1.0 if "date" in pattern_types else 0.0,
        1.0 if "amount" in pattern_types else 0.0,
        1.0 if "percent" in pattern_types else 0.0,
        1.0 if "id" in pattern_types else 0.0,
        1.0 if "number" in pattern_types else 0.0,
        1.0 if "cjk" in pattern_types else 0.0,
        1.0 if "thai" in pattern_types else 0.0,
        1.0 if "arabic" in pattern_types else 0.0,
        min(1.0, float(len(number_hits)) / 8.0),
        min(1.0, float(token_hit_count) / 8.0),
        min(1.0, float(len(query_hits)) / 8.0),
        min(1.0, float(len(visual_hits)) / 4.0),
        1.0 if id_route else 0.0,
        1.0 if hnum_route else 0.0,
        1.0 if cjknum_route else 0.0,
        1.0 if scriptnum_route else 0.0,
        1.0 if latintext_route else 0.0,
        1.0 if old_box is None else 0.0,
        1.0 if old_box is not None else 0.0,
    ]
    return cand_vec + old_vec + pair_vec + typed_vec


def fit_regression_tree(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int,
    min_leaf: int,
    n_quantiles: int,
) -> dict[str, Any]:
    value = float(y.mean()) if len(y) else 0.0
    if max_depth <= 0 or len(y) < max(2, min_leaf * 2) or float(y.var()) <= 1e-12:
        return {"leaf": value}
    base_sse = float(((y - value) ** 2).sum())
    best: tuple[float, int, float, np.ndarray] | None = None
    qs = np.linspace(0.08, 0.92, max(2, n_quantiles))
    for feat_idx in range(x.shape[1]):
        col = x[:, feat_idx]
        if float(col.max() - col.min()) <= 1e-12:
            continue
        thresholds = np.unique(np.quantile(col, qs))
        for threshold in thresholds:
            left_mask = col <= threshold
            left_count = int(left_mask.sum())
            right_count = len(y) - left_count
            if left_count < min_leaf or right_count < min_leaf:
                continue
            left = y[left_mask]
            right = y[~left_mask]
            left_mean = float(left.mean())
            right_mean = float(right.mean())
            sse = float(((left - left_mean) ** 2).sum() + ((right - right_mean) ** 2).sum())
            gain = base_sse - sse
            if best is None or gain > best[0]:
                best = (gain, feat_idx, float(threshold), left_mask)
    if best is None or best[0] <= 1e-12:
        return {"leaf": value}
    _gain, feat_idx, threshold, left_mask = best
    return {
        "feature": int(feat_idx),
        "threshold": float(threshold),
        "left": fit_regression_tree(
            x[left_mask],
            y[left_mask],
            max_depth=max_depth - 1,
            min_leaf=min_leaf,
            n_quantiles=n_quantiles,
        ),
        "right": fit_regression_tree(
            x[~left_mask],
            y[~left_mask],
            max_depth=max_depth - 1,
            min_leaf=min_leaf,
            n_quantiles=n_quantiles,
        ),
    }


def predict_regression_tree(x: np.ndarray, tree: dict[str, Any]) -> np.ndarray:
    if "leaf" in tree:
        return np.full(x.shape[0], float(tree["leaf"]), dtype=np.float64)
    feat_idx = int(tree["feature"])
    threshold = float(tree["threshold"])
    left_mask = x[:, feat_idx] <= threshold
    out = np.zeros(x.shape[0], dtype=np.float64)
    if left_mask.any():
        out[left_mask] = predict_regression_tree(x[left_mask], tree["left"])
    if (~left_mask).any():
        out[~left_mask] = predict_regression_tree(x[~left_mask], tree["right"])
    return out


def fit_gbdt_regressor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_leaf: int,
    n_quantiles: int,
) -> dict[str, Any]:
    init = float(y.mean()) if len(y) else 0.0
    pred = np.full(len(y), init, dtype=np.float64)
    trees: list[dict[str, Any]] = []
    for _ in range(max(0, n_estimators)):
        residual = y - pred
        tree = fit_regression_tree(
            x,
            residual,
            max_depth=max_depth,
            min_leaf=min_leaf,
            n_quantiles=n_quantiles,
        )
        update = predict_regression_tree(x, tree)
        if float(np.abs(update).max(initial=0.0)) <= 1e-12:
            break
        pred += learning_rate * update
        trees.append(tree)
    return {"init": init, "learning_rate": learning_rate, "trees": trees}


def predict_gbdt_regressor(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    pred = np.full(x.shape[0], float(model.get("init") or 0.0), dtype=np.float64)
    lr = float(model.get("learning_rate") or 0.0)
    for tree in model.get("trees") or []:
        pred += lr * predict_regression_tree(x, tree)
    return pred


def action_delta(
    *,
    gt_mask: np.ndarray | None,
    existing: list[list[int]],
    candidate_box: list[int],
    replace_index: int,
    width: int,
    height: int,
    base_f1: float,
) -> tuple[float, float, str]:
    if gt_mask is None:
        return 0.0, base_f1, "unknown"
    if replace_index >= 0:
        if replace_index >= len(existing):
            return 0.0, base_f1, "invalid"
        boxes = list(existing)
        boxes[replace_index] = candidate_box
        action = "replace"
    else:
        boxes = list(existing) + [candidate_box]
        action = "append"
    new_f1 = float(mask_stats(gt_mask, boxes, width, height).get("mask_f1") or 0.0)
    return new_f1 - base_f1, new_f1, action


def load_gt_rows(path: str, debug_root: Path) -> dict[str, dict[str, Any]]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = debug_root / p
    return {
        str(r.get("sample_id") or Path(str(r.get("image_file") or "")).stem): r
        for r in read_jsonl(p)
    }


def build_observations(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    raw_rows = read_jsonl(resolve_pipe_path(args.input_jsonl))
    gt_rows = load_gt_rows(args.gt_jsonl, debug_root)

    observations: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for row in raw_rows:
        sid = sample_id_from_row(row)
        sample_record: dict[str, Any] = {"sample_id": sid, "row": row, "selected": []}
        if not conclusion_is_forged(row):
            skipped["not_predicted_forged"] += 1
            sample_rows.append(sample_record)
            continue
        image_path = resolve_image_path(row, debug_root)
        if not image_path:
            skipped["missing_image"] += 1
            sample_rows.append(sample_record)
            continue
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        width, height = image.size
        existing = report_boxes(str(row.get("raw_output") or ""))
        lang = language_code(row)
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
            language_code=lang,
        )
        if args.rank_mode == "v145ocranchor":
            top = choose_topk_v145ocranchor(row, candidates, args.top_k, width, height, language_code=lang)
        elif args.rank_mode == "v87mix":
            top = choose_topk_v87mix(row, candidates, args.top_k, width, height)
        else:
            top = choose_topk_v89token(row, candidates, args.top_k, width, height)
        top_dicts = [
            {
                "label": c.label,
                "box": c.box,
                "source": c.source,
                "family": c.family,
                "score": c.score,
                "text": c.text,
                "meta": c.meta,
            }
            for c in top
        ]
        extras = select_apply_extras(
            top_dicts,
            existing,
            width=width,
            height=height,
            limit=args.candidate_limit,
            max_area_ratio=args.apply_max_area_ratio,
            duplicate_iou=args.candidate_duplicate_iou,
        )
        gt_mask = None
        gt_row = gt_rows.get(sid)
        if gt_row:
            gt_mask = read_gt_mask(gt_row, width, height, debug_root)
        base_f1 = float(mask_stats(gt_mask, existing, width, height).get("mask_f1") or 0.0) if gt_mask is not None else 0.0
        sample_obs: list[int] = []
        for cand_idx, cand in enumerate(extras):
            box = clamp_box([float(v) for v in cand["box"][:4]], width, height)
            if not box:
                continue
            cand = dict(cand)
            cand["box"] = box
            action_indices = list(range(len(existing)))
            if args.allow_append_actions:
                action_indices.append(-1)
            if not action_indices:
                action_indices = [-1]
            for replace_index in action_indices:
                old_box = existing[replace_index] if replace_index >= 0 and replace_index < len(existing) else None
                fvec = pair_features(
                    candidate=cand,
                    old_box=old_box,
                    old_index=replace_index,
                    image=image,
                    width=width,
                    height=height,
                    existing=existing,
                )
                delta, new_f1, action = action_delta(
                    gt_mask=gt_mask,
                    existing=existing,
                    candidate_box=box,
                    replace_index=replace_index,
                    width=width,
                    height=height,
                    base_f1=base_f1,
                )
                obs = {
                    "sample_id": sid,
                    "candidate_index": cand_idx,
                    "replace_index": replace_index,
                    "action": "replace" if replace_index >= 0 else "append",
                    "candidate": cand,
                    "features": fvec,
                    "base_f1": base_f1,
                    "new_f1": new_f1,
                    "delta": delta,
                    "fold": stable_fold(sid, args.folds),
                }
                sample_obs.append(len(observations))
                observations.append(obs)
        sample_record["candidate_indices"] = sample_obs
        sample_record["base_box_count"] = len(existing)
        sample_rows.append(sample_record)

    for sample in sample_rows:
        sample["build_skips"] = dict(skipped)
    return observations, sample_rows


def load_observation_cache(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cache_path = resolve_pipe_path(path)
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    observations = list(data.get("observations") or [])
    sample_rows = list(data.get("sample_rows") or [])
    for sample in sample_rows:
        sample["selected"] = []
    for obs in observations:
        obs.pop("pred_delta", None)
        obs.pop("pred_pos", None)
        obs.pop("pred_gain", None)
        obs.pop("selection_score", None)
    meta = dict(data.get("meta") or {})
    return observations, sample_rows, meta


def write_observation_cache(
    path: str | Path,
    observations: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    cache_path = resolve_pipe_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
            "gt_jsonl": str(args.gt_jsonl),
            "debug_root": str(Path(args.debug_root).expanduser().resolve()),
            "ocr_layout_cache_dir": str(Path(args.ocr_layout_cache_dir).expanduser().resolve()),
            "ocr_layout_model": args.ocr_layout_model,
            "coord_mode": args.coord_mode,
            "max_candidates": args.max_candidates,
            "top_k": args.top_k,
            "candidate_limit": args.candidate_limit,
            "apply_max_area_ratio": args.apply_max_area_ratio,
            "candidate_duplicate_iou": args.candidate_duplicate_iou,
            "allow_append_actions": bool(args.allow_append_actions),
            "enable_linegrid_candidates": bool(args.enable_linegrid_candidates),
            "enable_scriptgrid_candidates": bool(args.enable_scriptgrid_candidates),
            "rank_mode": args.rank_mode,
            "pair_observation_count": len(observations),
            "sample_count": len(sample_rows),
        },
        "observations": observations,
        "sample_rows": [{**sample, "selected": []} for sample in sample_rows],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def crossval_select(observations: list[dict[str, Any]], sample_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not observations:
        return {"candidate_count": 0, "selected_samples": 0, "selected_actions": 0}
    selection_family_allowlist = {
        item.strip()
        for item in str(args.selection_family_allowlist or "").split(",")
        if item.strip()
    }
    selection_action_allowlist = {
        item.strip()
        for item in str(args.selection_action_allowlist or "").split(",")
        if item.strip()
    }
    training_family_allowlist = {
        item.strip()
        for item in str(args.training_family_allowlist or "").split(",")
        if item.strip()
    }
    training_action_allowlist = {
        item.strip()
        for item in str(args.training_action_allowlist or "").split(",")
        if item.strip()
    }
    linegrid_required_patterns = {
        item.strip()
        for item in str(args.linegrid_required_patterns or "").split(",")
        if item.strip()
    }
    linegrid_excluded_patterns = {
        item.strip()
        for item in str(args.linegrid_excluded_patterns or "").split(",")
        if item.strip()
    }
    x_all = np.asarray([o["features"] for o in observations], dtype=np.float64)
    y_all = np.asarray([float(o["delta"]) for o in observations], dtype=np.float64)
    y_pos = (y_all > 0.0).astype(np.float64)
    y_gain = np.maximum(y_all, 0.0)
    base_pos_rate = float(y_pos.mean())
    pred = np.zeros(len(observations), dtype=np.float64)
    pred_pos = np.zeros(len(observations), dtype=np.float64)
    pred_gain = np.zeros(len(observations), dtype=np.float64)
    folds_used: list[int] = []

    def obs_family(index: int) -> str:
        return str(((observations[index].get("candidate") or {}).get("family")) or "")

    def obs_action(index: int) -> str:
        return str(observations[index].get("action") or "")

    def training_allowed(index: int) -> bool:
        if training_family_allowlist and obs_family(index) not in training_family_allowlist:
            return False
        if training_action_allowlist and obs_action(index) not in training_action_allowlist:
            return False
        return True

    def fit_three_heads(indices: list[int]) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        return (
            fit_ridge(x_all[indices], y_all[indices], args.ridge_alpha),
            fit_ridge(x_all[indices], y_pos[indices], args.pos_ridge_alpha),
            fit_ridge(x_all[indices], y_gain[indices], args.gain_ridge_alpha),
        )

    def fit_gbdt_heads(indices: list[int]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        common = {
            "n_estimators": args.gbdt_estimators,
            "learning_rate": args.gbdt_learning_rate,
            "max_depth": args.gbdt_max_depth,
            "min_leaf": args.gbdt_min_leaf,
            "n_quantiles": args.gbdt_quantiles,
        }
        return (
            fit_gbdt_regressor(x_all[indices], y_all[indices], **common),
            fit_gbdt_regressor(x_all[indices], y_pos[indices], **common),
            fit_gbdt_regressor(x_all[indices], y_gain[indices], **common),
        )

    for fold in range(args.folds):
        train_idx = [i for i, o in enumerate(observations) if int(o["fold"]) != fold]
        test_idx = [i for i, o in enumerate(observations) if int(o["fold"]) == fold]
        if not train_idx or not test_idx:
            continue
        filtered_train_idx = [i for i in train_idx if training_allowed(i)]
        # Keep filtered training honest: if a narrow typed slice is too small
        # for the requested learner, fall back to global training instead of
        # creating degenerate folds.
        min_train = max(args.gbdt_min_leaf * 2, 20) if args.score_mode == "gbdt_two_head" else 20
        model_train_idx = filtered_train_idx if len(filtered_train_idx) >= min_train else train_idx
        if args.score_mode == "gbdt_two_head":
            global_delta_model, global_pos_model, global_gain_model = fit_gbdt_heads(model_train_idx)
        else:
            global_delta_model, global_pos_model, global_gain_model = fit_three_heads(model_train_idx)
        family_models: dict[str, tuple[Any, Any, Any]] = {}
        if args.score_mode == "family_two_head":
            families = sorted({obs_family(i) for i in train_idx})
            for family in families:
                fam_idx = [i for i in train_idx if obs_family(i) == family]
                if len(fam_idx) >= args.family_min_train:
                    family_models[family] = fit_three_heads(fam_idx)
        if args.score_mode == "family_two_head":
            for i in test_idx:
                delta_model, pos_model, gain_model = family_models.get(
                    obs_family(i),
                    (global_delta_model, global_pos_model, global_gain_model),
                )
                row_x = x_all[[i]]
                pred[i] = float(predict_ridge(row_x, delta_model)[0])
                pred_pos[i] = float(np.clip(predict_ridge(row_x, pos_model)[0], 0.0, 1.0))
                pred_gain[i] = float(max(0.0, predict_ridge(row_x, gain_model)[0]))
        else:
            if args.score_mode == "gbdt_two_head":
                pred[test_idx] = predict_gbdt_regressor(x_all[test_idx], global_delta_model)
                pred_pos[test_idx] = np.clip(predict_gbdt_regressor(x_all[test_idx], global_pos_model), 0.0, 1.0)
                pred_gain[test_idx] = np.maximum(0.0, predict_gbdt_regressor(x_all[test_idx], global_gain_model))
            else:
                pred[test_idx] = predict_ridge(x_all[test_idx], global_delta_model)
                pred_pos[test_idx] = np.clip(predict_ridge(x_all[test_idx], global_pos_model), 0.0, 1.0)
                pred_gain[test_idx] = np.maximum(0.0, predict_ridge(x_all[test_idx], global_gain_model))
        folds_used.append(fold)
    for idx, value in enumerate(pred):
        observations[idx]["pred_delta"] = float(value)
        observations[idx]["pred_pos"] = float(pred_pos[idx])
        observations[idx]["pred_gain"] = float(pred_gain[idx])
        if args.score_mode in {"two_head", "family_two_head", "gbdt_two_head"}:
            score = (pred_pos[idx] * pred_gain[idx]) + args.delta_mix * pred[idx]
        elif args.score_mode == "candidate_prior":
            cand = observations[idx].get("candidate") or {}
            meta = cand.get("meta") or {}
            rank = meta.get("v87b_rank") or meta.get("v87_rank") or {}
            score = float(rank.get("score") or cand.get("score") or 0.0)
        elif args.score_mode == "pos_delta":
            score = pred[idx] + args.pos_score_scale * (pred_pos[idx] - base_pos_rate)
        else:
            score = pred[idx]
        observations[idx]["selection_score"] = float(score)

    selected_samples = 0
    selected_actions = 0
    selected_action_counts: Counter[str] = Counter()
    selected_target_delta: list[float] = []
    ocr_guard_reject_counts: Counter[str] = Counter()

    for sample in sample_rows:
        indices = list(sample.get("candidate_indices") or [])
        ranked = sorted((observations[i] for i in indices), key=lambda o: float(o.get("selection_score") or 0.0), reverse=True)
        row = sample["row"]
        occupied = report_boxes(str(row.get("raw_output") or ""))
        picked: list[dict[str, Any]] = []
        used_replace_indices: set[int] = set()
        used_candidate_keys: set[tuple[int, int, int, int]] = set()
        used_family_counts: Counter[str] = Counter()
        used_linegrid_action_counts: Counter[str] = Counter()
        used_linegrid_route_counts: Counter[str] = Counter()
        for obs in ranked:
            family = str((obs.get("candidate") or {}).get("family") or "")
            action = str(obs.get("action") or "")
            if selection_family_allowlist and family not in selection_family_allowlist:
                continue
            if selection_action_allowlist and action not in selection_action_allowlist:
                continue
            guard_reason = should_reject_ocr_replace(obs, occupied, args)
            if guard_reason:
                ocr_guard_reject_counts[guard_reason] += 1
                continue
            if float(obs.get("selection_score") or 0.0) < args.score_threshold:
                continue
            if family == "linegrid":
                if args.linegrid_max_per_sample >= 0 and used_family_counts["linegrid"] >= args.linegrid_max_per_sample:
                    continue
                if float(obs.get("selection_score") or 0.0) < args.linegrid_score_threshold:
                    continue
                cand_meta = (obs.get("candidate") or {}).get("meta") or {}
                span_box = cand_meta.get("span_box") or []
                cand_box = (obs.get("candidate") or {}).get("box") or []
                cand_text = str((obs.get("candidate") or {}).get("text") or "")
                cand_patterns = text_pattern_types(cand_text)
                span_height_ratio = 0.0
                if len(span_box) >= 4 and len(cand_box) >= 4:
                    span_h = max(1.0, float(span_box[3]) - float(span_box[1]))
                    cand_h = max(1.0, float(cand_box[3]) - float(cand_box[1]))
                    span_height_ratio = cand_h / span_h
                if args.linegrid_route_preset in {
                    "id_or_hnum",
                    "id_or_hnum_or_cjknum",
                    "id_or_hnum_or_scriptnum",
                    "id_or_hnum_or_latintext",
                }:
                    match_score = float(cand_meta.get("match_score") or 0.0)
                    number_hits = cand_meta.get("number_hits") or []
                    token_hit_count = int(cand_meta.get("token_hit_count") or 0)
                    has_number_hit = bool(number_hits)
                    id_branch = ("id" in cand_patterns and "date" not in cand_patterns and span_height_ratio >= 1.20 and match_score >= 8.0)
                    hnum_branch = (span_height_ratio >= 1.63 and match_score >= 5.0 and has_number_hit)
                    cjknum_branch = (
                        args.linegrid_route_preset in {
                            "id_or_hnum_or_cjknum",
                            "id_or_hnum_or_scriptnum",
                            "id_or_hnum_or_latintext",
                        }
                        and "cjk" in cand_patterns
                        and "number" in cand_patterns
                        and span_height_ratio >= 1.25
                        and match_score >= 8.0
                        and len(number_hits) >= 3
                        and len(cand_box) >= 4
                        and ((float(cand_box[2]) - float(cand_box[0])) * (float(cand_box[3]) - float(cand_box[1]))) <= 60000
                    )
                    latintext_branch = (
                        args.linegrid_route_preset == "id_or_hnum_or_latintext"
                        and not cand_patterns
                        and span_height_ratio >= 1.20
                        and match_score >= 9.0
                        and token_hit_count >= 4
                        and len(cand_box) >= 4
                        and ((float(cand_box[2]) - float(cand_box[0])) * (float(cand_box[3]) - float(cand_box[1]))) <= 40000
                    )
                    scriptnum_branch = (
                        args.linegrid_route_preset == "id_or_hnum_or_scriptnum"
                        and not cjknum_branch
                        and bool(cand_patterns & {"thai", "arabic"})
                        and "number" in cand_patterns
                        and span_height_ratio >= 1.25
                        and match_score >= 8.0
                        and len(number_hits) >= 2
                        and len(cand_box) >= 4
                        and ((float(cand_box[2]) - float(cand_box[0])) * (float(cand_box[3]) - float(cand_box[1]))) <= 60000
                    )
                    if not (id_branch or hnum_branch or cjknum_branch or scriptnum_branch or latintext_branch):
                        continue
                    route_key = (
                        "id"
                        if id_branch
                        else "hnum"
                        if hnum_branch
                        else "cjknum"
                        if cjknum_branch
                        else "scriptnum"
                        if scriptnum_branch
                        else "latintext"
                    )
                    if route_key in {"cjknum", "scriptnum", "latintext"} and args.linegrid_cjknum_max_per_sample >= 0:
                        if used_linegrid_route_counts[route_key] >= args.linegrid_cjknum_max_per_sample:
                            continue
                    if (
                        route_key == "scriptnum"
                        and args.linegrid_scriptnum_max_existing_boxes >= 0
                        and len(occupied) > args.linegrid_scriptnum_max_existing_boxes
                    ):
                        continue
                    if (
                        route_key == "latintext"
                        and args.linegrid_latintext_max_existing_boxes >= 0
                        and len(occupied) > args.linegrid_latintext_max_existing_boxes
                    ):
                        continue
                    if (
                        action == "append"
                        and args.linegrid_max_append_per_sample >= 0
                        and used_linegrid_action_counts["append"] >= args.linegrid_max_append_per_sample
                    ):
                        continue
                    if (
                        action == "replace"
                        and args.linegrid_max_replace_per_sample >= 0
                        and used_linegrid_action_counts["replace"] >= args.linegrid_max_replace_per_sample
                    ):
                        continue
                    obs["_linegrid_route_key"] = route_key
                else:
                    if args.linegrid_min_span_height_ratio > 0:
                        if span_height_ratio < args.linegrid_min_span_height_ratio:
                            continue
                    if args.linegrid_min_match_score > 0:
                        if float(cand_meta.get("match_score") or 0.0) < args.linegrid_min_match_score:
                            continue
                    if args.linegrid_require_number_hit:
                        if not (cand_meta.get("number_hits") or []):
                            continue
                    if linegrid_required_patterns:
                        if not (cand_patterns & linegrid_required_patterns):
                            continue
                    if linegrid_excluded_patterns:
                        if cand_patterns & linegrid_excluded_patterns:
                            continue
            if float(obs.get("pred_delta") or 0.0) < args.pred_delta_threshold:
                continue
            min_pred_pos = args.min_pred_pos
            if family == "linegrid":
                min_pred_pos = max(min_pred_pos, args.linegrid_min_pred_pos)
            if float(obs.get("pred_pos") or 0.0) < min_pred_pos:
                continue
            cand = obs["candidate"]
            box = cand["box"]
            key = tuple(int(v) for v in box[:4])
            replace_index = int(obs.get("replace_index") if obs.get("replace_index") is not None else -1)
            if key in used_candidate_keys:
                continue
            if replace_index >= 0:
                if replace_index in used_replace_indices or replace_index >= len(occupied):
                    continue
                if args.max_center_dist_for_replace > 0:
                    cx, cy = box_center(box)
                    ox, oy = box_center(occupied[replace_index])
                    old = occupied[replace_index]
                    diag = max(1.0, float((old[2] - old[0]) ** 2 + (old[3] - old[1]) ** 2) ** 0.5)
                    if (float((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 / diag) > args.max_center_dist_for_replace:
                        continue
                used_replace_indices.add(replace_index)
                occupied[replace_index] = box
            else:
                if any(box_iou(box, prev) >= args.apply_duplicate_iou for prev in occupied):
                    continue
                occupied.append(box)
            picked.append(obs)
            used_candidate_keys.add(key)
            used_family_counts[family] += 1
            route_key = str(obs.get("_linegrid_route_key") or "")
            if route_key:
                used_linegrid_route_counts[route_key] += 1
            if family == "linegrid":
                used_linegrid_action_counts[str(obs.get("action") or "")] += 1
            selected_target_delta.append(float(obs.get("delta") or 0.0))
            selected_action_counts[str(obs.get("action") or "unknown")] += 1
            if len(picked) >= args.apply_top_n:
                break
        sample["selected"] = picked
        if picked:
            selected_samples += 1
            selected_actions += len(picked)

    return {
        "folds": folds_used,
        "pair_observation_count": len(observations),
        "target_delta_mean": float(y_all.mean()),
        "target_delta_positive_rate": float((y_all > 0).mean()),
        "target_delta_max": float(y_all.max()),
        "pred_delta_mean": float(pred.mean()),
        "pred_delta_max": float(pred.max()),
        "pred_pos_mean": float(pred_pos.mean()),
        "pred_pos_max": float(pred_pos.max()),
        "pred_gain_mean": float(pred_gain.mean()),
        "pred_gain_max": float(pred_gain.max()),
        "selected_samples": selected_samples,
        "selected_actions": selected_actions,
        "selected_action_counts": dict(selected_action_counts),
        "selected_target_delta_mean": float(np.mean(selected_target_delta)) if selected_target_delta else 0.0,
        "selected_target_delta_positive_rate": float(np.mean([v > 0 for v in selected_target_delta])) if selected_target_delta else 0.0,
        "ocr_guard_reject_counts": dict(ocr_guard_reject_counts),
    }


def write_outputs(observations: list[dict[str, Any]], sample_rows: list[dict[str, Any]], args: argparse.Namespace, summary: dict[str, Any]) -> None:
    from postprocess import parse_cct_report  # type: ignore

    diag_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    raw_path = resolve_pipe_path(args.applied_raw_jsonl)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
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
                        "base_box_count": sample.get("base_box_count", 0),
                        "pair_observation_count": len(obs_by_sample.get(sample["sample_id"], [])),
                        "selected_count": len(selected),
                        "selected": [
                            {
                                "action": s.get("action"),
                                "replace_index": s.get("replace_index"),
                                "candidate": s.get("candidate"),
                                "pred_delta": s.get("pred_delta"),
                                "pred_pos": s.get("pred_pos"),
                                "pred_gain": s.get("pred_gain"),
                                "selection_score": s.get("selection_score"),
                                "target_delta": s.get("delta"),
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

    with raw_path.open("w", encoding="utf-8") as fh:
        for sample in sample_rows:
            row = sample["row"]
            selected = sample.get("selected") or []
            if selected:
                row = dict(row)
                report = str(row.get("raw_output") or "")
                replacements = {
                    int(s["replace_index"]): s["candidate"]["box"]
                    for s in selected
                    if int(s.get("replace_index") if s.get("replace_index") is not None else -1) >= 0
                }
                append_extras = [
                    s["candidate"]
                    for s in selected
                    if int(s.get("replace_index") if s.get("replace_index") is not None else -1) < 0
                ]
                report, replaced_count = replace_groundings(report, replacements)
                if append_extras:
                    report = insert_extra_anomalies(report, append_extras)
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs["qwen_pipe_pair_replace_model"] = {
                    "applied": True,
                    "selected_actions": len(selected),
                    "replaced_count": replaced_count,
                    "appended_count": len(append_extras),
                    "pred_delta_threshold": args.pred_delta_threshold,
                    "apply_top_n": args.apply_top_n,
                    "candidate_limit": args.candidate_limit,
                    "policy": "Sample-level CV ridge ranks candidate-by-existing-box replace/append actions from GT-free OCR/candidate/image features.",
                    "selected": [
                            {
                                "action": s.get("action"),
                                "replace_index": s.get("replace_index"),
                                "candidate": s.get("candidate"),
                                "pred_delta": s.get("pred_delta"),
                                "pred_pos": s.get("pred_pos"),
                                "pred_gain": s.get("pred_gain"),
                                "selection_score": s.get("selection_score"),
                            }
                            for s in selected
                        ],
                }
                row["stage_outputs"] = stage_outputs
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    full_summary = {
        **summary,
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "applied_raw_jsonl": str(raw_path),
        "output_jsonl": str(diag_path),
        "apply_top_n": args.apply_top_n,
        "candidate_limit": args.candidate_limit,
        "top_k": args.top_k,
        "pred_delta_threshold": args.pred_delta_threshold,
        "score_mode": args.score_mode,
        "rank_mode": args.rank_mode,
        "score_threshold": args.score_threshold,
        "min_pred_pos": args.min_pred_pos,
        "linegrid_min_pred_pos": args.linegrid_min_pred_pos,
        "linegrid_score_threshold": args.linegrid_score_threshold,
        "linegrid_max_per_sample": args.linegrid_max_per_sample,
        "linegrid_min_span_height_ratio": args.linegrid_min_span_height_ratio,
        "linegrid_min_match_score": args.linegrid_min_match_score,
        "linegrid_require_number_hit": args.linegrid_require_number_hit,
        "linegrid_required_patterns": args.linegrid_required_patterns,
        "linegrid_excluded_patterns": args.linegrid_excluded_patterns,
        "linegrid_route_preset": args.linegrid_route_preset,
        "linegrid_cjknum_max_per_sample": args.linegrid_cjknum_max_per_sample,
        "linegrid_max_append_per_sample": args.linegrid_max_append_per_sample,
        "linegrid_max_replace_per_sample": args.linegrid_max_replace_per_sample,
        "selection_family_allowlist": args.selection_family_allowlist,
        "selection_action_allowlist": args.selection_action_allowlist,
        "training_family_allowlist": args.training_family_allowlist,
        "training_action_allowlist": args.training_action_allowlist,
        "ocr_replace_guard": args.ocr_replace_guard,
        "ocr_guard_long_text_min_chars": args.ocr_guard_long_text_min_chars,
        "ocr_guard_long_text_max_query": args.ocr_guard_long_text_max_query,
        "ocr_guard_long_text_max_token_hits": args.ocr_guard_long_text_max_token_hits,
        "ocr_guard_digit_ratio": args.ocr_guard_digit_ratio,
        "ocr_guard_value_text_max_chars": args.ocr_guard_value_text_max_chars,
        "ocr_guard_header_y_ratio": args.ocr_guard_header_y_ratio,
        "ocr_guard_header_max_query": args.ocr_guard_header_max_query,
        "delta_mix": args.delta_mix,
        "pos_score_scale": args.pos_score_scale,
        "family_min_train": args.family_min_train,
        "gbdt_estimators": args.gbdt_estimators,
        "gbdt_learning_rate": args.gbdt_learning_rate,
        "gbdt_max_depth": args.gbdt_max_depth,
        "gbdt_min_leaf": args.gbdt_min_leaf,
        "gbdt_quantiles": args.gbdt_quantiles,
        "ridge_alpha": args.ridge_alpha,
        "pos_ridge_alpha": args.pos_ridge_alpha,
        "gain_ridge_alpha": args.gain_ridge_alpha,
        "folds": args.folds,
        "observation_cache": str(resolve_pipe_path(args.observation_cache)) if args.observation_cache else "",
    }
    summary_path.write_text(json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(full_summary, ensure_ascii=False))


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
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--apply-top-n", type=int, default=2)
    parser.add_argument("--apply-max-area-ratio", type=float, default=0.035)
    parser.add_argument("--candidate-duplicate-iou", type=float, default=0.50)
    parser.add_argument("--apply-duplicate-iou", type=float, default=0.50)
    parser.add_argument("--max-center-dist-for-replace", type=float, default=2.50)
    parser.add_argument("--allow-append-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-linegrid-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-scriptgrid-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rank-mode", choices=["v89token", "v87mix", "v145ocranchor"], default="v89token")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--pos-ridge-alpha", type=float, default=5.0)
    parser.add_argument("--gain-ridge-alpha", type=float, default=5.0)
    parser.add_argument("--score-mode", choices=["delta", "two_head", "family_two_head", "gbdt_two_head", "candidate_prior", "pos_delta"], default="delta")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--pred-delta-threshold", type=float, default=0.0)
    parser.add_argument("--min-pred-pos", type=float, default=0.0)
    parser.add_argument("--linegrid-min-pred-pos", type=float, default=0.0)
    parser.add_argument("--linegrid-score-threshold", type=float, default=-1e9)
    parser.add_argument("--linegrid-max-per-sample", type=int, default=-1)
    parser.add_argument("--linegrid-min-span-height-ratio", type=float, default=0.0)
    parser.add_argument("--linegrid-min-match-score", type=float, default=0.0)
    parser.add_argument("--linegrid-require-number-hit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--linegrid-required-patterns", default="", help="Comma-separated text pattern types required for linegrid candidates: url,date,amount,percent,id,number.")
    parser.add_argument("--linegrid-excluded-patterns", default="", help="Comma-separated text pattern types that reject linegrid candidates.")
    parser.add_argument(
        "--linegrid-route-preset",
        choices=[
            "",
            "id_or_hnum",
            "id_or_hnum_or_cjknum",
            "id_or_hnum_or_scriptnum",
            "id_or_hnum_or_latintext",
        ],
        default="",
    )
    parser.add_argument("--linegrid-cjknum-max-per-sample", type=int, default=-1)
    parser.add_argument("--linegrid-scriptnum-max-existing-boxes", type=int, default=-1)
    parser.add_argument("--linegrid-latintext-max-existing-boxes", type=int, default=-1)
    parser.add_argument("--linegrid-max-append-per-sample", type=int, default=-1)
    parser.add_argument("--linegrid-max-replace-per-sample", type=int, default=-1)
    parser.add_argument("--selection-family-allowlist", default="", help="Comma-separated candidate families allowed at final action selection time.")
    parser.add_argument("--selection-action-allowlist", default="", help="Comma-separated action types allowed at final action selection time, e.g. replace,append.")
    parser.add_argument("--training-family-allowlist", default="", help="Comma-separated candidate families used to train the cross-fold scorer.")
    parser.add_argument("--training-action-allowlist", default="", help="Comma-separated action types used to train the cross-fold scorer.")
    parser.add_argument("--ocr-replace-guard", choices=["off", "conservative"], default="off")
    parser.add_argument("--ocr-guard-long-text-min-chars", type=int, default=28)
    parser.add_argument("--ocr-guard-long-text-max-query", type=float, default=4.0)
    parser.add_argument("--ocr-guard-long-text-max-token-hits", type=int, default=3)
    parser.add_argument("--ocr-guard-digit-ratio", type=float, default=0.18)
    parser.add_argument("--ocr-guard-value-text-max-chars", type=int, default=24)
    parser.add_argument("--ocr-guard-header-y-ratio", type=float, default=0.18)
    parser.add_argument("--ocr-guard-footer-y-ratio", type=float, default=0.78)
    parser.add_argument("--ocr-guard-header-max-query", type=float, default=6.0)
    parser.add_argument("--delta-mix", type=float, default=0.25)
    parser.add_argument("--pos-score-scale", type=float, default=0.03)
    parser.add_argument("--family-min-train", type=int, default=80)
    parser.add_argument("--gbdt-estimators", type=int, default=24)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.08)
    parser.add_argument("--gbdt-max-depth", type=int, default=2)
    parser.add_argument("--gbdt-min-leaf", type=int, default=30)
    parser.add_argument("--gbdt-quantiles", type=int, default=8)
    parser.add_argument("--observation-cache", default="", help="Load/save candidate-action observations for repeated CV scoring.")
    parser.add_argument("--rebuild-observation-cache", action="store_true", help="Ignore an existing observation cache and rebuild it.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--applied-raw-jsonl", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    if args.observation_cache and resolve_pipe_path(args.observation_cache).exists() and not args.rebuild_observation_cache:
        observations, sample_rows, cache_meta = load_observation_cache(args.observation_cache)
        print(
            json.dumps(
                {
                    "loaded_observation_cache": str(resolve_pipe_path(args.observation_cache)),
                    "pair_observation_count": len(observations),
                    "sample_count": len(sample_rows),
                    "cache_meta": cache_meta,
                },
                ensure_ascii=False,
            )
        )
    else:
        observations, sample_rows = build_observations(args)
        if args.observation_cache:
            write_observation_cache(args.observation_cache, observations, sample_rows, args)
    cv_summary = crossval_select(observations, sample_rows, args)
    write_outputs(observations, sample_rows, args, cv_summary)


if __name__ == "__main__":
    main()
