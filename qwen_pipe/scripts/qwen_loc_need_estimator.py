#!/usr/bin/env python3
"""GT-blind selector for applying v87 exhaustive-recall boxes.

v87 proved that OCR/evidence/visual candidates can cover many missing GT
regions, but the best gains so far used an evaluation-derived low-S_Loc sample
list.  This script tests the deployable counterpart: infer whether the current
report likely under-localizes from only raw prediction artifacts, OCR boxes,
stage evidence, and candidate/report overlap.

GT/eval data is optional and is used only for selector diagnostics after the
selection has already been made.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v85_embedding_v4_basekey_stablecache_k8_300.jsonl"
DEFAULT_EVAL = PIPE_ROOT / "outputs/eval/qwen_pipe_v85_embedding_v4_basekey_stablecache_k8_300.json"
DEFAULT_OCR_LAYOUT_CACHE = DEFAULT_DEBUG_ROOT / "outputs/cache/ocr_layouts"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_exhaustive_recall import (  # noqa: E402
    RecallCandidate,
    box_iou,
    choose_topk_v145ocranchor,
    choose_topk_v89token,
    choose_topk_v87mix,
    generate_candidates,
    page_area_ratio,
    read_jsonl,
    sample_id_from_row,
    select_apply_extras,
    insert_extra_anomalies,
)
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_text_crop_verify import resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


def conclusion_is_forged(row: dict[str, Any]) -> bool:
    parsed = row.get("parsed") or {}
    conclusion = str(parsed.get("conclusion") or "").upper()
    if conclusion:
        return conclusion == "FORGED"
    raw = str(row.get("raw_output") or "").upper()
    return "FORGED" in raw and "AUTHENTIC" not in raw[:800]


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


def candidate_rank_meta(candidate: dict[str, Any] | RecallCandidate) -> dict[str, Any]:
    meta = candidate.meta if isinstance(candidate, RecallCandidate) else candidate.get("meta") or {}
    return meta.get("v87b_rank") or meta.get("v87_rank") or {}


def max_iou_to_any(box: list[int], boxes: list[list[int]]) -> float:
    return max((box_iou(box, other) for other in boxes), default=0.0)


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


def compact_text(text: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def build_features(
    row: dict[str, Any],
    *,
    debug_root: Path,
    ocr_layout_cache_dir: Path,
    ocr_layout_model: str,
    coord_mode: str,
    max_candidates: int,
    top_k: int,
    apply_top_n: int,
    apply_max_area_ratio: float,
    apply_duplicate_iou: float,
    rank_mode: str,
    enable_token_candidates: bool,
    enable_linegrid_candidates: bool,
    enable_scriptgrid_candidates: bool,
) -> dict[str, Any]:
    sid = sample_id_from_row(row)
    image_path = resolve_image_path(row, debug_root)
    if not image_path:
        raise RuntimeError(f"missing image for {sid}")
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    width, height = image.size
    lang = language_code(row)
    candidates = generate_candidates(
        row,
        image,
        debug_root,
        ocr_layout_cache_dir,
        ocr_layout_model,
        coord_mode,
        max_candidates,
        enable_token_candidates=enable_token_candidates,
        enable_linegrid_candidates=enable_linegrid_candidates,
        enable_scriptgrid_candidates=enable_scriptgrid_candidates,
        language_code=lang,
    )
    if rank_mode == "v145ocranchor":
        top = choose_topk_v145ocranchor(row, candidates, top_k, width, height, language_code=lang)
    elif rank_mode == "v89token":
        top = choose_topk_v89token(row, candidates, top_k, width, height)
    else:
        top = choose_topk_v87mix(row, candidates, top_k, width, height)
    top_dicts = []
    for cand in top:
        top_dicts.append(
            {
                "label": cand.label,
                "box": cand.box,
                "source": cand.source,
                "family": cand.family,
                "score": cand.score,
                "text": cand.text,
                "meta": cand.meta,
            }
        )
    existing = report_boxes(str(row.get("raw_output") or ""))
    extras = select_apply_extras(
        top_dicts,
        existing,
        width=width,
        height=height,
        limit=apply_top_n,
        max_area_ratio=apply_max_area_ratio,
        duplicate_iou=apply_duplicate_iou,
    )
    families = Counter(c.family for c in candidates)
    top_families = Counter(c.get("family") for c in top_dicts)
    extra_families = Counter(c.get("family") for c in extras)
    existing_area = sum(page_area_ratio(b, width, height) for b in existing)
    extra_area = sum(page_area_ratio(c["box"], width, height) for c in extras)
    strong_extras = []
    uncovered_evidence = []
    uncovered_textual = []
    compact_extras = []
    for cand in extras:
        box = cand["box"]
        rank = candidate_rank_meta(cand)
        query_score = float(rank.get("query_score") or 0.0)
        rank_score = float(rank.get("score") or cand.get("score") or 0.0)
        source = str(cand.get("source") or "")
        family = str(cand.get("family") or "")
        area_ratio = page_area_ratio(box, width, height)
        novelty = 1.0 - max_contain_overlap(box, existing)
        is_textual = family in {"evidence", "ocr", "row"} or "span_id" in source
        is_strong = (
            is_textual
            and area_ratio <= 0.055
            and novelty >= 0.45
            and (query_score > 0.0 or "span_id" in source or rank_score >= 7.0)
        )
        if is_strong:
            strong_extras.append(cand)
        if family == "evidence" and novelty >= 0.35:
            uncovered_evidence.append(cand)
        if is_textual and novelty >= 0.45:
            uncovered_textual.append(cand)
        if area_ratio <= 0.018 and novelty >= 0.40:
            compact_extras.append(cand)

    high_query_extra_count = 0
    rank_scores: list[float] = []
    query_scores: list[float] = []
    novelty_scores: list[float] = []
    for cand in extras:
        rank = candidate_rank_meta(cand)
        q = float(rank.get("query_score") or 0.0)
        s = float(rank.get("score") or cand.get("score") or 0.0)
        query_scores.append(q)
        rank_scores.append(s)
        novelty_scores.append(1.0 - max_contain_overlap(cand["box"], existing))
        if q > 0:
            high_query_extra_count += 1

    return {
        "sample_id": sid,
        "image_name": row.get("image_name"),
        "width": width,
        "height": height,
        "language_code": lang,
        "is_pred_forged": conclusion_is_forged(row),
        "existing_count": len(existing),
        "existing_area_ratio": existing_area,
        "candidate_count": len(candidates),
        "candidate_family_counts": dict(families),
        "top_family_counts": dict(top_families),
        "extra_family_counts": dict(extra_families),
        "extra_count": len(extras),
        "extra_area_ratio": extra_area,
        "strong_extra_count": len(strong_extras),
        "uncovered_evidence_count": len(uncovered_evidence),
        "uncovered_textual_count": len(uncovered_textual),
        "compact_extra_count": len(compact_extras),
        "high_query_extra_count": high_query_extra_count,
        "mean_extra_rank_score": float(np.mean(rank_scores)) if rank_scores else 0.0,
        "max_extra_rank_score": float(np.max(rank_scores)) if rank_scores else 0.0,
        "mean_extra_query_score": float(np.mean(query_scores)) if query_scores else 0.0,
        "max_extra_query_score": float(np.max(query_scores)) if query_scores else 0.0,
        "mean_extra_novelty": float(np.mean(novelty_scores)) if novelty_scores else 0.0,
        "max_extra_novelty": float(np.max(novelty_scores)) if novelty_scores else 0.0,
        "top_candidates": top_dicts,
        "extras": extras,
    }


def need_score(features: dict[str, Any], mode: str) -> tuple[bool, float, list[str]]:
    if not features.get("is_pred_forged"):
        return False, 0.0, ["pred_not_forged"]
    reasons: list[str] = []
    score = 0.0
    existing_count = int(features.get("existing_count") or 0)
    existing_area = float(features.get("existing_area_ratio") or 0.0)
    extra_count = int(features.get("extra_count") or 0)
    strong_extra = int(features.get("strong_extra_count") or 0)
    uncovered_evidence = int(features.get("uncovered_evidence_count") or 0)
    uncovered_textual = int(features.get("uncovered_textual_count") or 0)
    compact_extra = int(features.get("compact_extra_count") or 0)
    high_query = int(features.get("high_query_extra_count") or 0)
    max_rank = float(features.get("max_extra_rank_score") or 0.0)
    mean_novelty = float(features.get("mean_extra_novelty") or 0.0)
    extra_area = float(features.get("extra_area_ratio") or 0.0)

    if existing_count <= 2:
        score += 1.15
        reasons.append("few_existing_boxes")
    elif existing_count <= 3:
        score += 0.65
        reasons.append("moderate_existing_boxes")
    if existing_area < 0.020:
        score += 0.55
        reasons.append("small_existing_area")
    if extra_count >= 2:
        score += 0.75
        reasons.append("multiple_novel_candidates")
    if strong_extra >= 2:
        score += 1.40
        reasons.append("strong_textual_extras")
    elif strong_extra == 1:
        score += 0.65
        reasons.append("one_strong_textual_extra")
    if uncovered_evidence >= 1:
        score += 0.85
        reasons.append("uncovered_stage_evidence")
    if uncovered_textual >= 3:
        score += 0.85
        reasons.append("many_uncovered_textual")
    elif uncovered_textual >= 1:
        score += 0.35
        reasons.append("some_uncovered_textual")
    if compact_extra >= 2:
        score += 0.50
        reasons.append("compact_extras")
    if high_query >= 1:
        score += 0.40
        reasons.append("query_matched_extra")
    if max_rank >= 8.0:
        score += 0.45
        reasons.append("high_rank_extra")
    if mean_novelty >= 0.70:
        score += 0.35
        reasons.append("novel_vs_existing")
    if extra_area > 0.080:
        score -= 0.55
        reasons.append("large_extra_area_penalty")
    if existing_count >= 5 and existing_area >= 0.055:
        score -= 1.20
        reasons.append("already_many_boxes")

    thresholds = {
        "recall": 2.35,
        "balanced": 3.05,
        "precision": 3.85,
    }
    threshold = thresholds[mode]
    return score >= threshold, score, reasons


def load_eval_samples(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(s.get("sample_id") or ""): s for s in data.get("samples") or []}


def summarize(rows: list[dict[str, Any]], eval_samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pred_forged = [r for r in rows if r.get("is_pred_forged")]
    selected = [r for r in pred_forged if r.get("selected")]
    selected_ids = {r["sample_id"] for r in selected}
    eval_pred_forged = [
        s
        for s in eval_samples.values()
        if s.get("gt_label") == "FORGED" and s.get("pred_label") == "FORGED"
    ]
    lowloc = [s for s in eval_pred_forged if float(s.get("loc_score") or 0.0) < 0.02]
    lowloc_ids = {str(s.get("sample_id") or "") for s in lowloc}
    selected_lowloc = sorted(selected_ids & lowloc_ids)
    selected_non_lowloc = sorted(selected_ids - lowloc_ids)
    forged_selected = [
        s for s in eval_pred_forged if str(s.get("sample_id") or "") in selected_ids
    ]
    forged_unselected = [
        s for s in eval_pred_forged if str(s.get("sample_id") or "") not in selected_ids
    ]

    def mean_loc(samples: list[dict[str, Any]]) -> float:
        return float(np.mean([float(s.get("loc_score") or 0.0) for s in samples])) if samples else 0.0

    return {
        "n": len(rows),
        "pred_forged": len(pred_forged),
        "selected_count": len(selected),
        "selected_boxes_to_add": sum(int(r.get("applied_extra_count") or 0) for r in selected),
        "selector_lowloc_diagnostic": {
            "lowloc_threshold": 0.02,
            "lowloc_count": len(lowloc),
            "selected_lowloc_count": len(selected_lowloc),
            "selected_lowloc_recall": len(selected_lowloc) / max(1, len(lowloc)),
            "selected_precision_vs_lowloc": len(selected_lowloc) / max(1, len(selected)),
            "selected_lowloc_sample_ids": selected_lowloc[:80],
            "selected_non_lowloc_count": len(selected_non_lowloc),
            "selected_non_lowloc_sample_ids": selected_non_lowloc[:80],
            "mean_loc_selected_forged": mean_loc(forged_selected),
            "mean_loc_unselected_forged": mean_loc(forged_unselected),
        },
        "feature_means_selected": feature_means(selected),
        "feature_means_unselected_pred_forged": feature_means([r for r in pred_forged if not r.get("selected")]),
        "interpretation": "Selection uses only raw prediction/OCR/evidence/candidate overlap. Eval loc labels are only used in this summary to measure whether the selector hits known low-S_Loc failures.",
    }


def feature_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "existing_count",
        "existing_area_ratio",
        "extra_count",
        "strong_extra_count",
        "uncovered_evidence_count",
        "uncovered_textual_count",
        "compact_extra_count",
        "mean_extra_rank_score",
        "max_extra_rank_score",
        "mean_extra_novelty",
        "extra_area_ratio",
        "selection_score",
    ]
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(r.get(key) or 0.0) for r in rows]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def apply_rows(
    raw_rows: list[dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
    *,
    output_path: Path,
    apply_top_n: int,
) -> dict[str, Any]:
    from postprocess import parse_cct_report  # type: ignore

    stats = {"rows": 0, "changed": 0, "boxes_added": 0}
    with output_path.open("w", encoding="utf-8") as fh:
        for row in raw_rows:
            stats["rows"] += 1
            sid = sample_id_from_row(row)
            diag = diagnostics.get(sid)
            if diag and diag.get("selected"):
                extras = list(diag.get("extras") or [])[:apply_top_n]
                if extras:
                    row = dict(row)
                    new_report = insert_extra_anomalies(str(row.get("raw_output") or ""), extras)
                    row["raw_output"] = new_report
                    row["parsed"] = parse_cct_report(new_report)
                    stage_outputs = dict(row.get("stage_outputs") or {})
                    stage_outputs["qwen_pipe_v88_loc_need_estimator"] = {
                        "applied": True,
                        "selection_score": diag.get("selection_score"),
                        "selection_reasons": diag.get("selection_reasons"),
                        "boxes_added": len(extras),
                        "selected": extras,
                        "policy": "GT-blind v88 selector: apply v87mix extras only when current report boxes fail to cover strong OCR/evidence candidates.",
                    }
                    row["stage_outputs"] = stage_outputs
                    stats["changed"] += 1
                    stats["boxes_added"] += len(extras)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL))
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--ocr-layout-cache-dir", default=str(DEFAULT_OCR_LAYOUT_CACHE))
    parser.add_argument("--ocr-layout-model", default="qwen-vl-ocr")
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel", "auto"], default="normalized-1000")
    parser.add_argument("--max-candidates", type=int, default=900)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--apply-top-n", type=int, default=2)
    parser.add_argument("--apply-max-area-ratio", type=float, default=0.06)
    parser.add_argument("--apply-duplicate-iou", type=float, default=0.50)
    parser.add_argument("--mode", choices=["recall", "balanced", "precision"], default="balanced")
    parser.add_argument("--rank-mode", choices=["v87mix", "v89token", "v145ocranchor"], default="v87mix")
    parser.add_argument("--enable-token-candidates", action="store_true")
    parser.add_argument("--enable-linegrid-candidates", action="store_true")
    parser.add_argument("--enable-scriptgrid-candidates", action="store_true")
    parser.add_argument("--output-jsonl", required=True, help="Diagnostic feature/selection JSONL.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--applied-raw-jsonl", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    input_path = resolve_pipe_path(args.input_jsonl)
    eval_path = resolve_pipe_path(args.eval_json)
    output_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(input_path)
    eval_samples = load_eval_samples(eval_path)

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    with output_path.open("w", encoding="utf-8") as fh:
        for row in raw_rows:
            features = build_features(
                row,
                debug_root=debug_root,
                ocr_layout_cache_dir=Path(args.ocr_layout_cache_dir).expanduser().resolve(),
                ocr_layout_model=args.ocr_layout_model,
                coord_mode=args.coord_mode,
                max_candidates=args.max_candidates,
                top_k=args.top_k,
                apply_top_n=args.apply_top_n,
                apply_max_area_ratio=args.apply_max_area_ratio,
                apply_duplicate_iou=args.apply_duplicate_iou,
                rank_mode=args.rank_mode,
                enable_token_candidates=args.enable_token_candidates,
                enable_linegrid_candidates=args.enable_linegrid_candidates,
                enable_scriptgrid_candidates=args.enable_scriptgrid_candidates,
            )
            selected, score, reasons = need_score(features, args.mode)
            features["selected"] = bool(selected)
            features["selection_score"] = float(score)
            features["selection_reasons"] = reasons
            if selected:
                features["applied_extra_count"] = min(args.apply_top_n, len(features.get("extras") or []))
            else:
                features["applied_extra_count"] = 0
            rows.append(features)
            diagnostics[features["sample_id"]] = features
            fh.write(json.dumps(features, ensure_ascii=False) + "\n")

    summary = summarize(rows, eval_samples)
    summary["mode"] = args.mode
    summary["rank_mode"] = args.rank_mode
    summary["enable_token_candidates"] = bool(args.enable_token_candidates)
    summary["enable_linegrid_candidates"] = bool(args.enable_linegrid_candidates)
    summary["enable_scriptgrid_candidates"] = bool(args.enable_scriptgrid_candidates)
    summary["input_jsonl"] = str(input_path)
    summary["eval_json"] = str(eval_path)
    summary["apply_top_n"] = args.apply_top_n
    summary["apply_max_area_ratio"] = args.apply_max_area_ratio
    summary["apply_duplicate_iou"] = args.apply_duplicate_iou
    if args.applied_raw_jsonl:
        applied_path = resolve_pipe_path(args.applied_raw_jsonl)
        applied_path.parent.mkdir(parents=True, exist_ok=True)
        summary["applied_raw_jsonl"] = str(applied_path)
        summary["apply_stats"] = apply_rows(raw_rows, diagnostics, output_path=applied_path, apply_top_n=args.apply_top_n)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
