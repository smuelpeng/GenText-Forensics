#!/usr/bin/env python3
"""GT-free pseudo-mask diversity fusion for localization candidates.

This script consumes a diagnostic candidate JSONL from qwen_exhaustive_recall
and applies a deterministic candidate selector.  The selector is intentionally
not trained on GT: it scores candidate boxes by source family, physical size,
novelty relative to existing report boxes, and spatial diversity.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import (  # noqa: E402
    box_iou,
    insert_extra_anomalies,
    page_area_ratio,
    read_jsonl,
    sample_id_from_row,
)
from qwen_text_crop_verify import area, clamp_box, resolve_pipe_path  # noqa: E402


def conclusion_is_forged(row: dict[str, Any]) -> bool:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper() == "FORGED"


def compact_text(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def has_alpha_or_script(value: str) -> bool:
    for ch in str(value or ""):
        cp = ord(ch)
        if ch.isalpha() or 0x0E00 <= cp <= 0x0E7F or 0x0600 <= cp <= 0x06FF or 0x4E00 <= cp <= 0x9FFF:
            return True
    return False


def containment_ratio(inner: list[int], outer: list[int]) -> float:
    inner_area = max(1, int(area(inner)))
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / inner_area


def spatial_bin(box: list[int], width: int, height: int) -> tuple[int, int]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return (min(3, max(0, int(cx / max(1, width) * 4))), min(5, max(0, int(cy / max(1, height) * 6))))


def ideal_area_score(ratio: float, lo: float, hi: float) -> float:
    if ratio <= 0:
        return -3.0
    if lo <= ratio <= hi:
        return 1.0
    if ratio < lo:
        return -min(3.0, math.log(max(lo, 1e-6) / max(ratio, 1e-9), 2.0) * 0.7)
    return -min(4.0, math.log(max(ratio, 1e-9) / max(hi, 1e-6), 2.0) * 1.0)


def source_score(cand: dict[str, Any]) -> float:
    family = str(cand.get("family") or "")
    source = str(cand.get("source") or "")
    text = str(cand.get("text") or "")
    score = {
        "linegrid": 4.0,
        "ocr": 3.7,
        "grid": 3.5,
        "patch": 3.1,
        "evidence": 3.0,
        "row": 2.5,
        "token": 1.4,
        "paragraph": 1.2,
    }.get(family, 0.8)
    if source.startswith("ocr_scriptgrid:physical_window"):
        score += 1.4
    elif source.startswith("ocr_linegrid:subline_window"):
        score += 0.9
    elif source.startswith("ocr_span:qwen_ocr:expanded") or ":expanded" in source:
        score += 0.7
    elif source.startswith("ocr_span:qwen_ocr"):
        score += 0.4
    elif source.startswith("grid_"):
        score += 0.7
    elif source.startswith("patch:"):
        score += 0.6
    elif source.startswith("stage_evidence"):
        score += 0.5
    if source.startswith("ocr_token_subspan:exact"):
        score -= 0.9
    compact = compact_text(text)
    if family not in {"grid", "patch"}:
        if len(compact) <= 2:
            score -= 2.2
        elif len(compact) <= 5 and not has_alpha_or_script(text):
            score -= 1.4
    if re.fullmatch(r"[\d\s:.,/\\-]+", compact or "") and family not in {"grid", "patch"}:
        score -= 0.8
    return score


def candidate_score(
    cand: dict[str, Any],
    *,
    width: int,
    height: int,
    existing: list[list[int]],
    selected: list[dict[str, Any]],
    family_counts: Counter[str],
    bin_counts: Counter[tuple[int, int]],
) -> float:
    box = cand["box"]
    family = str(cand.get("family") or "")
    ratio = page_area_ratio(box, width, height)
    bw = max(1, box[2] - box[0])
    bh = max(1, box[3] - box[1])
    score = source_score(cand)
    if family in {"grid", "patch", "evidence"}:
        score += ideal_area_score(ratio, 0.012, 0.085)
    elif family in {"linegrid", "row"}:
        score += ideal_area_score(ratio, 0.0016, 0.040)
    elif family == "ocr":
        score += ideal_area_score(ratio, 0.0006, 0.032)
    else:
        score += ideal_area_score(ratio, 0.0004, 0.020)
    if bh / max(1, height) > 0.18 and family in {"token", "ocr"}:
        score -= 1.4
    if bw / max(1, width) < 0.018 and bh / max(1, height) < 0.018 and family not in {"grid", "patch"}:
        score -= 1.2
    max_existing_iou = max((box_iou(box, old) for old in existing), default=0.0)
    max_existing_contain = max((containment_ratio(box, old) for old in existing), default=0.0)
    score += min(1.3, (1.0 - max_existing_iou) * 0.8)
    if max_existing_contain > 0.85:
        score -= 1.2
    elif max_existing_contain > 0.45:
        score -= 0.35
    max_sel_iou = max((box_iou(box, old["box"]) for old in selected), default=0.0)
    score -= 4.2 * max_sel_iou
    bkey = spatial_bin(box, width, height)
    if bin_counts[bkey]:
        score -= 1.1
    if family_counts[family]:
        score -= 0.45 * family_counts[family]
    return score


def parse_filter(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def choose_candidates(
    candidates: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    existing: list[list[int]],
    limit: int,
    max_area_ratio: float,
    duplicate_iou: float,
    min_score: float,
    allowed_families: set[str],
) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for cand in candidates:
        if cand.get("family") == "existing" or str(cand.get("source") or "").startswith("current_final_report"):
            continue
        if allowed_families and str(cand.get("family") or "") not in allowed_families:
            continue
        raw_box = cand.get("box")
        if not isinstance(raw_box, list) or len(raw_box) < 4:
            continue
        box = clamp_box([float(v) for v in raw_box[:4]], width, height)
        if not box:
            continue
        if page_area_ratio(box, width, height) > max_area_ratio:
            continue
        if any(box_iou(box, old) >= duplicate_iou for old in existing):
            continue
        out = dict(cand)
        out["box"] = box
        usable.append(out)

    selected: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    bin_counts: Counter[tuple[int, int]] = Counter()
    while len(selected) < limit:
        best: dict[str, Any] | None = None
        best_score = -1e9
        for cand in usable:
            if cand in selected:
                continue
            if any(box_iou(cand["box"], old["box"]) >= duplicate_iou for old in selected):
                continue
            score = candidate_score(
                cand,
                width=width,
                height=height,
                existing=existing,
                selected=selected,
                family_counts=family_counts,
                bin_counts=bin_counts,
            )
            if score > best_score:
                best = cand
                best_score = score
        if best is None or best_score < min_score:
            break
        best = dict(best)
        best["pseudo_score"] = best_score
        selected.append(best)
        family_counts[str(best.get("family") or "")] += 1
        bin_counts[spatial_bin(best["box"], width, height)] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-area-ratio", type=float, default=0.09)
    parser.add_argument("--duplicate-iou", type=float, default=0.52)
    parser.add_argument("--min-score", type=float, default=3.8)
    parser.add_argument("--allowed-families", default="")
    args = parser.parse_args()

    raw_rows = read_jsonl(resolve_pipe_path(args.input_jsonl))
    candidate_rows = {str(r.get("sample_id") or ""): r for r in read_jsonl(resolve_pipe_path(args.candidate_jsonl))}
    out_rows: list[dict[str, Any]] = []
    changed = 0
    selected_total = 0
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    per_sample: list[dict[str, Any]] = []

    for row in raw_rows:
        sid = sample_id_from_row(row)
        diag = candidate_rows.get(sid)
        if not diag or not conclusion_is_forged(row):
            out_rows.append(row)
            continue
        width = int(diag.get("width") or row.get("width") or row.get("image_width") or 1)
        height = int(diag.get("height") or row.get("height") or row.get("image_height") or 1)
        existing = report_boxes(str(row.get("raw_output") or ""))
        selected = choose_candidates(
            list(diag.get("candidates") or []),
            width=width,
            height=height,
            existing=existing,
            limit=args.limit,
            max_area_ratio=args.max_area_ratio,
            duplicate_iou=args.duplicate_iou,
            min_score=args.min_score,
            allowed_families=parse_filter(args.allowed_families),
        )
        if not selected:
            out_rows.append(row)
            per_sample.append({"sample_id": sid, "selected_count": 0, "selected": []})
            continue
        new_row = dict(row)
        new_row["raw_output"] = insert_extra_anomalies(str(row.get("raw_output") or ""), selected)
        new_row["qwen_pseudomask_diverse_fusion"] = {
            "stage": "qwen_pseudomask_diverse_fusion",
            "selected_count": len(selected),
            "selected": selected,
        }
        out_rows.append(new_row)
        changed += 1
        selected_total += len(selected)
        for cand in selected:
            family_counts[str(cand.get("family") or "")] += 1
            source_counts[str(cand.get("source") or "")] += 1
        per_sample.append({"sample_id": sid, "selected_count": len(selected), "selected": selected})

    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "candidate_jsonl": str(resolve_pipe_path(args.candidate_jsonl)),
        "output_jsonl": str(out_path),
        "rows_total": len(raw_rows),
        "rows_changed": changed,
        "selected_total": selected_total,
        "family_counts": dict(family_counts),
        "source_counts": dict(source_counts),
        "limit": args.limit,
        "max_area_ratio": args.max_area_ratio,
        "duplicate_iou": args.duplicate_iou,
        "min_score": args.min_score,
        "allowed_families": args.allowed_families,
        "samples": per_sample,
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
