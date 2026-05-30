#!/usr/bin/env python3
"""Append evidence-level localization boxes to forged Qwen-pipe reports.

DocShield emphasizes grounding validated evidence, while FakeShield converts a
tamper description into a mask. This lightweight approximation keeps the final
verdict unchanged, but appends extra compact [GROUNDING] boxes from existing
high-confidence local evidence candidates when the report currently has too
few localized regions.

GT fields are never read by this script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v75b_hard_discard_rescue_300.jsonl"
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[([^\[\]]+)\]", re.IGNORECASE)

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_issue_refine import (  # noqa: E402
    BENIGN_RESCUE_EXCLUDE_TERMS,
    HARD_VISUAL_TERMS,
    evidence_candidates,
    has_any,
    parse_box,
    parsed_conclusion,
    project_box,
    resolve_pipe_path,
    setup_debug_import,
)


LOCAL_CATEGORIES = {
    "rendering_artifact",
    "color_mismatch",
    "layout_inconsistency",
    "font_mismatch",
    "copy_paste_boundary",
}


def report_boxes(report: str) -> list[list[int]]:
    boxes: list[list[int]] = []
    for match in GROUNDING_RE.finditer(report or ""):
        nums = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        if len(nums) < 4:
            continue
        try:
            box = [int(round(float(v))) for v in nums[:4]]
        except ValueError:
            continue
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / max(1, area(a) + area(b) - inter)


def candidate_text(cand: dict[str, Any]) -> str:
    return " ".join(str(cand.get(key) or "") for key in ("category", "evidence", "notes", "reason"))


def candidate_score(cand: dict[str, Any]) -> float:
    try:
        conf = float(cand.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    text = candidate_text(cand)
    hard_bonus = 0.2 if has_any(text, HARD_VISUAL_TERMS) else 0.0
    return conf + hard_bonus


def extra_candidates(
    row: dict[str, Any],
    *,
    min_confidence: float,
    coord_mode: str,
    max_area_ratio: float,
    duplicate_iou: float,
    existing: list[list[int]],
) -> list[tuple[dict[str, Any], list[int], float]]:
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    page_area = max(1, width * height)
    out: list[tuple[dict[str, Any], list[int], float]] = []

    for cand in evidence_candidates(row):
        category = str(cand.get("category") or "").lower()
        if category not in LOCAL_CATEGORIES:
            continue
        try:
            conf = float(cand.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < min_confidence:
            continue
        text = candidate_text(cand)
        if has_any(text, BENIGN_RESCUE_EXCLUDE_TERMS):
            continue
        if not has_any(text, HARD_VISUAL_TERMS) and category not in {"copy_paste_boundary", "font_mismatch"}:
            continue
        raw_box = parse_box(cand.get("bbox"))
        if not raw_box:
            continue
        box = project_box(raw_box, width, height, coord_mode)
        if not box:
            continue
        if area(box) / page_area > max_area_ratio:
            continue
        if any(iou(box, prev) >= duplicate_iou for prev in existing):
            continue
        out.append((cand, box, candidate_score(cand)))

    out.sort(key=lambda item: item[2], reverse=True)
    return out


def insert_extra_anomalies(report: str, extras: list[tuple[dict[str, Any], list[int], float]]) -> str:
    if not extras:
        return report
    block: list[str] = []
    for idx, (cand, box, score) in enumerate(extras, start=1):
        category = str(cand.get("category") or "local_evidence")
        evidence = re.sub(r"\s+", " ", str(cand.get("evidence") or "Localized evidence candidate.")).strip()
        block.extend(
            [
                f"### ANOMALY_EXTRA_{idx:03d}: Evidence-level Local Grounding ({category})",
                f"[GROUNDING]:{box}",
                f"[REASON]: {evidence} This extra grounding is retained from high-confidence local evidence to cover dispersed tamper regions.",
                "",
            ]
        )
    extra_text = "\n".join(block)
    marker = re.search(r"\n\s*-{3,}\s*\n\s*##\s*SUMMARY|\n\s*##\s*SUMMARY", report, re.IGNORECASE)
    if marker:
        return report[: marker.start()] + "\n\n" + extra_text + report[marker.start():]
    return report.rstrip() + "\n\n" + extra_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel"], default="normalized-1000")
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--min-existing-boxes", type=int, default=0)
    parser.add_argument("--max-total-boxes", type=int, default=6)
    parser.add_argument("--max-extra-boxes", type=int, default=4)
    parser.add_argument("--max-area-ratio", type=float, default=0.18)
    parser.add_argument("--duplicate-iou", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "rows": 0,
        "rows_changed": 0,
        "boxes_added": 0,
        "skipped_non_forged": 0,
        "skipped_enough_boxes": 0,
        "candidate_counts": {},
    }

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            out = dict(row)
            report = str(row.get("raw_output") or "")
            existing = report_boxes(report)
            changed = False
            extras: list[tuple[dict[str, Any], list[int], float]] = []

            if parsed_conclusion(row) != "FORGED":
                stats["skipped_non_forged"] += 1
            elif len(existing) >= args.max_total_boxes or len(existing) < args.min_existing_boxes:
                stats["skipped_enough_boxes"] += 1
            else:
                candidates = extra_candidates(
                    row,
                    min_confidence=args.min_confidence,
                    coord_mode=args.coord_mode,
                    max_area_ratio=args.max_area_ratio,
                    duplicate_iou=args.duplicate_iou,
                    existing=existing,
                )
                budget = max(0, min(args.max_extra_boxes, args.max_total_boxes - len(existing)))
                extras = candidates[:budget]
                if extras:
                    new_report = insert_extra_anomalies(report, extras)
                    out["raw_output"] = new_report
                    out["parsed"] = parse_cct_report(new_report)
                    changed = True

            stage_outputs = dict(out.get("stage_outputs") or {})
            stage_outputs["qwen_pipe_evidence_multibox_refine"] = {
                "applied": changed,
                "existing_boxes": len(existing),
                "boxes_added": len(extras),
                "policy": "Append high-confidence local evidence boxes only; keep verdict and existing anomalies unchanged. No GT fields are read.",
            }
            out["stage_outputs"] = stage_outputs
            if changed:
                stats["rows_changed"] += 1
                stats["boxes_added"] += len(extras)
                stats["candidate_counts"][str(len(extras))] = stats["candidate_counts"].get(str(len(extras)), 0) + 1
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
