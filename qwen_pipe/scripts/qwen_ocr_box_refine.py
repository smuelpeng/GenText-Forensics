#!/usr/bin/env python3
"""Refine final report grounding boxes with OCR/span-normalized boxes.

This is a GT-blind Qwen-pipe postprocess. It does not change verdicts,
anomaly reasons, risk scores, or anomaly counts. It only replaces final
``[GROUNDING]`` coordinates with the Stage-4 normalized boxes already derived
from OCR spans, auxiliary OCR spans, or evidence candidate boxes.
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
GROUNDING_RE = re.compile(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", re.IGNORECASE)


def setup_debug_import(debug_root: Path) -> None:
    docshield_dir = debug_root / "baselines" / "DocShield"
    if not docshield_dir.exists():
        raise SystemExit(f"DocShield helper directory not found: {docshield_dir}")
    sys.path.insert(0, str(docshield_dir))


def resolve_pipe_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PIPE_ROOT / p


def normalized_anomalies(row: dict[str, Any]) -> list[dict[str, Any]]:
    stage_outputs = row.get("stage_outputs") or {}
    for stage_name in ("validation_grounding", "grounding"):
        stage = stage_outputs.get(stage_name) or {}
        normalized = stage.get("normalized") or {}
        anomalies = normalized.get("validated_anomalies") or []
        if anomalies:
            return [a for a in anomalies if isinstance(a, dict)]
    return []


def project_boxes(raw_boxes: list[list[float]], width: int, height: int) -> list[list[int]]:
    from run_staged_docshield_api import detect_bbox_space, project_bbox, raw_bbox  # type: ignore

    parsed = [box for box in (raw_bbox(b) for b in raw_boxes) if box]
    if not parsed or width <= 0 or height <= 0:
        return []
    coord_space = detect_bbox_space(parsed, width, height, model_like=True)
    projected: list[list[int]] = []
    for box in parsed:
        projected_box = project_bbox(box, width, height, coord_space)
        if projected_box:
            projected.append(projected_box)
    return projected


def replace_groundings(report: str, boxes: list[list[int]]) -> tuple[str, int]:
    idx = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        if idx >= len(boxes):
            return match.group(0)
        box = boxes[idx]
        idx += 1
        return f"{match.group(1)}{box}"

    return GROUNDING_RE.sub(repl, report), idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument(
        "--require-count-match",
        action="store_true",
        help="Only refine rows where normalized anomaly count equals final report grounding count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)

    from postprocess import parse_cct_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows": 0,
        "rows_with_report_boxes": 0,
        "rows_with_normalized_boxes": 0,
        "rows_refined": 0,
        "boxes_replaced": 0,
        "skipped_count_mismatch": 0,
    }

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            stats["rows"] += 1
            report = str(row.get("raw_output") or "")
            report_box_count = len(GROUNDING_RE.findall(report))
            if report_box_count:
                stats["rows_with_report_boxes"] += 1

            anomalies = normalized_anomalies(row)
            raw_boxes = [a.get("bbox") for a in anomalies if a.get("bbox")]
            if raw_boxes:
                stats["rows_with_normalized_boxes"] += 1

            out = dict(row)
            refined = False
            replaced = 0
            if report_box_count and raw_boxes:
                if args.require_count_match and len(raw_boxes) != report_box_count:
                    stats["skipped_count_mismatch"] += 1
                else:
                    width = int(row.get("width") or 0)
                    height = int(row.get("height") or 0)
                    boxes = project_boxes(raw_boxes[:report_box_count], width, height)
                    if boxes:
                        new_report, replaced = replace_groundings(report, boxes)
                        if replaced and new_report != report:
                            out["raw_output"] = new_report
                            out["parsed"] = parse_cct_report(new_report)
                            refined = True

            stage_outputs = dict(out.get("stage_outputs") or {})
            stage_outputs["qwen_pipe_ocr_box_refine"] = {
                "applied": refined,
                "report_box_count": report_box_count,
                "normalized_box_count": len(raw_boxes),
                "boxes_replaced": replaced,
                "policy": "Replace final report grounding boxes with OCR/span-normalized Stage-4 boxes only; keep verdicts and reasons unchanged.",
            }
            out["stage_outputs"] = stage_outputs
            if refined:
                stats["rows_refined"] += 1
                stats["boxes_replaced"] += replaced
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
