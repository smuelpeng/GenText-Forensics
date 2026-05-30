#!/usr/bin/env python3
"""Reproject Qwen Stage-4 boxes with an explicit coordinate-space policy.

The first OCR box refinement relies on DocShield's automatic coordinate-space
detector. That detector can classify Qwen's 0-1000 boxes as pixel boxes when an
image is close to 1000 px wide. This GT-blind postprocess keeps verdicts,
reasons, and anomaly counts fixed, and only rewrites final [GROUNDING] boxes
from the existing Stage-4 normalized anomalies using a chosen projection policy.
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


def parse_box(value: Any) -> list[float] | None:
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(nums) < 4:
            return None
        return [float(v) for v in nums[:4]]
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[i]) for i in range(4)]
        except (TypeError, ValueError):
            return None
    return None


def clamp_box(box: list[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    out = [
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    ]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def normalized_1000_box(box: list[float], width: int, height: int) -> list[int] | None:
    return clamp_box(
        [
            box[0] * width / 1000.0,
            box[1] * height / 1000.0,
            box[2] * width / 1000.0,
            box[3] * height / 1000.0,
        ],
        width,
        height,
    )


def pixel_box(box: list[float], width: int, height: int) -> list[int] | None:
    return clamp_box(box, width, height)


def is_ambiguous_1000_space(boxes: list[list[float]], width: int, height: int) -> bool:
    if not boxes or width <= 0 or height <= 0:
        return False
    max_coord = max(max(abs(v) for v in box[:4]) for box in boxes)
    if max_coord > 1000:
        return False
    # Images near 1000 px on one axis are the failure case: a normalized Qwen
    # y-coordinate can look valid as a pixel coordinate, but lands too high.
    near_1000_width = 850 <= width <= 1300 and height >= 1200
    near_1000_height = 850 <= height <= 1300 and width >= 1200
    large_image = width > 1300 or height > 1300
    return near_1000_width or near_1000_height or large_image


def project_boxes(
    raw_boxes: list[list[float]],
    width: int,
    height: int,
    mode: str,
) -> tuple[list[list[int]], str]:
    if width <= 0 or height <= 0:
        return [], "invalid-size"
    policy = mode
    if mode == "ambiguous-1000":
        policy = "normalized-1000" if is_ambiguous_1000_space(raw_boxes, width, height) else "pixel"
    projected: list[list[int]] = []
    for box in raw_boxes:
        out = normalized_1000_box(box, width, height) if policy == "normalized-1000" else pixel_box(box, width, height)
        if out:
            projected.append(out)
    return projected, policy


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
    parser.add_argument("--mode", choices=["normalized-1000", "ambiguous-1000", "pixel"], default="ambiguous-1000")
    parser.add_argument("--require-count-match", action="store_true")
    parser.add_argument(
        "--final-forged-only",
        action="store_true",
        help="Only rewrite rows whose current final parsed conclusion is FORGED.",
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

    stats: dict[str, Any] = {
        "rows": 0,
        "rows_rewritten": 0,
        "boxes_replaced": 0,
        "skipped_count_mismatch": 0,
        "skipped_non_forged": 0,
        "projection_counts": {},
        "mode": args.mode,
        "require_count_match": args.require_count_match,
        "final_forged_only": args.final_forged_only,
    }

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            stats["rows"] += 1
            out = dict(row)
            report = str(row.get("raw_output") or "")
            report_box_count = len(GROUNDING_RE.findall(report))
            final_verdict = str((row.get("parsed") or {}).get("conclusion") or "").upper()

            rewritten = False
            replaced = 0
            policy = "none"
            if args.final_forged_only and final_verdict != "FORGED":
                stats["skipped_non_forged"] += 1
            elif report_box_count:
                raw_boxes = [parse_box(a.get("bbox")) for a in normalized_anomalies(row) if a.get("bbox") is not None]
                raw_boxes = [box for box in raw_boxes if box is not None]
                if raw_boxes:
                    if args.require_count_match and len(raw_boxes) != report_box_count:
                        stats["skipped_count_mismatch"] += 1
                    else:
                        width = int(row.get("width") or 0)
                        height = int(row.get("height") or 0)
                        boxes, policy = project_boxes(raw_boxes[:report_box_count], width, height, args.mode)
                        if boxes:
                            new_report, replaced = replace_groundings(report, boxes)
                            if replaced and new_report != report:
                                out["raw_output"] = new_report
                                out["parsed"] = parse_cct_report(new_report)
                                rewritten = True

            stage_outputs = dict(out.get("stage_outputs") or {})
            stage_outputs["qwen_pipe_box_space_refine"] = {
                "applied": rewritten,
                "mode": args.mode,
                "projection_policy": policy,
                "report_box_count": report_box_count,
                "boxes_replaced": replaced,
                "policy": "Reproject existing Stage-4 boxes only; keep verdicts, reasons, and anomaly counts unchanged.",
            }
            out["stage_outputs"] = stage_outputs
            if rewritten:
                stats["rows_rewritten"] += 1
                stats["boxes_replaced"] += replaced
            stats["projection_counts"][policy] = stats["projection_counts"].get(policy, 0) + 1
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
