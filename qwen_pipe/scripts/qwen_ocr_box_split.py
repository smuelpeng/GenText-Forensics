#!/usr/bin/env python3
"""Split Stage-4 OCR/span-normalized anomaly boxes into smaller OCR boxes.

This is a GT-blind localization-only postprocess for Qwen-pipe. It rebuilds
FORGED report anomaly blocks from Stage-4 normalized anomalies, using OCR span
boxes when an anomaly references multiple spans. Verdicts, risk scores, and
reasons are preserved from Stage-4/final report evidence.
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

SUMMARY_RE = re.compile(
    r"##\s*SUMMARY\s*\n(.*?)(?=\n\s*---|\n\s*\*\*END OF REPORT\*\*|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def setup_debug_import(debug_root: Path) -> None:
    docshield_dir = debug_root / "baselines" / "DocShield"
    if not docshield_dir.exists():
        raise SystemExit(f"DocShield helper directory not found: {docshield_dir}")
    sys.path.insert(0, str(docshield_dir))


def resolve_pipe_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PIPE_ROOT / p


def normalized_payload(row: dict[str, Any]) -> dict[str, Any]:
    stage_outputs = row.get("stage_outputs") or {}
    for stage_name in ("validation_grounding", "grounding"):
        stage = stage_outputs.get(stage_name) or {}
        normalized = stage.get("normalized") or {}
        if normalized.get("validated_anomalies"):
            return normalized
    return {}


def get_ocr_layout(row: dict[str, Any]) -> dict[str, Any]:
    stage_outputs = row.get("stage_outputs") or {}
    return ((stage_outputs.get("ocr_layout") or {}).get("parsed") or {})


def pad_box(box: list[int], width: int, height: int, pad_x: float, pad_y: float) -> list[int] | None:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    dx = int(round((x2 - x1) * pad_x))
    dy = int(round((y2 - y1) * pad_y))
    out = [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(width, x2 + dx),
        min(height, y2 + dy),
    ]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def anomaly_boxes(
    anomaly: dict[str, Any],
    span_boxes: dict[str, list[int]],
    *,
    width: int,
    height: int,
    max_boxes: int,
    pad_x: float,
    pad_y: float,
) -> list[list[int]]:
    from run_staged_docshield_api import normalize_bbox  # type: ignore

    boxes: list[list[int]] = []
    for span_id in anomaly.get("span_ids") or []:
        box = span_boxes.get(str(span_id))
        if box:
            padded = pad_box(box, width, height, pad_x, pad_y)
            if padded:
                boxes.append(padded)
    if boxes:
        return boxes[:max_boxes]

    fallback = normalize_bbox(anomaly.get("bbox"), width, height)
    if fallback:
        padded = pad_box(fallback, width, height, pad_x, pad_y)
        return [padded or fallback]
    return []


def original_summary(report: str) -> str:
    match = SUMMARY_RE.search(report or "")
    if match:
        return " ".join(match.group(1).split())
    return ""


def rebuild_report(
    row: dict[str, Any],
    normalized: dict[str, Any],
    split_boxes: list[tuple[dict[str, Any], list[list[int]]]],
    *,
    mode: str,
) -> str:
    parsed = row.get("parsed") or {}
    verdict = str(parsed.get("conclusion") or normalized.get("verdict") or "FORGED").upper()
    risk = int(normalized.get("risk_score") or parsed.get("risk_score") or 85)
    summary = original_summary(str(row.get("raw_output") or ""))
    if not summary:
        summary = f"The examination of the document has identified {len(split_boxes)} anomalies, resulting in a fraud risk score of {risk}."

    lines = [
        "# FORGERY ANALYSIS  REPORT",
        "",
        "**Report ID:** FAR-xxxx-xx-xx",
        "**Date of Examination:** xxxx-xx-xx",
        "**Case Type:** Document Authentication & Fraud Analysis",
        "",
        "**Overall Assessment:**",
        f"    **[Conclusion]:** {verdict}",
        f"    **[RISK_SCORE]:** {risk}",
        "",
        "---",
        "",
        "## DETAILED ANOMALY ANALYSIS",
        "",
    ]

    anomaly_index = 1
    if verdict == "FORGED" and split_boxes:
        for anomaly, boxes in split_boxes:
            category = str(anomaly.get("category") or "text tampering")
            reason = str(anomaly.get("reason") or anomaly.get("visual_support") or anomaly.get("logical_support") or "")
            if mode == "duplicate":
                for box in boxes:
                    lines.extend(
                        [
                            f"### ANOMALY_{anomaly_index:03d}: {category} (OCR-grounded region)",
                            f"[GROUNDING]:{box}",
                            f"[REASON]: {reason}",
                            "",
                        ]
                    )
                    anomaly_index += 1
            else:
                lines.append(f"### ANOMALY_{anomaly_index:03d}: {category} (OCR-grounded region)")
                for box in boxes:
                    lines.append(f"[GROUNDING]:{box}")
                lines.extend([f"[REASON]: {reason}", ""])
                anomaly_index += 1
    else:
        lines.extend(
            [
                "No anomalies detected. The document has been thoroughly examined and no signs of tampering, alteration, or forgery were found.",
                "",
            ]
        )

    lines.extend(["---", "", "## SUMMARY", summary, "", "---", "**END OF REPORT**"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--mode", choices=["multi_grounding", "duplicate"], default="duplicate")
    parser.add_argument("--max-boxes-per-anomaly", type=int, default=6)
    parser.add_argument("--max-boxes-per-row", type=int, default=12)
    parser.add_argument("--pad-x", type=float, default=0.05)
    parser.add_argument("--pad-y", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)

    from postprocess import parse_cct_report  # type: ignore
    from run_staged_docshield_api import collect_spans  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows": 0,
        "rows_rebuilt": 0,
        "source_anomalies": 0,
        "output_boxes": 0,
        "mode": args.mode,
        "max_boxes_per_anomaly": args.max_boxes_per_anomaly,
        "max_boxes_per_row": args.max_boxes_per_row,
        "pad_x": args.pad_x,
        "pad_y": args.pad_y,
    }

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            stats["rows"] += 1
            normalized = normalized_payload(row)
            anomalies = [a for a in normalized.get("validated_anomalies") or [] if isinstance(a, dict)]
            current_verdict = str((row.get("parsed") or {}).get("conclusion") or "").upper()
            out = dict(row)
            if current_verdict == "FORGED" and anomalies:
                width = int(row.get("width") or 0)
                height = int(row.get("height") or 0)
                span_boxes = collect_spans(get_ocr_layout(row), width, height)
                split: list[tuple[dict[str, Any], list[list[int]]]] = []
                total_boxes = 0
                for anomaly in anomalies:
                    boxes = anomaly_boxes(
                        anomaly,
                        span_boxes,
                        width=width,
                        height=height,
                        max_boxes=args.max_boxes_per_anomaly,
                        pad_x=args.pad_x,
                        pad_y=args.pad_y,
                    )
                    if not boxes:
                        continue
                    remaining = args.max_boxes_per_row - total_boxes
                    if remaining <= 0:
                        break
                    boxes = boxes[:remaining]
                    split.append((anomaly, boxes))
                    total_boxes += len(boxes)

                if split:
                    report = rebuild_report(out, normalized, split, mode=args.mode)
                    out["raw_output"] = report
                    out["parsed"] = parse_cct_report(report)
                    stage_outputs = dict(out.get("stage_outputs") or {})
                    stage_outputs["qwen_pipe_ocr_box_split"] = {
                        "applied": True,
                        "mode": args.mode,
                        "source_anomalies": len(anomalies),
                        "output_boxes": total_boxes,
                        "policy": "Split Stage-4 OCR/span boxes for localization only; keep verdict and anomaly reasons from normalized evidence.",
                    }
                    out["stage_outputs"] = stage_outputs
                    stats["rows_rebuilt"] += 1
                    stats["source_anomalies"] += len(anomalies)
                    stats["output_boxes"] += total_boxes

            dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
