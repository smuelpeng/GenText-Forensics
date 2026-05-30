#!/usr/bin/env python3
"""Thai OCR adjacent-row localization backfill.

This postprocess is GT-free at inference time. It targets a failure pattern seen
in Thai low-localization reports: all predicted boxes sit on one table row while
nearby OCR spans in the next row contain the missed forged region. The script
uses only the generated report, language metadata, OCR/layout cache, and image.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_OCR_LAYOUT_CACHE = DEFAULT_DEBUG_ROOT / "outputs/cache/ocr_layouts"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import (  # noqa: E402
    box_iou,
    generate_candidates,
    insert_extra_anomalies,
    read_jsonl,
    sample_id_from_row,
)
from qwen_text_crop_verify import area, resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


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


def language_code(row: dict[str, Any], eval_samples: dict[str, dict[str, Any]], sample_id: str) -> str:
    sample = eval_samples.get(sample_id) or {}
    return str(
        row.get("language_code")
        or ((row.get("metadata") or {}).get("language_code"))
        or ocr_layout_language(row)
        or sample.get("language_code")
        or ""
    )


def center(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def max_iou(box: list[int], boxes: list[list[int]]) -> float:
    return max((box_iou(box, old) for old in boxes), default=0.0)


def concentrated_single_row(boxes: list[list[int]], max_y_span: float) -> bool:
    if len(boxes) < 3:
        return False
    centers = [center(box)[1] for box in boxes]
    return max(centers) - min(centers) <= max_y_span


def thai_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    thai = sum(1 for ch in text if "\u0e00" <= ch <= "\u0e7f")
    return thai / max(1, len(text))


def pick_backfill_candidates(
    row: dict[str, Any],
    *,
    debug_root: Path,
    ocr_layout_cache: Path,
    max_candidates: int,
    max_added: int,
    max_y_span: float,
    min_y_gap: float,
    max_y_gap: float,
) -> list[dict[str, Any]]:
    existing = report_boxes(str(row.get("raw_output") or ""))
    if not concentrated_single_row(existing, max_y_span=max_y_span):
        return []
    image_path = resolve_image_path(row, debug_root)
    if not image_path:
        return []
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    median_y = statistics.median(center(box)[1] for box in existing)
    median_x = statistics.median(center(box)[0] for box in existing)
    candidates = generate_candidates(
        row,
        image,
        debug_root,
        ocr_layout_cache,
        "qwen-vl-ocr",
        "normalized-1000",
        max_candidates,
        enable_token_candidates=True,
        enable_linegrid_candidates=True,
        enable_scriptgrid_candidates=True,
        language_code="th",
    )
    ranked = []
    for cand in candidates:
        source = str(cand.source or "")
        label = str(cand.label or "")
        text = str(cand.text or "")
        box = [int(v) for v in cand.box]
        box_area = area(box)
        if not source.startswith("ocr_span:qwen_ocr:expanded"):
            continue
        if not label.startswith("span_q"):
            continue
        if not (1200 <= box_area <= 35000):
            continue
        if max_iou(box, existing) > 0.05:
            continue
        cx, cy = center(box)
        y_gap = cy - median_y
        if y_gap < min_y_gap or y_gap > max_y_gap:
            continue
        if thai_char_ratio(text) < 0.35:
            continue
        # Prefer the same visual column as the current clustered report boxes,
        # then the nearest lower row, then richer OCR text.
        score = -abs(cx - median_x) / max(1.0, width) - y_gap / max(1.0, height) + min(0.2, len(text) / 500.0)
        ranked.append((score, cand))
    ranked.sort(key=lambda item: item[0], reverse=True)
    picked = []
    occupied = list(existing)
    for _score, cand in ranked:
        box = [int(v) for v in cand.box]
        if max_iou(box, occupied) > 0.20:
            continue
        picked.append(
            {
                "label": f"thai_ocr_backfill_{len(picked) + 1}",
                "box": box,
                "source": cand.source,
                "family": cand.family,
                "score": float(cand.score),
                "text": cand.text,
                "meta": {
                    **(cand.meta or {}),
                    "policy": "thai_adjacent_row_ocr_backfill",
                },
            }
        )
        occupied.append(box)
        if len(picked) >= max_added:
            break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--ocr-layout-cache-dir", default=str(DEFAULT_OCR_LAYOUT_CACHE))
    parser.add_argument("--max-candidates", type=int, default=1500)
    parser.add_argument("--max-added", type=int, default=2)
    parser.add_argument("--max-y-span", type=float, default=90.0)
    parser.add_argument("--min-y-gap", type=float, default=45.0)
    parser.add_argument("--max-y-gap", type=float, default=230.0)
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    rows = read_jsonl(resolve_pipe_path(args.input_jsonl))
    eval_samples: dict[str, dict[str, Any]] = {}
    if args.eval_json:
        eval_data = json.loads(resolve_pipe_path(args.eval_json).read_text(encoding="utf-8"))
        eval_samples = {str(sample.get("sample_id") or ""): sample for sample in eval_data.get("samples") or []}
    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "output_jsonl": str(out_path),
        "changed_samples": 0,
        "boxes_added": 0,
        "selected": [],
        "max_added": args.max_added,
        "max_y_span": args.max_y_span,
        "min_y_gap": args.min_y_gap,
        "max_y_gap": args.max_y_gap,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            sid = sample_id_from_row(row)
            extras: list[dict[str, Any]] = []
            if conclusion_is_forged(row) and language_code(row, eval_samples, sid) == "th":
                extras = pick_backfill_candidates(
                    row,
                    debug_root=debug_root,
                    ocr_layout_cache=Path(args.ocr_layout_cache_dir).expanduser().resolve(),
                    max_candidates=args.max_candidates,
                    max_added=args.max_added,
                    max_y_span=args.max_y_span,
                    min_y_gap=args.min_y_gap,
                    max_y_gap=args.max_y_gap,
                )
            if extras:
                row = dict(row)
                report = insert_extra_anomalies(str(row.get("raw_output") or ""), extras)
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs["qwen_pipe_thai_ocr_backfill"] = {
                    "applied": True,
                    "boxes_added": len(extras),
                    "selected": extras,
                    "policy": "GT-free Thai adjacent-row OCR-expanded span backfill for clustered grounding rows.",
                }
                row["stage_outputs"] = stage_outputs
                summary["changed_samples"] += 1
                summary["boxes_added"] += len(extras)
                summary["selected"].append({"sample_id": sid, "selected": extras})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
