#!/usr/bin/env python3
"""GT-free pruning of low-support localization boxes.

Recent diagnostics showed that some low-S_Loc samples improve when a bad old
box is removed or replaced.  This script tests that mechanism without using GT:
it scores existing [GROUNDING] boxes by report-source reliability, geometry,
page-edge noise, duplication, and weak visual support, then removes at most a
small number of high-risk boxes from targeted samples.

GT/eval files are not read by this script.  A candidate JSONL may be supplied
only to define the target sample set and page dimensions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_candidate_quality_selector import box_iou, page_area_ratio, resolve_pipe_path  # noqa: E402
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402


GROUNDING_RE = re.compile(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", re.IGNORECASE)
BLOCK_RE = re.compile(r"(?ms)^###\s+ANOMALY[^\n]*\n.*?(?=^###\s+ANOMALY|\n\s*-{3,}\s*\n|\n\s*##\s*SUMMARY|\Z)")
VISUAL_TERMS_RE = re.compile(
    r"\b(?:smudge|blur|artifact|jagged|block|obscur|redaction|black|dark|gray|grey|font|render|overlap|misalign|"
    r"สีเทา|ทับ|บดบัง|ดำ|模糊|锯齿|遮挡|黑|灰|错位|重叠|طمس|أسود|رمادي)\b",
    re.IGNORECASE,
)
LOGICAL_ONLY_RE = re.compile(
    r"\b(?:logic|logical|identity_conflict|math_error|sequence_error|date|amount|total|sum|contradiction|"
    r"逻辑|矛盾|金额|日期|合计|ลำดับ|วันที่|จำนวน|تناقض|تاريخ)\b",
    re.IGNORECASE,
)
EXTRA_RE = re.compile(r"\b(?:ANOMALY_V\d+_EXTRA|ANOMALY_EXTRA|Exhaustive Recall Candidate|Additional localization candidate|GT-blind|grid search|OCR line)\b", re.IGNORECASE)
WEAK_STAGE_RE = re.compile(r"\b(?:Evidence-level Local Grounding|CROP_VERIFY|quality-selected|crop verifier)\b", re.IGNORECASE)
FOOTER_RE = re.compile(r"\b(?:footer|page|copyright|confidential|www\.|https?://|qq|交流群|页|หน้า|الصفحة)\b", re.IGNORECASE)
MAIN_TOP_RE = re.compile(r"\b(?:main character|glyph|title|header|logo|banner|页眉|标题|字符|字形)\b", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)


def parse_box_text(text: str) -> list[int] | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    if len(nums) < 4:
        return None
    try:
        box = [int(round(float(v))) for v in nums[:4]]
    except ValueError:
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def iter_grounding_entries(report: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    blocks = list(BLOCK_RE.finditer(report or ""))
    grounding_idx = 0
    for match in GROUNDING_RE.finditer(report or ""):
        box = parse_box_text(match.group(2))
        if not box:
            continue
        block_match = None
        for block in blocks:
            if block.start() <= match.start() < block.end():
                block_match = block
                break
        block_text = block_match.group(0) if block_match else ""
        entries.append(
            {
                "index": grounding_idx,
                "box": box,
                "grounding_span": match.span(),
                "block_span": block_match.span() if block_match else None,
                "block_text": block_text,
                "grounding_text": match.group(0),
            }
        )
        grounding_idx += 1
    return entries


def box_shape_features(box: list[int], width: int, height: int) -> dict[str, float]:
    bw = max(1, box[2] - box[0])
    bh = max(1, box[3] - box[1])
    return {
        "area_ratio": page_area_ratio(box, width, height),
        "width_ratio": bw / max(1.0, float(width)),
        "height_ratio": bh / max(1.0, float(height)),
        "aspect_h_over_w": bh / max(1.0, float(bw)),
        "y_center": (box[1] + box[3]) / max(1.0, 2.0 * height),
        "x_center": (box[0] + box[2]) / max(1.0, 2.0 * width),
    }


def duplicate_score(box: list[int], boxes: list[list[int]]) -> float:
    vals = [box_iou(box, other) for other in boxes if other is not box]
    return max(vals, default=0.0)


def badness(entry: dict[str, Any], boxes: list[list[int]], width: int, height: int, strategy: str) -> tuple[float, dict[str, Any]]:
    box = entry["box"]
    text = str(entry.get("block_text") or "")
    shape = box_shape_features(box, width, height)
    score = 0.0
    reasons: list[str] = []

    is_extra = bool(EXTRA_RE.search(text))
    is_weak_stage = bool(WEAK_STAGE_RE.search(text))
    has_visual = bool(VISUAL_TERMS_RE.search(text))
    logical_only = bool(LOGICAL_ONLY_RE.search(text)) and not has_visual
    near_edge = shape["y_center"] < 0.055 or shape["y_center"] > 0.935
    near_bottom = shape["y_center"] > 0.88
    dup = duplicate_score(box, boxes)

    if is_extra:
        score += 2.4
        reasons.append("extra_candidate")
    if is_weak_stage:
        score += 1.0
        reasons.append("weak_stage")
    if near_edge:
        score += 1.2
        reasons.append("page_edge")
    if near_bottom and FOOTER_RE.search(text):
        score += 2.2
        reasons.append("footer_text")
    if near_bottom and MAIN_TOP_RE.search(text):
        score += 2.2
        reasons.append("top_label_at_bottom")
    vertical_strip = shape["aspect_h_over_w"] >= 3.2 and shape["height_ratio"] >= 0.045
    compact_vertical_strip = vertical_strip and shape["height_ratio"] <= 0.22
    if vertical_strip:
        score += 2.0
        reasons.append("vertical_strip")
    if shape["width_ratio"] >= 0.32 and shape["height_ratio"] <= 0.020:
        score += 1.3
        reasons.append("thin_wide_row")
    if shape["area_ratio"] < 0.00035:
        score += 0.7
        reasons.append("tiny_box")
    if dup >= 0.72:
        score += 2.2
        reasons.append("near_duplicate")
    elif dup >= 0.45:
        score += 1.1
        reasons.append("partial_duplicate")
    if logical_only and shape["area_ratio"] > 0.006:
        score += 1.0
        reasons.append("broad_logical_box")
    if not has_visual and not logical_only and is_extra:
        score += 0.8
        reasons.append("unsupported_extra")

    shape_or_noise_supported = (
        near_edge
        or FOOTER_RE.search(text)
        or vertical_strip
        or dup >= 0.45
        or shape["area_ratio"] < 0.00035
    )
    if strategy == "extra_only" and not (is_extra or is_weak_stage):
        score -= 4.0
        reasons.append("strategy_extra_only_penalty")
    elif strategy == "shape_extra" and (not (is_extra or is_weak_stage) or not shape_or_noise_supported):
        score -= 4.0
        reasons.append("strategy_shape_extra_penalty")
    elif strategy == "compact_shape_extra":
        compact_shape_or_noise_supported = (
            near_edge
            or FOOTER_RE.search(text)
            or compact_vertical_strip
            or dup >= 0.45
            or shape["area_ratio"] < 0.00035
        )
        if not (is_extra or is_weak_stage) or not compact_shape_or_noise_supported:
            score -= 4.0
            reasons.append("strategy_compact_shape_extra_penalty")
    elif strategy == "edge_noise" and not (near_edge or FOOTER_RE.search(text) or MAIN_TOP_RE.search(text)):
        score -= 3.0
        reasons.append("strategy_edge_penalty")
    elif strategy == "duplicate" and dup < 0.45:
        score -= 4.0
        reasons.append("strategy_duplicate_penalty")

    return score, {
        **shape,
        "duplicate_iou": dup,
        "is_extra": is_extra,
        "is_weak_stage": is_weak_stage,
        "has_visual_terms": has_visual,
        "logical_only": logical_only,
        "near_edge": near_edge,
        "badness": score,
        "reasons": reasons,
        "block_preview": re.sub(r"\s+", " ", text).strip()[:220],
    }


def remove_entries(report: str, entries: list[dict[str, Any]], mode: str) -> str:
    if not entries:
        return report
    if mode == "drop_block":
        spans = [tuple(e["block_span"]) for e in entries if e.get("block_span")]
        if not spans:
            mode = "drop_grounding"
        else:
            merged: list[tuple[int, int]] = []
            for start, end in sorted(set(spans)):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            out = report
            for start, end in reversed(merged):
                out = out[:start].rstrip() + "\n\n" + out[end:].lstrip()
            return out
    out = report
    for entry in sorted(entries, key=lambda e: e["grounding_span"][0], reverse=True):
        start, end = entry["grounding_span"]
        if mode == "blank_grounding":
            out = out[:start] + "[GROUNDING]:[]" + out[end:]
        else:
            line_start = out.rfind("\n", 0, start) + 1
            line_end = out.find("\n", end)
            if line_end < 0:
                line_end = end
            out = out[:line_start] + out[line_end + 1 :]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--candidate-jsonl", default="", help="Optional target sample set and dimensions.")
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--strategy", choices=["balanced", "extra_only", "shape_extra", "compact_shape_extra", "edge_noise", "duplicate"], default="balanced")
    parser.add_argument("--mode", choices=["drop_grounding", "blank_grounding", "drop_block"], default="drop_grounding")
    parser.add_argument("--score-threshold", type=float, default=4.2)
    parser.add_argument("--max-prune-per-sample", type=int, default=1)
    parser.add_argument("--min-boxes-before", type=int, default=0)
    parser.add_argument("--min-boxes-after", type=int, default=2)
    parser.add_argument("--limit-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    target_dims: dict[str, tuple[int, int]] = {}
    if args.candidate_jsonl:
        for row in read_jsonl(resolve_pipe_path(args.candidate_jsonl)):
            sid = str(row.get("sample_id") or "")
            if sid:
                target_dims[sid] = (int(row.get("width") or 0), int(row.get("height") or 0))

    changed = 0
    rows_seen = 0
    rows_targeted = 0
    below_threshold = 0
    reason_counts: Counter[str] = Counter()
    selected_records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_seen += 1
            sid = sample_id(row)
            if target_dims and sid not in target_dims:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            if args.limit_samples and rows_targeted >= args.limit_samples:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            rows_targeted += 1
            width, height = target_dims.get(sid, (int(row.get("width") or 0), int(row.get("height") or 0)))
            if width <= 0 or height <= 0:
                width, height = int(row.get("width") or 0), int(row.get("height") or 0)
            report = str(row.get("raw_output") or "")
            entries = iter_grounding_entries(report)
            boxes = [entry["box"] for entry in entries]
            if args.min_boxes_before and len(entries) < args.min_boxes_before:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            if len(entries) <= args.min_boxes_after:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for entry in entries:
                score, info = badness(entry, boxes, width, height, args.strategy)
                scored.append((score, entry, info))
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = [
                (score, entry, info)
                for score, entry, info in scored
                if score >= args.score_threshold
            ][: max(0, min(args.max_prune_per_sample, len(entries) - args.min_boxes_after))]
            if not selected:
                below_threshold += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            row = dict(row)
            to_remove = [entry for _, entry, _ in selected]
            new_report = remove_entries(report, to_remove, args.mode)
            row["raw_output"] = new_report
            if row.get("raw_output_full") == report:
                row["raw_output_full"] = remove_entries(str(row.get("raw_output_full") or ""), to_remove, args.mode)
            stage_outputs = dict(row.get("stage_outputs") or {})
            stage_outputs["qwen_bad_box_prune"] = {
                "applied": True,
                "strategy": args.strategy,
                "mode": args.mode,
                "score_threshold": args.score_threshold,
                "max_prune_per_sample": args.max_prune_per_sample,
                "min_boxes_before": args.min_boxes_before,
                "min_boxes_after": args.min_boxes_after,
                "removed": [
                    {
                        "index": entry["index"],
                        "box": entry["box"],
                        "badness": score,
                        "features": info,
                    }
                    for score, entry, info in selected
                ],
                "gt_free": True,
            }
            row["stage_outputs"] = stage_outputs
            changed += 1
            for _, _, info in selected:
                for reason in info.get("reasons") or []:
                    reason_counts[reason] += 1
            selected_records.append(
                {
                    "sample_id": sid,
                    "removed": [
                        {
                            "index": entry["index"],
                            "box": entry["box"],
                            "badness": score,
                            "features": info,
                        }
                        for score, entry, info in selected
                    ],
                    "top_scores": [
                        {
                            "index": entry["index"],
                            "box": entry["box"],
                            "badness": score,
                            "reasons": info.get("reasons"),
                            "preview": info.get("block_preview"),
                        }
                        for score, entry, info in scored[:5]
                    ],
                }
            )
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "candidate_jsonl": str(resolve_pipe_path(args.candidate_jsonl)) if args.candidate_jsonl else "",
        "strategy": args.strategy,
        "mode": args.mode,
        "score_threshold": args.score_threshold,
        "max_prune_per_sample": args.max_prune_per_sample,
        "min_boxes_before": args.min_boxes_before,
        "min_boxes_after": args.min_boxes_after,
        "rows_seen": rows_seen,
        "rows_targeted": rows_targeted,
        "changed": changed,
        "below_threshold": below_threshold,
        "reason_counts": dict(reason_counts),
        "selected_records": selected_records,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("strategy", "mode", "changed", "rows_targeted", "reason_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
