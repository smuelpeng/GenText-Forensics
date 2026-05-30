#!/usr/bin/env python3
"""GT-free diverse append of OCR/token localization candidates.

This tests a different localization mechanism from single-candidate replacement:
for low-S_Loc candidate-diagnostic samples, select several high-quality,
spatially diverse OCR/token/linegrid candidates and append them as extra
anomalies.  GT artifacts are not read by this script; evaluation can be run
afterward to decide whether the mechanism helps.
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

from qwen_candidate_quality_selector import (  # noqa: E402
    box_iou,
    insert_extra_anomalies,
    read_jsonl,
    report_terms,
    resolve_pipe_path,
    score_candidate,
    setup_debug_import,
)
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402


CONCLUSION_FORGED_RE = re.compile(r"\[Conclusion\]\s*:\s*(?:\*\*)?\s*FORGED|\bConclusion\s*:\s*(?:\*\*)?\s*FORGED", re.IGNORECASE)


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)


def candidate_allowed(cand: dict[str, Any], info: dict[str, Any], args: argparse.Namespace) -> bool:
    family = str(cand.get("family") or "")
    if family not in set(args.allowed_families.split(",")):
        return False
    area_ratio = float(info.get("area_ratio") or 0.0)
    support = (
        len(info.get("useful_query_hits") or [])
        + len(info.get("useful_number_hits") or [])
        + int(info.get("report_overlap") or 0)
        + len(info.get("visual_hits") or [])
    )
    if support < args.min_support:
        return False
    if area_ratio > args.max_area_ratio:
        return False
    if float(info.get("duplicate_iou") or 0.0) >= args.max_existing_iou:
        return False
    return True


def select_diverse(
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]],
    *,
    max_add: int,
    min_pair_iou: float,
) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for score, cand, info in scored:
        box = [int(v) for v in info["box"]]
        if any(box_iou(box, [int(v) for v in other_info["box"]]) > min_pair_iou for _, _, other_info in selected):
            continue
        selected.append((score, cand, info))
        if len(selected) >= max_add:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--candidate-source", choices=["all", "top"], default="top")
    parser.add_argument("--allowed-families", default="ocr,token,linegrid,evidence,patch")
    parser.add_argument("--score-threshold", type=float, default=10.5)
    parser.add_argument("--max-area-ratio", type=float, default=0.006)
    parser.add_argument("--max-existing-iou", type=float, default=0.45)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-pair-iou", type=float, default=0.35)
    parser.add_argument("--max-add", type=int, default=3)
    parser.add_argument("--max-total-boxes", type=int, default=9)
    parser.add_argument("--min-existing-boxes", type=int, default=1)
    parser.add_argument("--max-existing-boxes", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    candidate_path = resolve_pipe_path(args.candidate_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_rows = {str(r.get("sample_id")): r for r in read_jsonl(candidate_path)}
    changed = 0
    rows_seen = 0
    rows_with_diag = 0
    rows_below = 0
    added_total = 0
    family_counts: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()
    selected_records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_seen += 1
            sid = sample_id(row)
            diag = candidate_rows.get(sid)
            report = str(row.get("raw_output") or "")
            if not diag or not CONCLUSION_FORGED_RE.search(report):
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            if args.limit_samples and rows_with_diag >= args.limit_samples:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            rows_with_diag += 1

            width = int(diag.get("width") or row.get("width") or 0)
            height = int(diag.get("height") or row.get("height") or 0)
            existing = report_boxes(report)
            if len(existing) < args.min_existing_boxes or len(existing) > args.max_existing_boxes:
                reject_counts["existing_count_gate"] += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            room = max(0, args.max_total_boxes - len(existing))
            if room <= 0:
                reject_counts["max_total_boxes"] += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            terms = report_terms(row)
            source_key = "candidates" if args.candidate_source == "all" else "top_candidates"
            scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for cand in list(diag.get(source_key) or []):
                score, info = score_candidate(
                    cand,
                    row,
                    width=width,
                    height=height,
                    existing_boxes=existing,
                    terms=terms,
                    max_area_ratio=args.max_area_ratio,
                )
                if score <= -1e8:
                    reject_counts[str(info.get("reject") or "invalid")] += 1
                    continue
                if score < args.score_threshold:
                    reject_counts["below_score"] += 1
                    continue
                if not candidate_allowed(cand, info, args):
                    reject_counts["quality_gate"] += 1
                    continue
                scored.append((score, cand, info))
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = select_diverse(scored, max_add=min(args.max_add, room), min_pair_iou=args.min_pair_iou)
            if not selected:
                rows_below += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            extras: list[dict[str, Any]] = []
            for score, cand, info in selected:
                item = dict(cand)
                item["box"] = [int(v) for v in info["box"]]
                item["diverse_append_score"] = score
                item["diverse_append_quality"] = info
                extras.append(item)
                family_counts[str(cand.get("family") or "")] += 1
            row = dict(row)
            new_report = insert_extra_anomalies(report, extras)
            row["raw_output"] = new_report
            row["parsed"] = parse_cct_report(new_report)
            stage_outputs = dict(row.get("stage_outputs") or {})
            stage_outputs["qwen_diverse_candidate_append"] = {
                "applied": True,
                "candidate_jsonl": str(candidate_path),
                "candidate_source": args.candidate_source,
                "score_threshold": args.score_threshold,
                "max_add": args.max_add,
                "added": extras,
                "gt_free": True,
                "policy": "Append multiple spatially diverse, report-supported OCR/token/linegrid candidates to improve localization recall.",
            }
            row["stage_outputs"] = stage_outputs
            changed += 1
            added_total += len(extras)
            selected_records.append(
                {
                    "sample_id": sid,
                    "existing_box_count": len(existing),
                    "added_count": len(extras),
                    "selected": [
                        {
                            "score": score,
                            "family": str(cand.get("family") or ""),
                            "source": str(cand.get("source") or ""),
                            "box": info.get("box"),
                            "local_text": info.get("local_text"),
                            "support": len(info.get("useful_query_hits") or [])
                            + len(info.get("useful_number_hits") or [])
                            + int(info.get("report_overlap") or 0)
                            + len(info.get("visual_hits") or []),
                            "area_ratio": info.get("area_ratio"),
                            "duplicate_iou": info.get("duplicate_iou"),
                        }
                        for score, cand, info in selected
                    ],
                    "top_scores": [
                        {
                            "score": score,
                            "family": str(cand.get("family") or ""),
                            "box": info.get("box"),
                            "local_text": info.get("local_text"),
                        }
                        for score, cand, info in scored[:8]
                    ],
                }
            )
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(input_path),
        "candidate_jsonl": str(candidate_path),
        "output_jsonl": str(output_path),
        "candidate_source": args.candidate_source,
        "score_threshold": args.score_threshold,
        "max_area_ratio": args.max_area_ratio,
        "min_support": args.min_support,
        "max_add": args.max_add,
        "max_total_boxes": args.max_total_boxes,
        "rows_seen": rows_seen,
        "rows_with_diag": rows_with_diag,
        "changed": changed,
        "added_total": added_total,
        "rows_below": rows_below,
        "selected_family_counts": dict(family_counts),
        "reject_counts": dict(reject_counts),
        "selected_records": selected_records,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("changed", "added_total", "selected_family_counts", "reject_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
