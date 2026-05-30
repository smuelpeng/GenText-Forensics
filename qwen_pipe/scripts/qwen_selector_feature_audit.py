#!/usr/bin/env python3
"""Audit why the GT-free candidate selector misses oracle-good boxes.

This is a diagnostic-only script.  It reads an oracle JSONL produced by local
evaluation and compares the selector-chosen candidate against the best oracle
candidate using the same GT-free feature extractor.  The oracle data is never
used for prompt construction or for producing deployable predictions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_candidate_quality_selector import (  # noqa: E402
    box_iou,
    candidate_local_text,
    read_jsonl,
    report_terms,
    resolve_pipe_path,
    score_candidate,
)
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Current prediction raw JSONL used by the selector.")
    parser.add_argument("--candidate-jsonl", required=True, help="Exhaustive candidate diagnostic JSONL.")
    parser.add_argument("--oracle-jsonl", required=True, help="Local oracle diagnostic JSONL; diagnostic only.")
    parser.add_argument("--selector-summary-json", required=True, help="Selector summary with selected_records.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-area-ratio", type=float, default=0.012)
    return parser.parse_args()


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)


def load_selected(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for rec in data.get("selected_records") or []:
        sid = str(rec.get("sample_id") or "")
        if sid:
            out[sid] = rec
    return out


def same_box(a: list[Any], b: list[Any]) -> bool:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 4 or len(b) < 4:
        return False
    return [int(round(float(v))) for v in a[:4]] == [int(round(float(v))) for v in b[:4]]


def find_full_candidate(diag: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    target_box = action.get("box") or []
    target_label = str(action.get("label") or "")
    target_family = str(action.get("family") or "")
    target_source = str(action.get("source") or "")
    candidates = list(diag.get("candidates") or []) + list(diag.get("top_candidates") or [])
    for cand in candidates:
        if target_label and str(cand.get("label") or "") == target_label and same_box(cand.get("box") or [], target_box):
            return cand
    for cand in candidates:
        if (
            same_box(cand.get("box") or [], target_box)
            and str(cand.get("family") or "") == target_family
            and str(cand.get("source") or "") == target_source
        ):
            return cand
    compact = dict(action.get("candidate") or {})
    compact.setdefault("label", target_label)
    compact.setdefault("family", target_family)
    compact.setdefault("source", target_source)
    compact.setdefault("box", target_box)
    compact.setdefault("text", compact.get("text_preview") or "")
    return compact


def feature_view(score: float, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": round(score, 6),
        "family": info.get("family"),
        "source": info.get("source"),
        "area_ratio": info.get("area_ratio"),
        "width_ratio": info.get("width_ratio"),
        "height_ratio": info.get("height_ratio"),
        "query_hits": len(info.get("useful_query_hits") or []),
        "number_hits": len(info.get("useful_number_hits") or []),
        "visual_hits": len(info.get("visual_hits") or []),
        "token_hits": info.get("token_hits"),
        "report_overlap": info.get("report_overlap"),
        "duplicate_iou": info.get("duplicate_iou"),
        "min_center_distance": info.get("min_center_distance"),
        "local_text": info.get("local_text"),
        "reject": info.get("reject"),
    }


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> None:
    args = parse_args()
    raw_rows = {sample_id(r): r for r in read_jsonl(resolve_pipe_path(args.input_jsonl))}
    diag_rows = {str(r.get("sample_id")): r for r in read_jsonl(resolve_pipe_path(args.candidate_jsonl))}
    oracle_rows = {str(r.get("sample_id")): r for r in read_jsonl(resolve_pipe_path(args.oracle_jsonl))}
    selected_rows = load_selected(resolve_pipe_path(args.selector_summary_json))

    cases: list[dict[str, Any]] = []
    diffs: dict[str, list[float]] = defaultdict(list)
    family_pairs: Counter[str] = Counter()
    missing = Counter()

    for sid, oracle in oracle_rows.items():
        row = raw_rows.get(sid)
        diag = diag_rows.get(sid)
        selected = selected_rows.get(sid)
        if not row or not diag:
            missing["raw_or_diag"] += 1
            continue
        actions = oracle.get("top_oracle_actions") or []
        if not actions:
            missing["oracle_action"] += 1
            continue
        oracle_action = actions[0]
        oracle_cand = find_full_candidate(diag, oracle_action)
        selected_cand = dict((selected or {}).get("candidate") or {})
        if not selected_cand:
            missing["selected"] += 1
            continue

        width = int(row.get("width") or diag.get("width") or 0)
        height = int(row.get("height") or diag.get("height") or 0)
        existing = report_boxes(str(row.get("raw_output") or ""))
        terms = report_terms(row)

        oracle_score, oracle_info = score_candidate(
            oracle_cand,
            row,
            width=width,
            height=height,
            existing_boxes=existing,
            terms=terms,
            max_area_ratio=args.max_area_ratio,
        )
        selected_score, selected_info = score_candidate(
            selected_cand,
            row,
            width=width,
            height=height,
            existing_boxes=existing,
            terms=terms,
            max_area_ratio=args.max_area_ratio,
        )
        if oracle_score <= -1e8:
            missing[f"oracle_reject:{oracle_info.get('reject')}"] += 1

        oracle_box = [int(v) for v in oracle_action.get("box")[:4]]
        selected_box = [int(v) for v in selected_cand.get("box")[:4]]
        selected_vs_oracle_iou = box_iou(selected_box, oracle_box)
        pair_key = f"{selected_info.get('family')}->{oracle_info.get('family')}"
        family_pairs[pair_key] += 1
        for key in (
            "score",
            "area_ratio",
            "width_ratio",
            "height_ratio",
            "query_hits",
            "number_hits",
            "visual_hits",
            "token_hits",
            "report_overlap",
            "duplicate_iou",
            "min_center_distance",
        ):
            ov = feature_view(oracle_score, oracle_info).get(key)
            sv = feature_view(selected_score, selected_info).get(key)
            if isinstance(ov, (int, float)) and isinstance(sv, (int, float)):
                diffs[f"selected_minus_oracle_{key}"].append(float(sv) - float(ov))

        cases.append(
            {
                "sample_id": sid,
                "language_code": oracle.get("language_code") or (row.get("meta") or {}).get("language_code"),
                "base_f1": oracle.get("base_f1"),
                "best_oracle_delta": oracle.get("best_oracle_delta"),
                "best_deployable_delta": oracle.get("best_deployable_delta"),
                "selected_vs_oracle_iou": selected_vs_oracle_iou,
                "oracle_rank_idx": oracle_action.get("rank_idx"),
                "oracle_box": oracle_box,
                "selected_box": selected_box,
                "oracle": feature_view(oracle_score, oracle_info),
                "selected": feature_view(selected_score, selected_info),
                "oracle_text": candidate_local_text(oracle_cand)[:220],
                "selected_text": candidate_local_text(selected_cand)[:220],
            }
        )

    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "candidate_jsonl": str(resolve_pipe_path(args.candidate_jsonl)),
        "oracle_jsonl": str(resolve_pipe_path(args.oracle_jsonl)),
        "selector_summary_json": str(resolve_pipe_path(args.selector_summary_json)),
        "case_count": len(cases),
        "missing": dict(missing),
        "family_pairs_selected_to_oracle": dict(family_pairs),
        "mean_feature_deltas": {k: mean(v) for k, v in sorted(diffs.items())},
        "cases": sorted(cases, key=lambda c: (c["selected_vs_oracle_iou"], -(c.get("best_oracle_delta") or 0.0))),
    }
    out_path = resolve_pipe_path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("case_count", "missing", "family_pairs_selected_to_oracle", "mean_feature_deltas")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
