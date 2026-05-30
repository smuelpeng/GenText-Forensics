#!/usr/bin/env python3
"""GT-only oracle diagnosis for low-S_Loc candidate actions.

This is a local evaluation tool, not an inference stage.  It consumes an
existing candidate-diagnostic JSONL, scores candidate append/replace actions
against the GT mask, and writes a compact per-sample oracle report.  GT is used
only after candidates already exist; do not feed this output into model prompts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_GT = DEFAULT_DEBUG_ROOT / "data/val_300.jsonl"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_candidate_delta_model import best_replace_delta, infer_replace_index, replace_delta_for_index  # noqa: E402
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import mask_stats, read_gt_mask, read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Current prediction raw JSONL.")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--candidate-jsonl", required=True, help="Candidate diagnostic JSONL with top_candidates.")
    parser.add_argument("--gt-jsonl", default=str(DEFAULT_GT))
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--candidate-field", default="top_candidates")
    parser.add_argument("--low-loc-threshold", type=float, default=0.06)
    parser.add_argument("--positive-eps", type=float, default=1e-4)
    parser.add_argument("--top-actions", type=int, default=8)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_pipe_path(path).read_text(encoding="utf-8"))


def gt_rows_by_id(path: str | Path, debug_root: Path) -> dict[str, dict[str, Any]]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = debug_root / p
    rows = read_jsonl(p)
    return {
        str(row.get("sample_id") or Path(str(row.get("image_file") or row.get("image_name") or "")).stem): row
        for row in rows
    }


def candidate_box(candidate: dict[str, Any]) -> list[int] | None:
    raw = candidate.get("box")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def sanitize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    text = str(candidate.get("text") or "")
    meta = candidate.get("meta") or {}
    rank = meta.get("v145ocranchor_rank") or meta.get("v87b_rank") or meta.get("v87_rank") or {}
    return {
        "label": candidate.get("label"),
        "family": candidate.get("family"),
        "source": candidate.get("source"),
        "score": candidate.get("score"),
        "box": candidate.get("box"),
        "text_preview": text[:180],
        "rank_score": rank.get("score"),
        "query_hits": rank.get("query_hits") or [],
        "number_hits": rank.get("number_hits") or [],
        "visual_hits": rank.get("visual_hits") or [],
    }


def family_summary(rows: list[dict[str, Any]], eps: float) -> dict[str, dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family") or "")].append(row)
    out: dict[str, dict[str, Any]] = {}
    for family, items in sorted(by_family.items()):
        best_oracle = max((float(x.get("best_delta") or 0.0) for x in items), default=0.0)
        best_deploy = max((float(x.get("deployable_delta") or 0.0) for x in items), default=0.0)
        out[family] = {
            "count": len(items),
            "positive_oracle": sum(1 for x in items if float(x.get("best_delta") or 0.0) > eps),
            "positive_deployable": sum(1 for x in items if float(x.get("deployable_delta") or 0.0) > eps),
            "best_oracle_delta": best_oracle,
            "best_deployable_delta": best_deploy,
            "mean_oracle_delta": mean(float(x.get("best_delta") or 0.0) for x in items),
            "mean_deployable_delta": mean(float(x.get("deployable_delta") or 0.0) for x in items),
        }
    return out


def summarize_samples(samples: list[dict[str, Any]], eps: float) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    return {
        "sample_count": len(samples),
        "mean_base_f1": mean(float(x.get("base_f1") or 0.0) for x in samples),
        "mean_best_oracle_delta": mean(float(x.get("best_oracle_delta") or 0.0) for x in samples),
        "mean_best_deployable_delta": mean(float(x.get("best_deployable_delta") or 0.0) for x in samples),
        "positive_oracle_samples": sum(1 for x in samples if float(x.get("best_oracle_delta") or 0.0) > eps),
        "positive_deployable_samples": sum(1 for x in samples if float(x.get("best_deployable_delta") or 0.0) > eps),
        "replace_beats_append_samples": sum(
            1 for x in samples if float(x.get("best_oracle_replace_delta") or 0.0) > float(x.get("best_append_delta") or 0.0) + eps
        ),
        "append_beats_replace_samples": sum(
            1 for x in samples if float(x.get("best_append_delta") or 0.0) > float(x.get("best_oracle_replace_delta") or 0.0) + eps
        ),
        "best_oracle_family_counts": dict(Counter(str(x.get("best_oracle_family") or "") for x in samples)),
        "best_deployable_family_counts": dict(Counter(str(x.get("best_deployable_family") or "") for x in samples)),
    }


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)

    eval_data = read_json(args.eval_json)
    eval_by_id = {str(sample.get("sample_id") or ""): sample for sample in eval_data.get("samples") or []}
    gt_by_id = gt_rows_by_id(args.gt_jsonl, debug_root)
    raw_by_id = {
        sample_id_from_row(row): row
        for row in read_jsonl(resolve_pipe_path(args.input_jsonl))
    }

    rows_out: list[dict[str, Any]] = []
    action_rows_all: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate_row in read_jsonl(resolve_pipe_path(args.candidate_jsonl)):
        sid = str(candidate_row.get("sample_id") or "")
        sample_eval = eval_by_id.get(sid) or {}
        loc = float(sample_eval.get("loc_score") or candidate_row.get("baseline", {}).get("loc_score") or 0.0)
        if sample_eval.get("gt_label") != "FORGED" or sample_eval.get("pred_label") != "FORGED":
            continue
        if loc >= args.low_loc_threshold:
            continue
        raw_row = raw_by_id.get(sid)
        gt_row = gt_by_id.get(sid)
        if not raw_row or not gt_row:
            missing.append(sid)
            continue
        width = int(candidate_row.get("width") or sample_eval.get("width") or raw_row.get("width") or 0)
        height = int(candidate_row.get("height") or sample_eval.get("height") or raw_row.get("height") or 0)
        if width <= 0 or height <= 0:
            missing.append(sid)
            continue

        gt_mask = read_gt_mask(gt_row, width, height, debug_root)
        existing = report_boxes(str(raw_row.get("raw_output") or ""))
        base_f1 = float(mask_stats(gt_mask, existing, width, height).get("mask_f1") or 0.0)

        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        actions: list[dict[str, Any]] = []
        candidates = candidate_row.get(args.candidate_field) or []
        for rank_idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            box = candidate_box(candidate)
            if not box:
                continue
            family = str(candidate.get("family") or "")
            key = (family, tuple(box))
            if key in seen:
                continue
            seen.add(key)
            append_f1 = float(mask_stats(gt_mask, existing + [box], width, height).get("mask_f1") or 0.0)
            append_delta = append_f1 - base_f1
            oracle_replace_delta, oracle_replace_index, oracle_replace_f1 = best_replace_delta(
                gt_mask=gt_mask,
                existing=existing,
                candidate_box=box,
                width=width,
                height=height,
                base_f1=base_f1,
            )
            inferred_replace_index = infer_replace_index(box, existing)
            inferred_replace_delta, inferred_replace_f1 = replace_delta_for_index(
                gt_mask=gt_mask,
                existing=existing,
                candidate_box=box,
                replace_index=inferred_replace_index,
                width=width,
                height=height,
                base_f1=base_f1,
            )
            if oracle_replace_delta >= append_delta:
                best_delta = oracle_replace_delta
                best_mode = "replace"
                best_f1 = oracle_replace_f1
            else:
                best_delta = append_delta
                best_mode = "append"
                best_f1 = append_f1
            if inferred_replace_delta >= append_delta:
                deployable_delta = inferred_replace_delta
                deployable_mode = "replace"
                deployable_f1 = inferred_replace_f1
            else:
                deployable_delta = append_delta
                deployable_mode = "append"
                deployable_f1 = append_f1
            action = {
                "sample_id": sid,
                "rank_idx": rank_idx,
                "family": family,
                "source": candidate.get("source"),
                "label": candidate.get("label"),
                "box": box,
                "area_ratio": box_area(box) / max(1, width * height),
                "append_delta": append_delta,
                "append_f1": append_f1,
                "oracle_replace_delta": oracle_replace_delta,
                "oracle_replace_index": oracle_replace_index,
                "oracle_replace_f1": oracle_replace_f1,
                "inferred_replace_delta": inferred_replace_delta,
                "inferred_replace_index": inferred_replace_index,
                "inferred_replace_f1": inferred_replace_f1,
                "best_delta": best_delta,
                "best_mode": best_mode,
                "best_f1": best_f1,
                "deployable_delta": deployable_delta,
                "deployable_mode": deployable_mode,
                "deployable_f1": deployable_f1,
                "candidate": sanitize_candidate(candidate),
            }
            actions.append(action)
            action_rows_all.append(action)

        actions_by_oracle = sorted(actions, key=lambda x: float(x.get("best_delta") or 0.0), reverse=True)
        actions_by_deploy = sorted(actions, key=lambda x: float(x.get("deployable_delta") or 0.0), reverse=True)
        best_oracle = actions_by_oracle[0] if actions_by_oracle else {}
        best_deploy = actions_by_deploy[0] if actions_by_deploy else {}
        best_append = max(actions, key=lambda x: float(x.get("append_delta") or 0.0), default={})
        best_replace = max(actions, key=lambda x: float(x.get("oracle_replace_delta") or 0.0), default={})
        rows_out.append(
            {
                "sample_id": sid,
                "language_code": sample_eval.get("language_code"),
                "loc_score": loc,
                "base_f1": base_f1,
                "existing_box_count": len(existing),
                "candidate_count": len(actions),
                "best_oracle_delta": float(best_oracle.get("best_delta") or 0.0),
                "best_oracle_mode": best_oracle.get("best_mode"),
                "best_oracle_family": best_oracle.get("family"),
                "best_oracle_box": best_oracle.get("box"),
                "best_deployable_delta": float(best_deploy.get("deployable_delta") or 0.0),
                "best_deployable_mode": best_deploy.get("deployable_mode"),
                "best_deployable_family": best_deploy.get("family"),
                "best_deployable_box": best_deploy.get("box"),
                "best_append_delta": float(best_append.get("append_delta") or 0.0),
                "best_oracle_replace_delta": float(best_replace.get("oracle_replace_delta") or 0.0),
                "best_oracle_replace_index": best_replace.get("oracle_replace_index"),
                "family_summary": family_summary(actions, args.positive_eps),
                "top_oracle_actions": actions_by_oracle[: args.top_actions],
                "top_deployable_actions": actions_by_deploy[: args.top_actions],
            }
        )

    rows_out.sort(key=lambda x: (float(x.get("best_oracle_delta") or 0.0), -float(x.get("loc_score") or 0.0)), reverse=True)
    output_jsonl = resolve_pipe_path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lowloc = rows_out
    positive = [x for x in lowloc if float(x.get("best_oracle_delta") or 0.0) > args.positive_eps]
    deploy_positive = [x for x in lowloc if float(x.get("best_deployable_delta") or 0.0) > args.positive_eps]
    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "eval_json": str(resolve_pipe_path(args.eval_json)),
        "candidate_jsonl": str(resolve_pipe_path(args.candidate_jsonl)),
        "gt_jsonl": str(Path(args.gt_jsonl).expanduser()),
        "output_json": str(resolve_pipe_path(args.output_json)),
        "output_jsonl": str(output_jsonl),
        "low_loc_threshold": args.low_loc_threshold,
        "positive_eps": args.positive_eps,
        "groups": {
            "lowloc_gt_forged_pred_forged": summarize_samples(lowloc, args.positive_eps),
            "oracle_positive": summarize_samples(positive, args.positive_eps),
            "deployable_positive": summarize_samples(deploy_positive, args.positive_eps),
        },
        "action_family_summary": family_summary(action_rows_all, args.positive_eps),
        "top_oracle_samples": lowloc[: min(20, len(lowloc))],
        "missing_sample_ids": sorted(set(missing)),
        "diagnostic_only": True,
        "gt_leakage_guard": "GT masks are used only in this offline diagnostic after candidate generation; do not use these oracle choices in prompts.",
    }
    output_json = resolve_pipe_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("output_json", "output_jsonl", "groups", "missing_sample_ids")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
