#!/usr/bin/env python3
"""Diagnose low-localization samples against cached pair candidates.

This script is intentionally evaluation-side only: GT-derived loc scores and
candidate deltas are used to understand failure modes after predictions exist.
It must not feed any GT field into prompts or model calls.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]


def resolve_pipe_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PIPE_ROOT / p
    return p


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_pipe_path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in resolve_pipe_path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def count_boxes(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int):
        return value
    return 0


def candidate_brief(obs: dict[str, Any] | None) -> dict[str, Any] | None:
    if not obs:
        return None
    cand = obs.get("candidate") or {}
    meta = cand.get("meta") or {}
    rank = meta.get("v145ocranchor_rank") or meta.get("v87b_rank") or meta.get("v87_rank") or {}
    return {
        "action": obs.get("action"),
        "replace_index": obs.get("replace_index"),
        "delta": float(obs.get("delta") or 0.0),
        "candidate_index": obs.get("candidate_index"),
        "family": cand.get("family"),
        "source": cand.get("source"),
        "box": cand.get("box"),
        "text": str(cand.get("text") or "")[:180],
        "rank_score": rank.get("score"),
        "query_score": rank.get("query_score"),
        "token_hit_count": rank.get("token_hit_count"),
        "number_hits": rank.get("number_hits"),
        "visual_hits": rank.get("visual_hits"),
    }


def classify_sample(
    sample: dict[str, Any],
    observations: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    min_positive_delta: float,
) -> str:
    if str(sample.get("pred_label") or "").upper() != "FORGED":
        return "detector_false_negative"
    if not observations:
        return "no_pair_candidates"
    max_delta = max(float(o.get("delta") or 0.0) for o in observations)
    if max_delta <= min_positive_delta:
        return "candidate_gap_no_positive_pair"
    if not selected:
        return "selector_missed_positive_pair"
    selected_best = max(float(s.get("target_delta") or 0.0) for s in selected)
    if selected_best <= min_positive_delta:
        return "selector_chose_nonpositive_pair"
    return "covered_by_positive_selection"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--observation-cache", required=True)
    parser.add_argument("--pair-output-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--loc-threshold", type=float, default=0.02)
    parser.add_argument("--min-positive-delta", type=float, default=1e-6)
    parser.add_argument("--top-n-examples", type=int, default=20)
    args = parser.parse_args()

    eval_data = load_json(args.eval_json)
    cache = load_json(args.observation_cache)
    observations = list(cache.get("observations") or [])
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        by_sid[str(obs.get("sample_id") or "")].append(obs)

    pair_rows = {str(r.get("sample_id") or ""): r for r in read_jsonl(args.pair_output_jsonl)}
    forged = [s for s in eval_data.get("samples") or [] if str(s.get("gt_label") or "").upper() == "FORGED"]
    lowloc = [s for s in forged if float(s.get("loc_score") or 0.0) < args.loc_threshold]

    bucket_counts: Counter[str] = Counter()
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    family_positive_counts: Counter[str] = Counter()
    source_positive_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for sample in lowloc:
        sid = str(sample.get("sample_id") or "")
        obs_list = by_sid.get(sid, [])
        selected = list((pair_rows.get(sid) or {}).get("selected") or [])
        best_obs = max(obs_list, key=lambda o: float(o.get("delta") or 0.0), default=None)
        positive_obs = [o for o in obs_list if float(o.get("delta") or 0.0) > args.min_positive_delta]
        bucket = classify_sample(sample, obs_list, selected, min_positive_delta=args.min_positive_delta)
        bucket_counts[bucket] += 1
        by_language[str(sample.get("language_code") or "")][bucket] += 1
        if positive_obs:
            best_pos = max(positive_obs, key=lambda o: float(o.get("delta") or 0.0))
            cand = best_pos.get("candidate") or {}
            family_positive_counts[str(cand.get("family") or "")] += 1
            source_positive_counts[str(cand.get("source") or "")] += 1
        else:
            best_pos = None
        rows.append(
            {
                "sample_id": sid,
                "language_code": sample.get("language_code"),
                "loc_score": sample.get("loc_score"),
                "pred_label": sample.get("pred_label"),
                "pred_box_count": count_boxes(sample.get("pred_boxes")),
                "gt_box_count": count_boxes(sample.get("gt_boxes")),
                "bucket": bucket,
                "pair_observation_count": len(obs_list),
                "positive_pair_count": len(positive_obs),
                "selected_count": len(selected),
                "selected_target_delta_max": max([float(s.get("target_delta") or 0.0) for s in selected] or [0.0]),
                "best_candidate": candidate_brief(best_obs),
                "best_positive_candidate": candidate_brief(best_pos),
                "selected": selected[:3],
            }
        )

    rows.sort(key=lambda r: (str(r["bucket"]), float(r["loc_score"] or 0.0), -float((r.get("best_candidate") or {}).get("delta") or 0.0)))
    out = {
        "eval_json": str(resolve_pipe_path(args.eval_json)),
        "observation_cache": str(resolve_pipe_path(args.observation_cache)),
        "pair_output_jsonl": str(resolve_pipe_path(args.pair_output_jsonl)),
        "loc_threshold": args.loc_threshold,
        "forged_count": len(forged),
        "lowloc_count": len(lowloc),
        "bucket_counts": dict(bucket_counts),
        "by_language": {lang: dict(counts) for lang, counts in sorted(by_language.items())},
        "best_positive_candidate_families": dict(family_positive_counts.most_common()),
        "best_positive_candidate_sources_top20": dict(source_positive_counts.most_common(20)),
        "examples": rows[: args.top_n_examples],
        "all_rows": rows,
    }

    out_path = resolve_pipe_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["lowloc_count", "bucket_counts", "by_language", "best_positive_candidate_families"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
