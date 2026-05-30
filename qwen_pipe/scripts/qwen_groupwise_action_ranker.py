#!/usr/bin/env python3
"""Fold-local groupwise ranker over cached localization actions.

The pair-replace ridge model regresses action delta globally.  For low-S_Loc
samples this failed even when each sample had at least one positive broad
candidate.  This script changes the objective: within each training sample,
learn pairwise preferences between better and worse replacement actions, then
rank actions per held-out sample.  GT-derived deltas are used only to build
fold-local training comparisons and never enter prompts or inference inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_candidate_delta_model import replace_groundings, stable_fold  # noqa: E402
from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


def conclusion_is_forged(row: dict[str, Any]) -> bool:
    parsed = row.get("parsed") or {}
    conclusion = str(parsed.get("conclusion") or "").upper()
    if conclusion:
        return conclusion == "FORGED"
    raw = str(row.get("raw_output") or "").upper()
    return "FORGED" in raw and "AUTHENTIC" not in raw[:800]


def fit_pairwise_ranker(
    observations: list[dict[str, Any]],
    train_indices: list[int],
    *,
    alpha: float,
    pair_margin: float,
    max_pos_per_sample: int,
    max_neg_per_pos: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not train_indices:
        raise ValueError("empty training fold")
    x_train = np.asarray([observations[i]["features"] for i in train_indices], dtype=np.float64)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    by_sample: dict[str, list[int]] = defaultdict(list)
    for idx in train_indices:
        by_sample[str(observations[idx]["sample_id"])].append(idx)

    diffs: list[np.ndarray] = []
    targets: list[float] = []
    pair_samples = 0
    for _sid, idxs in by_sample.items():
        ranked = sorted(idxs, key=lambda i: float(observations[i].get("delta") or 0.0), reverse=True)
        positives = [i for i in ranked if float(observations[i].get("delta") or 0.0) > 0.0][:max_pos_per_sample]
        if not positives:
            continue
        # Hard negatives first: actions close to the top but still worse, then
        # true non-positive actions.  This gives the ranker old-box harm cases.
        for pos_idx in positives:
            pos_delta = float(observations[pos_idx].get("delta") or 0.0)
            negs = [
                i
                for i in ranked
                if i != pos_idx and pos_delta - float(observations[i].get("delta") or 0.0) >= pair_margin
            ][:max_neg_per_pos]
            zp = (np.asarray(observations[pos_idx]["features"], dtype=np.float64) - mean) / std
            for neg_idx in negs:
                zn = (np.asarray(observations[neg_idx]["features"], dtype=np.float64) - mean) / std
                diff = zp - zn
                diffs.append(diff)
                targets.append(1.0)
                diffs.append(-diff)
                targets.append(-1.0)
                pair_samples += 1
    if not diffs:
        # Degenerate fold fallback: zero model, no selections should survive a
        # positive min-score/margin gate.
        return np.zeros(x_train.shape[1], dtype=np.float64), mean, std, {"pair_samples": 0}
    x_pair = np.vstack(diffs)
    y_pair = np.asarray(targets, dtype=np.float64)
    xtx = x_pair.T @ x_pair
    reg = alpha * np.eye(xtx.shape[0])
    w = np.linalg.solve(xtx + reg, x_pair.T @ y_pair)
    return w, mean, std, {"pair_samples": pair_samples}


def score_model(obs: dict[str, Any], model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    w, mean, std = model
    x = np.asarray(obs["features"], dtype=np.float64)
    z = (x - mean) / std
    return float(z @ w)


def select_actions(
    observations: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    allowed_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    selected_by_sample: dict[str, list[dict[str, Any]]] = {}
    fold_pair_counts: dict[int, int] = {}
    for fold in range(args.folds):
        train_indices = [i for i, obs in enumerate(observations) if int(obs.get("fold") or stable_fold(str(obs.get("sample_id")), args.folds)) != fold]
        model_w, mean, std, meta = fit_pairwise_ranker(
            observations,
            train_indices,
            alpha=args.alpha,
            pair_margin=args.pair_margin,
            max_pos_per_sample=args.max_pos_per_sample,
            max_neg_per_pos=args.max_neg_per_pos,
        )
        fold_pair_counts[fold] = int(meta.get("pair_samples") or 0)
        model = (model_w, mean, std)
        for sample in sample_rows:
            sid = str(sample.get("sample_id") or "")
            if allowed_sample_ids is not None and sid not in allowed_sample_ids:
                selected_by_sample[sid] = []
                continue
            if stable_fold(sid, args.folds) != fold:
                continue
            idxs = list(sample.get("candidate_indices") or [])
            scored: list[dict[str, Any]] = []
            for idx in idxs:
                obs = observations[idx]
                action = str(obs.get("action") or "")
                if args.action_allowlist and action not in args.action_allowlist:
                    continue
                family = str((obs.get("candidate") or {}).get("family") or "")
                if args.family_allowlist and family not in args.family_allowlist:
                    continue
                source = str((obs.get("candidate") or {}).get("source") or "")
                if args.allowed_source_prefixes and not any(
                    source.startswith(prefix) for prefix in args.allowed_source_prefixes
                ):
                    continue
                if args.blocked_source_substrings and any(
                    needle in source for needle in args.blocked_source_substrings
                ):
                    continue
                score = score_model(obs, model)
                out = dict(obs)
                out["groupwise_score"] = score
                scored.append(out)
            scored.sort(key=lambda o: float(o.get("groupwise_score") or 0.0), reverse=True)
            if not scored:
                selected_by_sample[sid] = []
                continue
            top_score = float(scored[0].get("groupwise_score") or 0.0)
            second_score = float(scored[1].get("groupwise_score") or 0.0) if len(scored) > 1 else -1e9
            margin = top_score - second_score
            picked: list[dict[str, Any]] = []
            used_replace: set[int] = set()
            used_boxes: set[tuple[int, int, int, int]] = set()
            for obs in scored:
                if len(picked) >= args.apply_top_n:
                    break
                score = float(obs.get("groupwise_score") or 0.0)
                if score < args.min_score:
                    continue
                if not picked and margin < args.min_margin:
                    continue
                replace_index = int(obs.get("replace_index") if obs.get("replace_index") is not None else -1)
                if replace_index < 0:
                    continue
                if replace_index in used_replace:
                    continue
                box = [int(v) for v in (obs.get("candidate") or {}).get("box") or []]
                if len(box) < 4:
                    continue
                key = tuple(box[:4])
                if key in used_boxes:
                    continue
                obs["_rank_margin"] = margin
                picked.append(obs)
                used_replace.add(replace_index)
                used_boxes.add(key)
            selected_by_sample[sid] = picked
    deltas = [float(obs.get("delta") or 0.0) for rows in selected_by_sample.values() for obs in rows]
    return {
        "selected_by_sample": selected_by_sample,
        "fold_pair_counts": fold_pair_counts,
        "selected_samples": sum(1 for rows in selected_by_sample.values() if rows),
        "selected_actions": len(deltas),
        "selected_target_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "selected_target_delta_positive_rate": float(np.mean([d > 0 for d in deltas])) if deltas else 0.0,
        "selected_target_delta_sum": float(np.sum(deltas)) if deltas else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-cache", required=True)
    parser.add_argument("--base-jsonl", required=True)
    parser.add_argument("--applied-raw-jsonl", required=True)
    parser.add_argument("--diag-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--pair-margin", type=float, default=0.005)
    parser.add_argument("--max-pos-per-sample", type=int, default=3)
    parser.add_argument("--max-neg-per-pos", type=int, default=24)
    parser.add_argument("--apply-top-n", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=-1e9)
    parser.add_argument("--min-margin", type=float, default=-1e9)
    parser.add_argument("--family-allowlist", default="")
    parser.add_argument("--action-allowlist", default="replace")
    parser.add_argument(
        "--allowed-source-prefixes",
        default="",
        help="Comma-separated candidate source prefixes allowed during selection.",
    )
    parser.add_argument(
        "--blocked-source-substrings",
        default="",
        help="Comma-separated substrings that reject candidate sources during selection.",
    )
    parser.add_argument(
        "--only-pred-forged",
        action="store_true",
        help="Only apply replacements to samples whose current parsed/raw conclusion is FORGED.",
    )
    args = parser.parse_args()

    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    cache = json.loads(resolve_pipe_path(args.observation_cache).read_text(encoding="utf-8"))
    observations = list(cache.get("observations") or [])
    sample_rows = list(cache.get("sample_rows") or [])
    args.family_allowlist = {part.strip() for part in str(args.family_allowlist or "").split(",") if part.strip()}
    args.action_allowlist = {part.strip() for part in str(args.action_allowlist or "").split(",") if part.strip()}
    args.allowed_source_prefixes = [
        part.strip() for part in str(args.allowed_source_prefixes or "").split(",") if part.strip()
    ]
    args.blocked_source_substrings = [
        part.strip() for part in str(args.blocked_source_substrings or "").split(",") if part.strip()
    ]

    base_rows = read_jsonl(resolve_pipe_path(args.base_jsonl))
    allowed_sample_ids: set[str] | None = None
    if args.only_pred_forged:
        allowed_sample_ids = {sample_id_from_row(row) for row in base_rows if conclusion_is_forged(row)}

    result = select_actions(observations, sample_rows, args, allowed_sample_ids=allowed_sample_ids)
    selected_by_sample: dict[str, list[dict[str, Any]]] = result.pop("selected_by_sample")

    diag_path = resolve_pipe_path(args.diag_jsonl)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    selected_family_counts: Counter[str] = Counter()
    with diag_path.open("w", encoding="utf-8") as fh:
        for sample in sample_rows:
            sid = str(sample.get("sample_id") or "")
            selected = selected_by_sample.get(sid) or []
            for obs in selected:
                cand = obs.get("candidate") or {}
                selected_family_counts[f"{cand.get('family')}:{cand.get('source')}"] += 1
            fh.write(
                json.dumps(
                    {
                        "sample_id": sid,
                        "selected_count": len(selected),
                        "selected": [
                            {
                                "replace_index": obs.get("replace_index"),
                                "groupwise_score": obs.get("groupwise_score"),
                                "rank_margin": obs.get("_rank_margin"),
                                "target_delta": obs.get("delta"),
                                "base_f1": obs.get("base_f1"),
                                "new_f1": obs.get("new_f1"),
                                "candidate": obs.get("candidate"),
                            }
                            for obs in selected
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    raw_path = resolve_pipe_path(args.applied_raw_jsonl)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    selected_lookup = selected_by_sample
    with raw_path.open("w", encoding="utf-8") as fh:
        for row in base_rows:
            sid = sample_id_from_row(row)
            selected = selected_lookup.get(sid) or []
            if selected:
                row = dict(row)
                replacements = {
                    int(obs["replace_index"]): [int(v) for v in (obs.get("candidate") or {}).get("box") or []][:4]
                    for obs in selected
                    if int(obs.get("replace_index") if obs.get("replace_index") is not None else -1) >= 0
                }
                report, replaced_count = replace_groundings(str(row.get("raw_output") or ""), replacements)
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs["qwen_pipe_groupwise_action_ranker"] = {
                    "applied": True,
                    "diagnostic_fold_local": True,
                    "selected_actions": len(selected),
                    "replaced_count": replaced_count,
                    "policy": "Fold-local groupwise pairwise ranker over cached candidate-by-old-box actions.",
                    "selected": [
                        {
                            "replace_index": obs.get("replace_index"),
                            "groupwise_score": obs.get("groupwise_score"),
                            "rank_margin": obs.get("_rank_margin"),
                            "target_delta_local_diagnostic": obs.get("delta"),
                            "candidate": obs.get("candidate"),
                        }
                        for obs in selected
                    ],
                }
                row["stage_outputs"] = stage_outputs
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_deltas = [float(obs.get("delta") or 0.0) for rows in selected_by_sample.values() for obs in rows]
    full_summary = {
        **result,
        "observation_cache": str(resolve_pipe_path(args.observation_cache)),
        "base_jsonl": str(resolve_pipe_path(args.base_jsonl)),
        "applied_raw_jsonl": str(raw_path),
        "diag_jsonl": str(diag_path),
        "sample_count": len(sample_rows),
        "observation_count": len(observations),
        "params": {
            "folds": args.folds,
            "alpha": args.alpha,
            "pair_margin": args.pair_margin,
            "max_pos_per_sample": args.max_pos_per_sample,
            "max_neg_per_pos": args.max_neg_per_pos,
            "apply_top_n": args.apply_top_n,
            "min_score": args.min_score,
            "min_margin": args.min_margin,
            "only_pred_forged": bool(args.only_pred_forged),
            "family_allowlist": sorted(args.family_allowlist),
            "action_allowlist": sorted(args.action_allowlist),
            "allowed_source_prefixes": sorted(args.allowed_source_prefixes),
            "blocked_source_substrings": sorted(args.blocked_source_substrings),
        },
        "allowed_sample_count": len(allowed_sample_ids) if allowed_sample_ids is not None else None,
        "selected_family_counts": dict(selected_family_counts),
        "selected_target_delta_min": float(np.min(all_deltas)) if all_deltas else 0.0,
        "selected_target_delta_max": float(np.max(all_deltas)) if all_deltas else 0.0,
        "diagnostic_note": "GT deltas train pairwise rankers only on non-heldout folds; sample set may be GT-diagnostic if the cache was built that way.",
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(full_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
