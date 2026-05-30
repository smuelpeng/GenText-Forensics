#!/usr/bin/env python3
"""Fold-local KNN rescue for hard-positive localization actions.

The v231 coverage diagnostic showed many low-localization failures already have
positive pair candidates, but the global two-head scorer misses them.  This
script adds a second-stage prototype retriever trained only on other folds:

1. Build positive prototypes from actions that improve low-localization training
   samples.
2. Build negative prototypes from non-improving actions.
3. For held-out samples, replace or append one grounding only when its KNN
   score clears a threshold chosen on the training folds.

GT-derived deltas are used only for training/diagnostics after predictions
exist; no model prompt is constructed here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import insert_extra_anomalies, read_jsonl, sample_id_from_row  # noqa: E402
from qwen_pair_replace_model import should_reject_ocr_replace  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402
from qwen_candidate_delta_model import replace_groundings  # noqa: E402


def load_eval(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(resolve_pipe_path(path).read_text(encoding="utf-8"))
    return {str(s.get("sample_id") or ""): s for s in data.get("samples") or []}


def load_cache(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    data = json.loads(resolve_pipe_path(path).read_text(encoding="utf-8"))
    observations = list(data.get("observations") or [])
    by_sid: dict[str, list[int]] = defaultdict(list)
    for idx, obs in enumerate(observations):
        by_sid[str(obs.get("sample_id") or "")].append(idx)
    return observations, by_sid


def read_selected(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(r.get("sample_id") or ""): r for r in read_jsonl(resolve_pipe_path(path))}


def is_lowloc(sample: dict[str, Any], threshold: float) -> bool:
    return (
        str(sample.get("gt_label") or "").upper() == "FORGED"
        and str(sample.get("pred_label") or "").upper() == "FORGED"
        and float(sample.get("loc_score") or 0.0) < threshold
    )


def row_is_pred_forged(sample: dict[str, Any] | None) -> bool:
    return str((sample or {}).get("pred_label") or "").upper() == "FORGED"


def candidate_rank(obs: dict[str, Any]) -> dict[str, Any]:
    cand = obs.get("candidate") or {}
    meta = cand.get("meta") or {}
    return meta.get("v145ocranchor_rank") or meta.get("v87b_rank") or meta.get("v87_rank") or {}


def selected_keys(row: dict[str, Any] | None) -> tuple[set[tuple[int, int, int, int]], set[int]]:
    boxes: set[tuple[int, int, int, int]] = set()
    replace_indices: set[int] = set()
    for item in (row or {}).get("selected") or []:
        cand = item.get("candidate") or {}
        box = cand.get("box") or []
        if len(box) >= 4:
            boxes.add(tuple(int(v) for v in box[:4]))
        idx = item.get("replace_index")
        if idx is not None and int(idx) >= 0:
            replace_indices.add(int(idx))
    return boxes, replace_indices


def observation_delta(obs: dict[str, Any], action_mode: str) -> float:
    if action_mode == "append":
        return float(obs.get("append_delta") or 0.0)
    return float(obs.get("replace_delta") or obs.get("delta") or 0.0)


def feature_matrix(observations: list[dict[str, Any]], indices: list[int]) -> np.ndarray:
    return np.asarray([observations[i].get("features") or [] for i in indices], dtype=np.float64)


LANGS = ["ar", "en", "id", "ms", "th", "zh", ""]
FAMILIES = ["ocr", "linegrid", "evidence", "token", "row", "patch", "grid", "paragraph", "existing"]


def box_area(box: list[int]) -> float:
    if len(box) < 4:
        return 0.0
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def sample_feature_vector(
    *,
    sid: str,
    sample: dict[str, Any],
    raw_row: dict[str, Any],
    obs_indices: list[int],
    observations: list[dict[str, Any]],
    previous_selected: dict[str, Any] | None,
    args: argparse.Namespace,
) -> list[float]:
    existing = report_boxes(str(raw_row.get("raw_output") or ""))
    selected = list((previous_selected or {}).get("selected") or [])
    allowed: list[dict[str, Any]] = [
        observations[i]
        for i in obs_indices
        if obs_allowed(
            observations[i],
            sample_row=raw_row,
            selected_row=previous_selected,
            args=args,
        )
    ]
    candidates = [o.get("candidate") or {} for o in allowed]
    ranks = [candidate_rank(o) for o in allowed]
    query_scores = [float(r.get("query_score") or 0.0) for r in ranks]
    rank_scores = [float(r.get("score") or 0.0) for r in ranks]
    token_hits = [int(r.get("token_hit_count") or 0) for r in ranks]
    texts = [str(c.get("text") or "") for c in candidates]
    digit_texts = [1.0 if re.search(r"\d", t) else 0.0 for t in texts]
    long_texts = [1.0 if len(t) >= 28 else 0.0 for t in texts]
    areas = [box_area(c.get("box") or []) for c in candidates]
    existing_areas = [box_area(b) for b in existing]
    family_counts = Counter(str(c.get("family") or "") for c in candidates)
    source_text = " ".join(str(c.get("source") or "") for c in candidates[:12])
    sel_scores = [float(s.get("selection_score") or 0.0) for s in selected]
    sel_pos = [float(s.get("pred_pos") or 0.0) for s in selected]
    sel_gain = [float(s.get("pred_gain") or 0.0) for s in selected]
    pred_box_count = len(existing)
    allowed_count = len(allowed)
    vec = [
        math.log1p(pred_box_count),
        math.log1p(allowed_count),
        math.log1p(len(obs_indices)),
        math.log1p(len(selected)),
        math.log1p(max(0, allowed_count - len(selected))),
        float(allowed_count) / max(1.0, float(pred_box_count)),
        float(pred_box_count) / 10.0,
        float((sample or {}).get("pred_report_len") or 0.0) / 4000.0,
        max(query_scores or [0.0]) / 30.0,
        float(np.mean(query_scores)) / 30.0 if query_scores else 0.0,
        max(rank_scores or [0.0]) / 50.0,
        float(np.mean(rank_scores)) / 50.0 if rank_scores else 0.0,
        max(token_hits or [0]) / 20.0,
        float(np.mean(token_hits)) / 20.0 if token_hits else 0.0,
        float(sum(q >= 6.0 for q in query_scores)) / max(1.0, allowed_count),
        float(sum(q >= 10.0 for q in query_scores)) / max(1.0, allowed_count),
        float(sum(digit_texts)) / max(1.0, allowed_count),
        float(sum(long_texts)) / max(1.0, allowed_count),
        max(areas or [0.0]) / max(1.0, max(existing_areas or [1.0])),
        float(np.mean(areas)) / max(1.0, max(existing_areas or [1.0])) if areas else 0.0,
        max(sel_scores or [0.0]),
        max(sel_pos or [0.0]),
        max(sel_gain or [0.0]),
        float(any("heading" in str(c.get("source") or "") for c in candidates)),
        float(any("scriptgrid" in source_text for _ in [0])),
        float(any("span_id" in source_text for _ in [0])),
    ]
    total_fams = max(1.0, sum(family_counts.values()))
    vec.extend(float(family_counts.get(fam) or 0.0) / total_fams for fam in FAMILIES)
    lang = str((sample or {}).get("language_code") or "")
    vec.extend(1.0 if lang == item else 0.0 for item in LANGS)
    return vec


def fit_ridge_classifier(x: np.ndarray, y: np.ndarray, *, reg: float, pos_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    weights = np.where(y > 0.5, pos_weight, 1.0)
    a = xb.T @ (xb * weights[:, None]) + reg * np.eye(xb.shape[1])
    a[0, 0] -= reg
    b = xb.T @ (y * weights)
    coef = np.linalg.solve(a, b)
    return coef, mean, std


def predict_ridge_classifier(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    coef, mean, std = model
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    return xb @ coef


def standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x - mean) / std, mean, std


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1)
    norm[norm < 1e-9] = 1.0
    return x / norm[:, None]


def topk_mean(sim: np.ndarray, k: int) -> np.ndarray:
    if sim.shape[1] == 0:
        return np.zeros(sim.shape[0], dtype=np.float64)
    kk = min(k, sim.shape[1])
    part = np.partition(sim, -kk, axis=1)[:, -kk:]
    return part.mean(axis=1)


def knn_scores(
    train_x: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    x: np.ndarray,
    *,
    k: int,
    neg_weight: float,
) -> np.ndarray:
    if not pos_mask.any():
        return np.full(x.shape[0], -1e9, dtype=np.float64)
    x_std, mean, std = standardize(train_x, x)
    train_std = (train_x - mean) / std
    x_norm = normalize_rows(x_std)
    train_norm = normalize_rows(train_std)
    pos_sim = x_norm @ train_norm[pos_mask].T
    pos = topk_mean(pos_sim, k)
    if neg_mask.any():
        neg_sim = x_norm @ train_norm[neg_mask].T
        neg = topk_mean(neg_sim, k)
    else:
        neg = 0.0
    return pos - neg_weight * neg


def obs_allowed(
    obs: dict[str, Any],
    *,
    sample_row: dict[str, Any],
    selected_row: dict[str, Any] | None,
    args: argparse.Namespace,
) -> bool:
    action = str(obs.get("action") or "")
    # qwen_candidate_delta_model caches are replacement-only and older
    # versions did not materialize an "action" field.
    if args.action_mode == "replace":
        if action and action != "replace":
            return False
        if int(obs.get("replace_index") if obs.get("replace_index") is not None else -1) < 0:
            return False
    elif args.action_mode == "append":
        if action and action != "append":
            return False
        if float(obs.get("append_delta") or 0.0) == 0.0 and float(obs.get("replace_delta") or 0.0) == 0.0:
            return False
    cand = obs.get("candidate") or {}
    family = str(cand.get("family") or "")
    if args.family_allowlist:
        allowed = {x.strip() for x in args.family_allowlist.split(",") if x.strip()}
        if family not in allowed:
            return False
    box = cand.get("box") or []
    if len(box) < 4:
        return False
    prev_boxes, prev_replace = selected_keys(selected_row)
    if tuple(int(v) for v in box[:4]) in prev_boxes:
        return False
    if args.action_mode == "replace" and int(obs.get("replace_index")) in prev_replace:
        return False
    occupied = report_boxes(str(sample_row.get("raw_output") or ""))
    guard_obs = obs
    if args.action_mode == "append" and action == "append":
        guard_obs = dict(obs)
        guard_obs["action"] = ""
    if should_reject_ocr_replace(guard_obs, occupied, args):
        return False
    return True


def choose_threshold(train_candidates: list[tuple[float, float]], mode: str) -> float:
    if not train_candidates:
        return 1e9
    scores = sorted({s for s, _d in train_candidates})
    best_thr = scores[-1] + 1.0
    best_key = (-1e18, 0.0, 0.0)
    for thr in scores:
        chosen = [delta for score, delta in train_candidates if score >= thr]
        if not chosen:
            continue
        total_delta = float(sum(chosen))
        precision = sum(1 for d in chosen if d > 0.0) / len(chosen)
        mean_delta = total_delta / len(chosen)
        if mode == "delta":
            key = (total_delta, precision, -len(chosen))
        elif mode == "precision":
            key = (precision, mean_delta, -len(chosen))
        else:
            key = (mean_delta, precision, -len(chosen))
        if key > best_key:
            best_key = key
            best_thr = thr
    return best_thr


def choose_dual_thresholds(train_candidates: list[tuple[float, float, float]], mode: str) -> tuple[float, float]:
    """Choose action and sample-risk thresholds from training-fold candidates."""
    if not train_candidates:
        return 1e9, 1e9
    action_scores = sorted({a for a, _r, _d in train_candidates})
    risk_scores = sorted({r for _a, r, _d in train_candidates})
    best = (-1e18, 0.0, 0.0, 0.0)
    best_pair = (action_scores[-1] + 1.0, risk_scores[-1] + 1.0)
    for action_thr in action_scores:
        for risk_thr in risk_scores:
            chosen = [delta for action, risk, delta in train_candidates if action >= action_thr and risk >= risk_thr]
            if not chosen:
                continue
            total_delta = float(sum(chosen))
            precision = sum(1 for d in chosen if d > 0.0) / len(chosen)
            mean_delta = total_delta / len(chosen)
            recall_proxy = sum(1 for d in chosen if d > 0.0)
            if mode == "delta":
                key = (total_delta, precision, -len(chosen), mean_delta)
            elif mode == "precision":
                key = (precision, mean_delta, total_delta, -len(chosen))
            else:
                key = (mean_delta, precision, total_delta, recall_proxy)
            if key > best:
                best = key
                best_pair = (action_thr, risk_thr)
    return best_pair


def apply_rescue(
    *,
    raw_path: Path,
    output_path: Path,
    selected_by_sid: dict[str, dict[str, Any]],
    rescue_by_sid: dict[str, dict[str, Any]],
    debug_root: Path,
) -> dict[str, int]:
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    stats = {"rows": 0, "changed": 0, "replace_attempts": 0, "append_attempts": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in read_jsonl(raw_path):
            stats["rows"] += 1
            sid = sample_id_from_row(row)
            rescue = rescue_by_sid.get(sid)
            if rescue:
                row = dict(row)
                report = str(row.get("raw_output") or "")
                obs = rescue["observation"]
                action = str(rescue.get("action") or "replace")
                replace_index = int(obs.get("replace_index") if obs.get("replace_index") is not None else -1)
                candidate = obs["candidate"]
                if action == "append":
                    report = insert_extra_anomalies(report, [candidate])
                    changed_count = 1
                    stats["append_attempts"] += 1
                else:
                    box = candidate["box"]
                    report, changed_count = replace_groundings(report, {replace_index: box})
                    stats["replace_attempts"] += 1
                if changed_count:
                    row["raw_output"] = report
                    row["parsed"] = parse_cct_report(report)
                    stage_outputs = dict(row.get("stage_outputs") or {})
                    stage_outputs["qwen_pipe_hard_positive_knn_rescue"] = {
                        "applied": True,
                        "action": action,
                        "replace_index": replace_index,
                        "candidate": candidate,
                        "knn_score": rescue.get("score"),
                        "threshold": rescue.get("threshold"),
                        "risk_score": rescue.get("risk_score"),
                        "risk_threshold": rescue.get("risk_threshold"),
                        "policy": "Fold-local KNN prototype rescue for hard-positive low-localization candidates.",
                    }
                    row["stage_outputs"] = stage_outputs
                    stats["changed"] += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-raw-jsonl", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--observation-cache", required=True)
    parser.add_argument("--previous-pair-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--diag-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(PIPE_ROOT.parent / "debug_distribution"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--loc-threshold", type=float, default=0.02)
    parser.add_argument("--min-positive-delta", type=float, default=0.0005)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--neg-weight", type=float, default=0.35)
    parser.add_argument("--threshold-mode", choices=["delta", "precision", "mean"], default="delta")
    parser.add_argument("--action-mode", choices=["replace", "append"], default="replace")
    parser.add_argument("--family-allowlist", default="ocr,linegrid,evidence")
    parser.add_argument("--max-per-sample", type=int, default=1)
    parser.add_argument("--sample-risk-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-risk-reg", type=float, default=3.0)
    parser.add_argument("--sample-risk-pos-weight", type=float, default=8.0)
    parser.add_argument("--ocr-replace-guard", choices=["off", "conservative"], default="conservative")
    parser.add_argument("--ocr-guard-long-text-min-chars", type=int, default=28)
    parser.add_argument("--ocr-guard-long-text-max-query", type=float, default=4.0)
    parser.add_argument("--ocr-guard-long-text-max-token-hits", type=int, default=3)
    parser.add_argument("--ocr-guard-digit-ratio", type=float, default=0.18)
    parser.add_argument("--ocr-guard-value-text-max-chars", type=int, default=24)
    parser.add_argument("--ocr-guard-header-y-ratio", type=float, default=0.18)
    parser.add_argument("--ocr-guard-footer-y-ratio", type=float, default=0.78)
    parser.add_argument("--ocr-guard-header-max-query", type=float, default=6.0)
    args = parser.parse_args()

    eval_samples = load_eval(args.eval_json)
    observations, by_sid = load_cache(args.observation_cache)
    previous_selected = read_selected(args.previous_pair_jsonl)
    raw_rows = {sample_id_from_row(r): r for r in read_jsonl(resolve_pipe_path(args.base_raw_jsonl))}

    all_indices = list(range(len(observations)))
    x_all = feature_matrix(observations, all_indices)
    lowloc_ids = {sid for sid, sample in eval_samples.items() if is_lowloc(sample, args.loc_threshold)}
    rescue_by_sid: dict[str, dict[str, Any]] = {}
    diag_rows: list[dict[str, Any]] = []
    thresholds: dict[int, float] = {}
    risk_thresholds: dict[int, float] = {}
    sample_features = {
        sid: sample_feature_vector(
            sid=sid,
            sample=eval_samples.get(sid) or {},
            raw_row=raw_rows.get(sid) or {},
            obs_indices=by_sid.get(sid, []),
            observations=observations,
            previous_selected=previous_selected.get(sid),
            args=args,
        )
        for sid in raw_rows
    }

    for fold in range(args.folds):
        train_idx = [i for i, obs in enumerate(observations) if int(obs.get("fold") or 0) != fold]
        test_sids = sorted({str(obs.get("sample_id") or "") for obs in observations if int(obs.get("fold") or 0) == fold})
        if not train_idx or not test_sids:
            continue
        train_x = x_all[train_idx]
        train_low_pos = np.asarray(
            [
                str(observations[i].get("sample_id") or "") in lowloc_ids
                and observation_delta(observations[i], args.action_mode) > args.min_positive_delta
                for i in train_idx
            ],
            dtype=bool,
        )
        train_neg = np.asarray([observation_delta(observations[i], args.action_mode) <= 0.0 for i in train_idx], dtype=bool)
        if not train_low_pos.any():
            continue

        train_sids = sorted({str(observations[i].get("sample_id") or "") for i in train_idx if row_is_pred_forged(eval_samples.get(str(observations[i].get("sample_id") or "")))})
        risk_model: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        train_risk_by_sid: dict[str, float] = {}
        test_risk_by_sid: dict[str, float] = {}
        if args.sample_risk_gate and train_sids:
            risk_x = np.asarray([sample_features[sid] for sid in train_sids], dtype=np.float64)
            risk_y = np.asarray([1.0 if sid in lowloc_ids else 0.0 for sid in train_sids], dtype=np.float64)
            risk_model = fit_ridge_classifier(
                risk_x,
                risk_y,
                reg=args.sample_risk_reg,
                pos_weight=args.sample_risk_pos_weight,
            )
            train_pred = predict_ridge_classifier(risk_x, risk_model)
            train_risk_by_sid = {sid: float(score) for sid, score in zip(train_sids, train_pred)}
            test_pred_sids = [sid for sid in test_sids if sid in sample_features]
            if test_pred_sids:
                test_x = np.asarray([sample_features[sid] for sid in test_pred_sids], dtype=np.float64)
                test_pred = predict_ridge_classifier(test_x, risk_model)
                test_risk_by_sid = {sid: float(score) for sid, score in zip(test_pred_sids, test_pred)}

        # Choose a fold-local threshold by asking the retriever to pick one
        # candidate per training sample, then maximizing training-fold delta.
        train_best: list[tuple[float, float]] = []
        train_best_dual: list[tuple[float, float, float]] = []
        train_by_sid: dict[str, list[int]] = defaultdict(list)
        for i in train_idx:
            train_by_sid[str(observations[i].get("sample_id") or "")].append(i)
        for sid, indices in train_by_sid.items():
            sample = eval_samples.get(sid)
            if not row_is_pred_forged(sample):
                continue
            raw_row = raw_rows.get(sid) or {}
            allowed = [
                i
                for i in indices
                if obs_allowed(
                    observations[i],
                    sample_row=raw_row,
                    selected_row=previous_selected.get(sid),
                    args=args,
                )
            ]
            if not allowed:
                continue
            scores = knn_scores(
                train_x,
                train_low_pos,
                train_neg,
                x_all[allowed],
                k=args.k,
                neg_weight=args.neg_weight,
            )
            best_local = int(np.argmax(scores))
            action_score = float(scores[best_local])
            delta = observation_delta(observations[allowed[best_local]], args.action_mode)
            train_best.append((action_score, delta))
            train_best_dual.append((action_score, train_risk_by_sid.get(sid, 0.0), delta))
        if args.sample_risk_gate:
            threshold, risk_threshold = choose_dual_thresholds(train_best_dual, args.threshold_mode)
        else:
            threshold = choose_threshold(train_best, args.threshold_mode)
            risk_threshold = -1e18
        thresholds[fold] = threshold
        risk_thresholds[fold] = risk_threshold

        for sid in test_sids:
            sample = eval_samples.get(sid)
            if not row_is_pred_forged(sample):
                continue
            raw_row = raw_rows.get(sid) or {}
            allowed = [
                i
                for i in by_sid.get(sid, [])
                if obs_allowed(
                    observations[i],
                    sample_row=raw_row,
                    selected_row=previous_selected.get(sid),
                    args=args,
                )
            ]
            if not allowed:
                continue
            scores = knn_scores(
                train_x,
                train_low_pos,
                train_neg,
                x_all[allowed],
                k=args.k,
                neg_weight=args.neg_weight,
            )
            order = np.argsort(scores)[::-1]
            for rank_idx in order[: max(1, args.max_per_sample)]:
                obs_idx = allowed[int(rank_idx)]
                score = float(scores[int(rank_idx)])
                obs = observations[obs_idx]
                risk_score = test_risk_by_sid.get(sid, 0.0)
                selected = score >= threshold and risk_score >= risk_threshold
                diag = {
                    "sample_id": sid,
                    "fold": fold,
                    "selected": selected,
                    "score": score,
                    "risk_score": risk_score,
                    "threshold": threshold,
                    "risk_threshold": risk_threshold,
                    "target_delta": observation_delta(obs, args.action_mode),
                    "action": args.action_mode,
                    "is_lowloc": sid in lowloc_ids,
                    "candidate": obs.get("candidate"),
                    "replace_index": obs.get("replace_index"),
                }
                diag_rows.append(diag)
                if selected and sid not in rescue_by_sid:
                    rescue_by_sid[sid] = {
                        "action": args.action_mode,
                        "score": score,
                        "threshold": threshold,
                        "risk_score": risk_score,
                        "risk_threshold": risk_threshold,
                        "observation": obs,
                    }
                break

    diag_path = resolve_pipe_path(args.diag_jsonl)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with diag_path.open("w", encoding="utf-8") as fh:
        for row in diag_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    apply_stats = apply_rescue(
        raw_path=resolve_pipe_path(args.base_raw_jsonl),
        output_path=resolve_pipe_path(args.output_jsonl),
        selected_by_sid=previous_selected,
        rescue_by_sid=rescue_by_sid,
        debug_root=Path(args.debug_root).expanduser().resolve(),
    )

    selected_rows = [r for r in diag_rows if r.get("selected")]
    summary = {
        "base_raw_jsonl": str(resolve_pipe_path(args.base_raw_jsonl)),
        "output_jsonl": str(resolve_pipe_path(args.output_jsonl)),
        "eval_json": str(resolve_pipe_path(args.eval_json)),
        "observation_cache": str(resolve_pipe_path(args.observation_cache)),
        "previous_pair_jsonl": str(resolve_pipe_path(args.previous_pair_jsonl)),
        "fold_thresholds": thresholds,
        "fold_risk_thresholds": risk_thresholds,
        "lowloc_count": len(lowloc_ids),
        "diag_candidate_count": len(diag_rows),
        "selected_count": len(selected_rows),
        "selected_lowloc_count": sum(1 for r in selected_rows if r.get("is_lowloc")),
        "selected_positive_rate": sum(1 for r in selected_rows if float(r.get("target_delta") or 0.0) > 0.0) / max(1, len(selected_rows)),
        "selected_target_delta_mean": float(np.mean([float(r.get("target_delta") or 0.0) for r in selected_rows])) if selected_rows else 0.0,
        "selected_target_delta_sum": float(sum(float(r.get("target_delta") or 0.0) for r in selected_rows)),
        "selected_by_family": dict(Counter(str(((r.get("candidate") or {}).get("family")) or "") for r in selected_rows)),
        "selected_samples": [r.get("sample_id") for r in selected_rows],
        "apply_stats": apply_stats,
        "params": {
            "loc_threshold": args.loc_threshold,
            "min_positive_delta": args.min_positive_delta,
            "k": args.k,
            "neg_weight": args.neg_weight,
            "threshold_mode": args.threshold_mode,
            "action_mode": args.action_mode,
            "family_allowlist": args.family_allowlist,
            "sample_risk_gate": args.sample_risk_gate,
            "sample_risk_reg": args.sample_risk_reg,
            "sample_risk_pos_weight": args.sample_risk_pos_weight,
        },
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
