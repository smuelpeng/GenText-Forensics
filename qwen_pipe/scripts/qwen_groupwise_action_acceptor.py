#!/usr/bin/env python3
"""Fold-local accept/reject layer for groupwise localization replacements.

The groupwise ranker improves localization recall, but it still applies many
negative replacement actions.  This script treats the ranker's top action as a
candidate proposal and learns a second, fold-local acceptor.  Training labels
come from local evaluation deltas in non-heldout folds only; inference uses only
GT-free action geometry, OCR metadata, report relation features, and groupwise
scores already present in the diagnostic JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_candidate_delta_model import replace_groundings, replacement_risk_features, stable_fold  # noqa: E402
from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


FAMILIES = ["ocr", "linegrid", "evidence", "row", "token", "patch", "grid", "paragraph", "existing"]
LANGS = ["ar", "en", "id", "ms", "th", "zh", ""]


def read_diag(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_eval_samples(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(s.get("sample_id") or ""): s for s in data.get("samples") or []}


def language_code(row: dict[str, Any], eval_samples: dict[str, dict[str, Any]], sid: str) -> str:
    if sid in eval_samples:
        return str(eval_samples[sid].get("language") or eval_samples[sid].get("language_code") or "")
    return str(row.get("language_code") or (row.get("metadata") or {}).get("language_code") or "")


def box_area(box: list[int]) -> float:
    if len(box) < 4:
        return 0.0
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def candidate_meta_features(candidate: dict[str, Any], row: dict[str, Any], lang: str) -> list[float]:
    box = [int(v) for v in candidate.get("box") or [0, 0, 0, 0]][:4]
    width = float(row.get("width") or row.get("image_width") or 1)
    height = float(row.get("height") or row.get("image_height") or 1)
    text = str(candidate.get("text") or "")
    source = str(candidate.get("source") or "")
    family = str(candidate.get("family") or "")
    meta = candidate.get("meta") or {}
    rank = meta.get("v87b_rank") or meta.get("v87_rank") or {}
    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    query_hits = meta.get("query_hits") or []
    number_hits = meta.get("number_hits") or []
    visual_hits = meta.get("visual_hits") or []
    start_frac = float(meta.get("start_frac") if meta.get("start_frac") is not None else meta.get("physical_start_frac") or 0.0)
    end_frac = float(meta.get("end_frac") if meta.get("end_frac") is not None else meta.get("physical_end_frac") or 0.0)
    return [
        *[1.0 if family == fam else 0.0 for fam in FAMILIES],
        *[1.0 if lang == item else 0.0 for item in LANGS],
        box_area(box) / max(1.0, width * height),
        (box[2] - box[0]) / max(1.0, width) if len(box) == 4 else 0.0,
        (box[3] - box[1]) / max(1.0, height) if len(box) == 4 else 0.0,
        float(candidate.get("score") or 0.0),
        float(rank.get("score") or 0.0),
        float(rank.get("query_score") or 0.0),
        float(meta.get("match_score") or 0.0),
        float(len(query_hits)),
        float(len(number_hits)),
        float(len(visual_hits)),
        1.0 if "span_id" in source else 0.0,
        1.0 if source.startswith("ocr_span") else 0.0,
        1.0 if source.startswith(("ocr_linegrid", "ocr_scriptgrid")) else 0.0,
        1.0 if "expanded" in source else 0.0,
        min(1.0, len(text) / 240.0),
        min(1.0, digits / 12.0),
        digits / max(1.0, digits + letters),
        1.0 if re.search(r"(\\d|\\$|%|年|月|日|บาท|rm|rp)", text.lower()) else 0.0,
        start_frac,
        end_frac,
        max(0.0, end_frac - start_frac),
        min(start_frac, max(0.0, 1.0 - end_frac)),
    ]


def vectorize(action: dict[str, Any], sample_rows: dict[str, dict[str, Any]], eval_samples: dict[str, dict[str, Any]]) -> list[float]:
    sid = str(action.get("sample_id") or "")
    row = sample_rows.get(sid) or {}
    selected = action.get("selected") or {}
    candidate = selected.get("candidate") or {}
    risk_obs = {
        "sample_id": sid,
        "candidate": candidate,
        "replace_index": selected.get("replace_index"),
    }
    lang = language_code(row, eval_samples, sid)
    return [
        float(selected.get("groupwise_score") or 0.0),
        float(selected.get("rank_margin") or 0.0),
        float(selected.get("replace_index") if selected.get("replace_index") is not None else -1),
        *candidate_meta_features(candidate, row, lang),
        *replacement_risk_features(risk_obs, sample_rows),
    ]


def fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    reg = alpha * np.eye(xb.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    return coef, mean, std


def predict_ridge(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    coef, mean, std = model
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    return xb @ coef


def choose_threshold(scores: np.ndarray, deltas: np.ndarray, *, min_accept: int) -> float:
    if len(scores) == 0:
        return 1e9
    candidates = sorted(set(float(s) for s in scores))
    best_thr = candidates[-1] + 1.0
    best_key = (-1e9, -1e9, -1e9)
    for thr in candidates:
        keep = scores >= thr
        count = int(keep.sum())
        if count < min_accept:
            continue
        delta_sum = float(deltas[keep].sum())
        pos_rate = float((deltas[keep] > 0).mean()) if count else 0.0
        key = (delta_sum, pos_rate, -count)
        if key > best_key:
            best_key = key
            best_thr = thr
    return best_thr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-jsonl", required=True)
    parser.add_argument("--base-jsonl", required=True)
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--applied-raw-jsonl", required=True)
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--target", choices=["delta", "positive"], default="delta")
    parser.add_argument("--min-train-accept", type=int, default=8)
    args = parser.parse_args()

    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    diag_rows = read_diag(resolve_pipe_path(args.diag_jsonl))
    base_rows = read_jsonl(resolve_pipe_path(args.base_jsonl))
    sample_rows = {sample_id_from_row(row): row for row in base_rows}
    eval_samples = load_eval_samples(resolve_pipe_path(args.eval_json) if args.eval_json else None)

    actions: list[dict[str, Any]] = []
    for row in diag_rows:
        sid = str(row.get("sample_id") or "")
        selected = list(row.get("selected") or [])
        if not selected:
            continue
        first = dict(selected[0])
        actions.append(
            {
                "sample_id": sid,
                "fold": stable_fold(sid, args.folds),
                "selected": first,
                "delta": float(first.get("target_delta") or 0.0),
                "features": vectorize({"sample_id": sid, "selected": first}, sample_rows, eval_samples),
            }
        )

    accepted_by_sid: dict[str, dict[str, Any]] = {}
    fold_counts: dict[int, dict[str, Any]] = {}
    for fold in range(args.folds):
        train = [a for a in actions if int(a["fold"]) != fold]
        test = [a for a in actions if int(a["fold"]) == fold]
        if not train or not test:
            continue
        x_train = np.asarray([a["features"] for a in train], dtype=np.float64)
        deltas_train = np.asarray([float(a["delta"]) for a in train], dtype=np.float64)
        y_train = deltas_train if args.target == "delta" else (deltas_train > 0).astype(np.float64)
        model = fit_ridge(x_train, y_train, alpha=args.alpha)
        train_scores = predict_ridge(x_train, model)
        threshold = choose_threshold(train_scores, deltas_train, min_accept=args.min_train_accept)
        x_test = np.asarray([a["features"] for a in test], dtype=np.float64)
        test_scores = predict_ridge(x_test, model)
        accepted = 0
        for action, score in zip(test, test_scores, strict=True):
            action = dict(action)
            action["acceptor_score"] = float(score)
            action["acceptor_threshold"] = float(threshold)
            action["accepted"] = bool(score >= threshold)
            if action["accepted"]:
                accepted_by_sid[str(action["sample_id"])] = action
                accepted += 1
        fold_counts[fold] = {"train": len(train), "test": len(test), "accepted": accepted, "threshold": float(threshold)}

    decision_path = resolve_pipe_path(args.decision_jsonl)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    with decision_path.open("w", encoding="utf-8") as fh:
        for action in actions:
            sid = str(action["sample_id"])
            out = dict(action)
            accepted = accepted_by_sid.get(sid)
            out["accepted"] = bool(accepted)
            if accepted:
                out["acceptor_score"] = accepted.get("acceptor_score")
                out["acceptor_threshold"] = accepted.get("acceptor_threshold")
            out.pop("features", None)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    raw_path = resolve_pipe_path(args.applied_raw_jsonl)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as fh:
        for row in base_rows:
            sid = sample_id_from_row(row)
            action = accepted_by_sid.get(sid)
            if action:
                selected = action.get("selected") or {}
                candidate = selected.get("candidate") or {}
                replace_index = int(selected.get("replace_index") if selected.get("replace_index") is not None else -1)
                if replace_index >= 0:
                    row = dict(row)
                    replacements = {replace_index: [int(v) for v in candidate.get("box") or []][:4]}
                    report, replaced_count = replace_groundings(str(row.get("raw_output") or ""), replacements)
                    row["raw_output"] = report
                    row["parsed"] = parse_cct_report(report)
                    stage_outputs = dict(row.get("stage_outputs") or {})
                    stage_outputs["qwen_pipe_groupwise_action_acceptor"] = {
                        "applied": True,
                        "selected_actions": 1,
                        "replaced_count": replaced_count,
                        "acceptor_score": action.get("acceptor_score"),
                        "acceptor_threshold": action.get("acceptor_threshold"),
                        "target_delta_local_diagnostic": action.get("delta"),
                        "policy": "Fold-local acceptor filters groupwise OCR/linegrid replacement proposals using GT-free action/report geometry features.",
                        "selected": selected,
                    }
                    row["stage_outputs"] = stage_outputs
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    accepted_actions = list(accepted_by_sid.values())
    deltas = [float(a.get("delta") or 0.0) for a in accepted_actions]
    summary = {
        "diag_jsonl": str(resolve_pipe_path(args.diag_jsonl)),
        "base_jsonl": str(resolve_pipe_path(args.base_jsonl)),
        "applied_raw_jsonl": str(raw_path),
        "decision_jsonl": str(decision_path),
        "actions_in": len(actions),
        "accepted_actions": len(accepted_actions),
        "accepted_delta_sum": float(np.sum(deltas)) if deltas else 0.0,
        "accepted_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "accepted_positive_rate": float(np.mean([d > 0 for d in deltas])) if deltas else 0.0,
        "fold_counts": fold_counts,
        "params": {
            "folds": args.folds,
            "alpha": args.alpha,
            "target": args.target,
            "min_train_accept": args.min_train_accept,
        },
        "diagnostic_note": "GT deltas are used only to train/calibrate non-heldout folds and summarize accepted actions.",
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
