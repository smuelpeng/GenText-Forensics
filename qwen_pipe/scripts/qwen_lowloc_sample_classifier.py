#!/usr/bin/env python3
"""Cross-validated sample-level selector for localization rescue.

The input feature JSONL must be produced by qwen_loc_need_estimator.py.  Features
are inference-available sample/candidate summaries.  Eval labels are used only
to train fold-local sample selectors and to report diagnostics; the held-out
sample never trains on itself.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import box_iou, insert_extra_anomalies, read_jsonl, replace_groundings, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


NUMERIC_KEYS = [
    "existing_count",
    "existing_area_ratio",
    "candidate_count",
    "extra_count",
    "extra_area_ratio",
    "strong_extra_count",
    "uncovered_evidence_count",
    "uncovered_textual_count",
    "compact_extra_count",
    "high_query_extra_count",
    "mean_extra_rank_score",
    "max_extra_rank_score",
    "mean_extra_query_score",
    "max_extra_query_score",
    "mean_extra_novelty",
    "max_extra_novelty",
    "selection_score",
]
COUNT_DICT_KEYS = ["candidate_family_counts", "top_family_counts", "extra_family_counts"]
FAMILY_KEYS = ["evidence", "token", "linegrid", "ocr", "row", "patch", "grid", "paragraph", "existing"]
LANG_KEYS = ["ar", "en", "id", "ms", "th", "zh", ""]


def stable_fold(sample_id: str, folds: int) -> int:
    return sum(sample_id.encode("utf-8")) % folds


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_eval(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(s.get("sample_id") or ""): s for s in data.get("samples") or []}


def label_lowloc(sample: dict[str, Any], threshold: float) -> int:
    return int(
        sample.get("gt_label") == "FORGED"
        and sample.get("pred_label") == "FORGED"
        and float(sample.get("loc_score") or 0.0) < threshold
    )


def label_trainable(sample: dict[str, Any]) -> bool:
    return sample.get("pred_label") == "FORGED"


def vectorize(row: dict[str, Any]) -> list[float]:
    vec: list[float] = []
    for key in NUMERIC_KEYS:
        value = float(row.get(key) or 0.0)
        if key.endswith("_count") or key == "candidate_count":
            value = math.log1p(max(0.0, value))
        vec.append(value)
    for key in COUNT_DICT_KEYS:
        counts = row.get(key) or {}
        total = max(1.0, sum(float(v or 0.0) for v in counts.values()))
        for family in FAMILY_KEYS:
            vec.append(float(counts.get(family) or 0.0) / total)
    lang = str(row.get("language_code") or "")
    for item in LANG_KEYS:
        vec.append(1.0 if lang == item else 0.0)
    extras = row.get("extras") or []
    first = extras[0] if extras else {}
    first_family = str(first.get("family") or "")
    first_source = str(first.get("source") or "")
    first_text = str(first.get("text") or "")
    for family in FAMILY_KEYS:
        vec.append(1.0 if first_family == family else 0.0)
    source_flags = [
        "stage_evidence" in first_source,
        "qwen_ocr" in first_source,
        "linegrid" in first_source,
        "scriptgrid" in first_source,
        first_source.startswith("grid_"),
        bool(re.search(r"\d", first_text)),
        len(first_text) >= 20,
    ]
    vec.extend(1.0 if flag else 0.0 for flag in source_flags)
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


def predict(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    return xb @ coef


def choose_threshold(scores: np.ndarray, y: np.ndarray, mode: str) -> float:
    if len(scores) == 0:
        return 1e9
    candidates = sorted(set(float(v) for v in scores))
    best_thr = candidates[-1] + 1.0
    best_key = (-1.0, -1.0, -1.0)
    beta = {"recall": 2.0, "balanced": 1.0, "precision": 0.5}[mode]
    beta2 = beta * beta
    for thr in candidates:
        pred = scores >= thr
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        fbeta = (1 + beta2) * precision * recall / max(1e-9, beta2 * precision + recall)
        # Prefer smaller selected sets when the quality is tied.
        key = (fbeta, precision, -float(pred.sum()))
        if key > best_key:
            best_key = key
            best_thr = thr
    return best_thr


def apply_rows(
    *,
    base_raw: Path,
    diagnostics: dict[str, dict[str, Any]],
    output_path: Path,
    stage_name: str,
    top_n: int,
    apply_mode: str,
    replace_policy: str,
) -> dict[str, Any]:
    setup_debug_import(DEFAULT_DEBUG_ROOT.resolve())
    from postprocess import parse_cct_report  # type: ignore

    stats = {"rows": 0, "changed": 0, "boxes_added": 0}
    with output_path.open("w", encoding="utf-8") as fh:
        for row in read_jsonl(base_raw):
            stats["rows"] += 1
            sid = sample_id_from_row(row)
            diag = diagnostics.get(sid) or {}
            extras = list(diag.get("extras") or [])[:top_n] if diag.get("selected") else []
            if extras:
                row = dict(row)
                old_report = str(row.get("raw_output") or "")
                if apply_mode == "replace":
                    existing = report_boxes(old_report)
                    replacements: dict[int, list[int]] = {}
                    used: set[int] = set()
                    for extra in extras:
                        box = [int(v) for v in extra.get("box") or []]
                        if len(box) < 4 or not existing:
                            continue
                        available = [idx for idx in range(len(existing)) if idx not in used]
                        if not available:
                            continue
                        if replace_policy == "largest":
                            replace_idx = max(available, key=lambda idx: (existing[idx][2] - existing[idx][0]) * (existing[idx][3] - existing[idx][1]))
                        elif replace_policy == "closest":
                            replace_idx = max(available, key=lambda idx: box_iou(box, existing[idx]))
                        else:
                            replace_idx = min(available, key=lambda idx: box_iou(box, existing[idx]))
                        replacements[replace_idx] = box[:4]
                        used.add(replace_idx)
                    report, changed_count = replace_groundings(old_report, replacements)
                    if changed_count == 0:
                        report = insert_extra_anomalies(old_report, extras)
                else:
                    report = insert_extra_anomalies(old_report, extras)
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs[stage_name] = {
                    "applied": True,
                    "apply_mode": apply_mode,
                    "replace_policy": replace_policy if apply_mode == "replace" else "",
                    "boxes_added": len(extras) if apply_mode == "append" else 0,
                    "boxes_replaced": len(extras) if apply_mode == "replace" else 0,
                    "selector_score": diag.get("selector_score"),
                    "fold": diag.get("fold"),
                    "selected": extras,
                    "policy": "Fold-local ridge classifier predicts low localization need from GT-free candidate/OCR/report features.",
                }
                row["stage_outputs"] = stage_outputs
                stats["changed"] += 1
                stats["boxes_added"] += len(extras)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def summarize(rows: list[dict[str, Any]], eval_samples: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
    selected = [r for r in rows if r.get("selected")]
    selected_ids = {str(r.get("sample_id") or "") for r in selected}
    pred_forged = [s for s in eval_samples.values() if s.get("pred_label") == "FORGED"]
    lowloc = [s for s in pred_forged if label_lowloc(s, threshold)]
    lowloc_ids = {str(s.get("sample_id") or "") for s in lowloc}
    selected_lowloc = sorted(selected_ids & lowloc_ids)
    return {
        "n": len(rows),
        "pred_forged": len(pred_forged),
        "selected_count": len(selected),
        "boxes_to_add": sum(min(1, len(r.get("extras") or [])) for r in selected),
        "lowloc_threshold": threshold,
        "lowloc_count": len(lowloc),
        "selected_lowloc_count": len(selected_lowloc),
        "selected_lowloc_recall": len(selected_lowloc) / max(1, len(lowloc)),
        "selected_precision_vs_lowloc": len(selected_lowloc) / max(1, len(selected)),
        "selected_lowloc_sample_ids": selected_lowloc[:80],
        "selected_non_lowloc_count": len(selected_ids - lowloc_ids),
        "selected_non_lowloc_sample_ids": sorted(selected_ids - lowloc_ids)[:80],
        "family_counts": dict(Counter(str(((r.get("extras") or [{}])[0]).get("family") or "") for r in selected)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-jsonl", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--base-raw-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True, help="Fold-scored feature JSONL")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--applied-raw-jsonl", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lowloc-threshold", type=float, default=0.02)
    parser.add_argument("--mode", choices=["recall", "balanced", "precision"], default="balanced")
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--apply-top-n", type=int, default=1)
    parser.add_argument("--apply-mode", choices=["append", "replace"], default="append")
    parser.add_argument("--replace-policy", choices=["farthest", "closest", "largest"], default="farthest")
    parser.add_argument("--stage-name", default="qwen_pipe_lowloc_sample_classifier")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = resolve_pipe_path(args.features_jsonl)
    eval_path = resolve_pipe_path(args.eval_json)
    base_path = resolve_pipe_path(args.base_raw_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    applied_path = resolve_pipe_path(args.applied_raw_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    applied_path.parent.mkdir(parents=True, exist_ok=True)

    features = read_jsonl_dicts(features_path)
    eval_samples = load_eval(eval_path)
    trainable = [
        row
        for row in features
        if bool(row.get("is_pred_forged")) and label_trainable(eval_samples.get(str(row.get("sample_id") or ""), {}))
    ]
    all_vectors = {str(row.get("sample_id") or ""): vectorize(row) for row in features}
    diagnostics: dict[str, dict[str, Any]] = {}
    for fold in range(args.folds):
        train = [r for r in trainable if stable_fold(str(r.get("sample_id") or ""), args.folds) != fold]
        test = [r for r in trainable if stable_fold(str(r.get("sample_id") or ""), args.folds) == fold]
        if not train or not test:
            continue
        x_train = np.asarray([all_vectors[str(r.get("sample_id") or "")] for r in train], dtype=float)
        y_train = np.asarray(
            [label_lowloc(eval_samples.get(str(r.get("sample_id") or ""), {}), args.lowloc_threshold) for r in train],
            dtype=float,
        )
        x_test = np.asarray([all_vectors[str(r.get("sample_id") or "")] for r in test], dtype=float)
        coef, mean, std = fit_ridge_classifier(x_train, y_train, reg=args.reg, pos_weight=args.pos_weight)
        train_scores = predict(x_train, coef, mean, std)
        test_scores = predict(x_test, coef, mean, std)
        threshold = choose_threshold(train_scores, y_train, args.mode)
        for row, score in zip(test, test_scores, strict=True):
            out = dict(row)
            out["fold"] = fold
            out["selector_score"] = float(score)
            out["selector_threshold"] = float(threshold)
            out["selected"] = bool(score >= threshold)
            diagnostics[str(row.get("sample_id") or "")] = out

    with output_path.open("w", encoding="utf-8") as fh:
        for row in features:
            sid = str(row.get("sample_id") or "")
            out = diagnostics.get(sid) or dict(row, selected=False, selector_score=None, selector_threshold=None)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    scored_rows = [diagnostics.get(str(row.get("sample_id") or "")) or dict(row, selected=False) for row in features]
    summary = summarize(scored_rows, eval_samples, args.lowloc_threshold)
    summary.update(
        {
            "features_jsonl": str(features_path),
            "eval_json": str(eval_path),
            "base_raw_jsonl": str(base_path),
            "output_jsonl": str(output_path),
            "applied_raw_jsonl": str(applied_path),
            "folds": args.folds,
            "mode": args.mode,
            "reg": args.reg,
            "pos_weight": args.pos_weight,
            "apply_mode": args.apply_mode,
            "replace_policy": args.replace_policy,
        }
    )
    summary["apply_stats"] = apply_rows(
        base_raw=base_path,
        diagnostics=diagnostics,
        output_path=applied_path,
        stage_name=args.stage_name,
        top_n=args.apply_top_n,
        apply_mode=args.apply_mode,
        replace_policy=args.replace_policy,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
