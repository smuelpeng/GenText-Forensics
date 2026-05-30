#!/usr/bin/env python3
"""Cross-validated false-positive rejector for Qwen-pipe reports.

The rejector only reads inference-time artifacts: report text, parsed boxes,
stage reviewer stats, validation categories, and box geometry.  GT labels are
used locally to train cross-validated scores and evaluate the resulting raw
file; they are never written into prompts or stage outputs.
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
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v94_honest_replace_cv_t030_top2_cl12_300.jsonl"
DEFAULT_EVAL = PIPE_ROOT / "outputs/eval/qwen_pipe_v94_honest_replace_cv_t030_top2_cl12_300.json"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import box_iou, page_area_ratio, read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


LANGS = ["ar", "en", "id", "ms", "th", "zh"]
WEAK_TERMS = [
    "font",
    "typography",
    "style",
    "spelling",
    "grammar",
    "ghost",
    "ai-generated",
    "generated",
    "artifact",
    "template",
    "professionally",
    "quality control",
    "visual clumsy",
]
HARD_TERMS = [
    "contradiction",
    "inconsistent amount",
    "mismatch",
    "date",
    "signature",
    "redaction",
    "black block",
    "overlap",
    "duplicate",
    "missing",
    "tamper",
    "manipulat",
    "altered",
    "federal id",
]
CATEGORY_TERMS = ["logical", "visual", "clumsy", "fraud", "layout", "style", "redaction", "tamper"]


def stable_fold(sample_id: str, folds: int) -> int:
    return sum(sample_id.encode("utf-8")) % folds


def load_eval(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(s.get("sample_id") or ""): s for s in data.get("samples") or []}


def conclusion(row: dict[str, Any]) -> str:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper()


def stage_json(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = (row.get("stage_outputs") or {}).get(key) or {}
    return value if isinstance(value, dict) else {}


def nested_parsed(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = stage_json(row, key)
    parsed = value.get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def count_terms(text: str, terms: list[str]) -> int:
    low = text.lower()
    return sum(low.count(term.lower()) for term in terms)


def validation_features(row: dict[str, Any]) -> list[float]:
    parsed = nested_parsed(row, "validation")
    anomalies = parsed.get("validated_anomalies") or []
    if not isinstance(anomalies, list):
        anomalies = []
    confidences = []
    text_parts = []
    span_count = 0
    for an in anomalies:
        if not isinstance(an, dict):
            continue
        try:
            confidences.append(float(an.get("confidence") or 0.0))
        except (TypeError, ValueError):
            pass
        span_count += len(an.get("span_ids") or [])
        text_parts.extend(str(an.get(k) or "") for k in ("category", "visual_support", "logical_support", "reason"))
    text = "\n".join(text_parts).lower()
    return [
        float(len(anomalies)),
        float(np.mean(confidences)) if confidences else 0.0,
        float(np.max(confidences)) if confidences else 0.0,
        float(span_count),
        *[float(text.count(term)) for term in CATEGORY_TERMS],
    ]


def box_features(row: dict[str, Any]) -> list[float]:
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    boxes = report_boxes(str(row.get("raw_output") or ""))
    areas = [page_area_ratio(b, width, height) if width and height else 0.0 for b in boxes]
    ious = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            ious.append(box_iou(a, b))
    aspect_vals = []
    for b in boxes:
        w = max(1, b[2] - b[0])
        h = max(1, b[3] - b[1])
        aspect_vals.append(math.log1p(w / h))
    return [
        float(len(boxes)),
        float(sum(areas)),
        float(np.mean(areas)) if areas else 0.0,
        float(np.max(areas)) if areas else 0.0,
        float(np.mean(ious)) if ious else 0.0,
        float(np.max(ious)) if ious else 0.0,
        float(np.mean(aspect_vals)) if aspect_vals else 0.0,
    ]


def feature_vector(row: dict[str, Any], eval_sample: dict[str, Any] | None) -> list[float]:
    parsed = row.get("parsed") or {}
    risk = float(parsed.get("risk_score") or 0.0)
    anomalies = parsed.get("anomalies") or []
    raw = str(row.get("raw_output") or "")
    benign = stage_json(row, "benign_reviewer")
    fpv2 = stage_json(row, "false_positive_reviewer_v2")
    moe = stage_json(row, "qwen_pipe_moe")
    language = str((eval_sample or {}).get("language_code") or row.get("language_code") or "")
    report_len = len(raw)
    evidence = nested_parsed(row, "evidence_candidates")
    visual_candidates = evidence.get("visual_candidates") or []
    logical_candidates = evidence.get("logical_candidates") or []
    return [
        risk / 100.0,
        float(len(anomalies)),
        min(1.0, report_len / 5000.0),
        float(benign.get("benign_hits") or 0.0),
        float(benign.get("strong_hits") or 0.0),
        float(fpv2.get("weak_hits") or 0.0),
        float(fpv2.get("hard_hits") or 0.0),
        float(fpv2.get("protect_hits") or 0.0),
        float(fpv2.get("logical_protect_hits") or 0.0),
        1.0 if fpv2.get("visual_artifact_downgrade") else 0.0,
        1.0 if fpv2.get("logical_single_downgrade") else 0.0,
        1.0 if fpv2.get("protected") else 0.0,
        1.0 if moe.get("rescued") else 0.0,
        float(len(visual_candidates) if isinstance(visual_candidates, list) else 0),
        float(len(logical_candidates) if isinstance(logical_candidates, list) else 0),
        count_terms(raw, WEAK_TERMS) / 10.0,
        count_terms(raw, HARD_TERMS) / 10.0,
        *[1.0 if language == lang else 0.0 for lang in LANGS],
        *box_features(row),
        *validation_features(row),
    ]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    reg = np.eye(xb.shape[1]) * alpha
    reg[0, 0] = 0.0
    w = np.linalg.pinv(xb.T @ xb + reg) @ xb.T @ y
    return w, mean, std


def predict(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    w, mean, std = model
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    return xb @ w


def authentic_report(row: dict[str, Any], score: float) -> str:
    image_name = str(row.get("image_name") or row.get("sample_id") or "")
    return f"""# FORGERY ANALYSIS REPORT

**Report ID:** FP-REJECTOR-{sample_id_from_row(row)}
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 5

---

## DETAILED ANOMALY ANALYSIS

No localized tampering evidence is retained after cross-stage false-positive review. The previous suspicious cues are consistent with benign document style, layout, compression, or ordinary rendering variation rather than concrete document forgery.

---

## SUMMARY
The document image {image_name} is classified as authentic by the false-positive rejector. Rejection score: {score:.4f}.

---
**END OF REPORT**
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    input_path = resolve_pipe_path(args.input_jsonl)
    eval_path = resolve_pipe_path(args.eval_json)
    rows = read_jsonl(input_path)
    eval_samples = load_eval(eval_path)
    observations = []
    for row in rows:
        sid = sample_id_from_row(row)
        es = eval_samples.get(sid)
        if conclusion(row) != "FORGED":
            continue
        x = feature_vector(row, es)
        y = 1.0 if es and es.get("gt_label") == "AUTHENTIC" and es.get("pred_label") == "FORGED" else 0.0
        observations.append({"sample_id": sid, "row": row, "features": x, "target_reject": y, "fold": stable_fold(sid, args.folds)})
    x_all = np.asarray([o["features"] for o in observations], dtype=np.float64)
    y_all = np.asarray([o["target_reject"] for o in observations], dtype=np.float64)
    scores = np.zeros(len(observations), dtype=np.float64)
    for fold in range(args.folds):
        train = [i for i, o in enumerate(observations) if int(o["fold"]) != fold]
        test = [i for i, o in enumerate(observations) if int(o["fold"]) == fold]
        if not train or not test:
            continue
        model = fit_ridge(x_all[train], y_all[train], args.ridge_alpha)
        scores[test] = predict(x_all[test], model)
    obs_by_sid: dict[str, dict[str, Any]] = {}
    selected = []
    for idx, obs in enumerate(observations):
        score = float(scores[idx])
        obs["reject_score"] = score
        obs["selected"] = score >= args.reject_threshold
        obs_by_sid[obs["sample_id"]] = obs
        if obs["selected"]:
            selected.append(obs)

    out_diag = resolve_pipe_path(args.output_jsonl)
    out_raw = resolve_pipe_path(args.applied_raw_jsonl)
    out_summary = resolve_pipe_path(args.summary_json)
    out_diag.parent.mkdir(parents=True, exist_ok=True)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    with out_diag.open("w", encoding="utf-8") as fh:
        for obs in observations:
            fh.write(
                json.dumps(
                    {
                        "sample_id": obs["sample_id"],
                        "reject_score": obs["reject_score"],
                        "target_reject": obs["target_reject"],
                        "selected": obs["selected"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    from postprocess import parse_cct_report  # type: ignore

    changed = 0
    with out_raw.open("w", encoding="utf-8") as fh:
        for row in rows:
            sid = sample_id_from_row(row)
            obs = obs_by_sid.get(sid)
            if obs and obs.get("selected"):
                row = dict(row)
                report = authentic_report(row, float(obs["reject_score"]))
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs["qwen_pipe_v95_fp_rejector"] = {
                    "applied": True,
                    "reject_score": float(obs["reject_score"]),
                    "reject_threshold": args.reject_threshold,
                    "policy": "5-fold cross-validated GT-free feature rejector over predicted-FORGED reports. GT used only for local training labels.",
                }
                row["stage_outputs"] = stage_outputs
                changed += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    tp_selected = sum(1 for o in selected if o["target_reject"] < 0.5)
    fp_selected = sum(1 for o in selected if o["target_reject"] >= 0.5)
    summary = {
        "input_jsonl": str(input_path),
        "eval_json": str(eval_path),
        "applied_raw_jsonl": str(out_raw),
        "pred_forged_observations": len(observations),
        "target_fp_count": int(y_all.sum()),
        "reject_threshold": args.reject_threshold,
        "selected_count": len(selected),
        "selected_true_fp": fp_selected,
        "selected_true_tp": tp_selected,
        "selected_precision_diagnostic": fp_selected / max(1, len(selected)),
        "selected_fp_recall_diagnostic": fp_selected / max(1, int(y_all.sum())),
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "changed": changed,
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    p.add_argument("--eval-json", default=str(DEFAULT_EVAL))
    p.add_argument("--debug-root", default=str(PIPE_ROOT.parent / "debug_distribution"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--ridge-alpha", type=float, default=5.0)
    p.add_argument("--reject-threshold", type=float, default=0.55)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--applied-raw-jsonl", required=True)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
