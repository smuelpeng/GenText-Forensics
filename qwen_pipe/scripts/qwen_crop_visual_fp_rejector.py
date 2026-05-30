#!/usr/bin/env python3
"""Cross-validated FP rejector with local crop visual evidence features.

This extends the older report-feature rejector with GT-free image crop
statistics for every predicted grounding box.  The hypothesis is that many
remaining false positives describe generic rendering/style issues whose boxes
do not contain strong local visual evidence compared with their surrounding
context.

GT labels are used only as local cross-validation labels after predictions
exist.  They are never used in prompts or per-sample feature extraction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import page_area_ratio, read_jsonl, sample_id_from_row  # noqa: E402
from qwen_fp_rejector import feature_vector as report_feature_vector  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


def stable_fold(sample_id: str, folds: int) -> int:
    return sum(sample_id.encode("utf-8")) % folds


def load_eval(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(s.get("sample_id") or ""): s for s in data.get("samples") or []}


def conclusion(row: dict[str, Any]) -> str:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper()


def image_path(row: dict[str, Any], debug_root: Path) -> Path:
    raw = Path(str(row.get("image_path") or ""))
    if raw.is_absolute():
        return raw
    candidate = debug_root / raw
    if candidate.exists():
        return candidate
    return debug_root / "data" / "images" / str(row.get("image_name") or "")


def clamp_box(box: list[int], width: int, height: int, pad: int = 0) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1 = max(0, min(width, x1 - pad))
    y1 = max(0, min(height, y1 - pad))
    x2 = max(0, min(width, x2 + pad))
    y2 = max(0, min(height, y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def gray_array(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def patch_stats(gray: np.ndarray) -> dict[str, float]:
    if gray.size <= 1:
        return {"mean": 0.0, "std": 0.0, "edge": 0.0, "dark": 0.0, "bright": 0.0}
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge = float((gx > 22).mean() + (gy > 22).mean()) / 2.0
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "edge": edge,
        "dark": float((gray < 64).mean()),
        "bright": float((gray > 224).mean()),
    }


def crop_visual_features(row: dict[str, Any], debug_root: Path) -> list[float]:
    boxes = report_boxes(str(row.get("raw_output") or ""))
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    if not boxes or width <= 0 or height <= 0:
        return [0.0] * 28
    path = image_path(row, debug_root)
    try:
        img = Image.open(path)
        gray = gray_array(img)
    except Exception:
        return [0.0] * 28
    h_img, w_img = gray.shape[:2]
    if width != w_img or height != h_img:
        width, height = w_img, h_img

    areas: list[float] = []
    stds: list[float] = []
    edges: list[float] = []
    darks: list[float] = []
    brights: list[float] = []
    std_ratios: list[float] = []
    edge_ratios: list[float] = []
    weak_boxes = 0
    strong_boxes = 0
    page_edge_boxes = 0
    for box in boxes:
        clipped = clamp_box(box, width, height)
        if not clipped:
            continue
        x1, y1, x2, y2 = clipped
        bw = x2 - x1
        bh = y2 - y1
        pad = int(max(10, 0.6 * max(bw, bh)))
        context = clamp_box(box, width, height, pad=pad) or clipped
        crop = gray[y1:y2, x1:x2]
        cx1, cy1, cx2, cy2 = context
        ctx = gray[cy1:cy2, cx1:cx2]
        cs = patch_stats(crop)
        xs = patch_stats(ctx)
        area = page_area_ratio([x1, y1, x2, y2], width, height)
        areas.append(area)
        stds.append(cs["std"])
        edges.append(cs["edge"])
        darks.append(cs["dark"])
        brights.append(cs["bright"])
        std_ratio = cs["std"] / max(1.0, xs["std"])
        edge_ratio = cs["edge"] / max(0.005, xs["edge"])
        std_ratios.append(std_ratio)
        edge_ratios.append(edge_ratio)
        y_center = (y1 + y2) / max(1.0, 2.0 * height)
        if y_center < 0.06 or y_center > 0.94:
            page_edge_boxes += 1
        if cs["std"] < 18 and cs["edge"] < 0.055 and cs["dark"] < 0.08:
            weak_boxes += 1
        if cs["std"] > 38 or cs["edge"] > 0.14 or cs["dark"] > 0.18 or std_ratio > 1.45 or edge_ratio > 1.55:
            strong_boxes += 1

    def mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    def mx(vals: list[float]) -> float:
        return float(np.max(vals)) if vals else 0.0

    def mn(vals: list[float]) -> float:
        return float(np.min(vals)) if vals else 0.0

    n = max(1, len(areas))
    return [
        float(len(areas)),
        mean(areas),
        mx(areas),
        float(sum(areas)),
        mean(stds),
        mx(stds),
        mn(stds),
        mean(edges),
        mx(edges),
        mn(edges),
        mean(darks),
        mx(darks),
        mean(brights),
        mx(brights),
        mean(std_ratios),
        mx(std_ratios),
        mn(std_ratios),
        mean(edge_ratios),
        mx(edge_ratios),
        mn(edge_ratios),
        weak_boxes / n,
        strong_boxes / n,
        page_edge_boxes / n,
        float(weak_boxes),
        float(strong_boxes),
        float(page_edge_boxes),
        1.0 if strong_boxes == 0 else 0.0,
        1.0 if weak_boxes == len(areas) else 0.0,
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
    sid = sample_id_from_row(row)
    image_name = str(row.get("image_name") or sid)
    return f"""# FORGERY ANALYSIS REPORT

**Report ID:** CROP-VISUAL-FP-REJECT-{sid}
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 5

---

## DETAILED ANOMALY ANALYSIS

No localized tampering evidence is retained after crop-level visual evidence review. The predicted regions do not provide sufficiently distinctive local visual evidence relative to their surrounding context.

---

## SUMMARY
The document image {image_name} is classified as authentic by the crop visual false-positive rejector. Rejection score: {score:.4f}.

---
**END OF REPORT**
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--applied-raw-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=8.0)
    parser.add_argument("--reject-threshold", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    eval_path = resolve_pipe_path(args.eval_json)
    out_diag = resolve_pipe_path(args.output_jsonl)
    out_summary = resolve_pipe_path(args.summary_json)
    out_raw = resolve_pipe_path(args.applied_raw_jsonl)
    out_diag.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    eval_samples = load_eval(eval_path)
    observations: list[dict[str, Any]] = []
    for row in rows:
        sid = sample_id_from_row(row)
        es = eval_samples.get(sid)
        if conclusion(row) != "FORGED":
            continue
        feats = report_feature_vector(row, es) + crop_visual_features(row, debug_root)
        observations.append(
            {
                "sample_id": sid,
                "row": row,
                "features": feats,
                "target_reject": 1.0 if es and es.get("gt_label") == "AUTHENTIC" and es.get("pred_label") == "FORGED" else 0.0,
                "fold": stable_fold(sid, args.folds),
            }
        )

    x_all = np.asarray([o["features"] for o in observations], dtype=np.float64)
    y_all = np.asarray([o["target_reject"] for o in observations], dtype=np.float64)
    scores = np.zeros(len(observations), dtype=np.float64)
    for fold in range(args.folds):
        train = [i for i, o in enumerate(observations) if int(o["fold"]) != fold]
        test = [i for i, o in enumerate(observations) if int(o["fold"]) == fold]
        if train and test:
            model = fit_ridge(x_all[train], y_all[train], args.ridge_alpha)
            scores[test] = predict(x_all[test], model)

    selected = []
    obs_by_sid: dict[str, dict[str, Any]] = {}
    with out_diag.open("w", encoding="utf-8") as fh:
        for idx, obs in enumerate(observations):
            score = float(scores[idx])
            obs["reject_score"] = score
            obs["selected"] = score >= args.reject_threshold
            obs_by_sid[obs["sample_id"]] = obs
            if obs["selected"]:
                selected.append(obs)
            fh.write(
                json.dumps(
                    {
                        "sample_id": obs["sample_id"],
                        "reject_score": score,
                        "target_reject": obs["target_reject"],
                        "selected": obs["selected"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

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
                stage_outputs["qwen_crop_visual_fp_rejector"] = {
                    "applied": True,
                    "reject_score": float(obs["reject_score"]),
                    "reject_threshold": args.reject_threshold,
                    "policy": "5-fold crop visual false-positive rejector; GT used only for local fold labels.",
                }
                row["stage_outputs"] = stage_outputs
                changed += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_true_fp = sum(1 for o in selected if o["target_reject"] >= 0.5)
    selected_true_tp = sum(1 for o in selected if o["target_reject"] < 0.5)
    target_fp_count = int(y_all.sum())
    summary = {
        "input_jsonl": str(input_path),
        "eval_json": str(eval_path),
        "applied_raw_jsonl": str(out_raw),
        "pred_forged_observations": len(observations),
        "target_fp_count": target_fp_count,
        "reject_threshold": args.reject_threshold,
        "ridge_alpha": args.ridge_alpha,
        "selected_count": len(selected),
        "selected_true_fp": selected_true_fp,
        "selected_true_tp": selected_true_tp,
        "selected_precision_diagnostic": selected_true_fp / max(1, len(selected)),
        "selected_fp_recall_diagnostic": selected_true_fp / max(1, target_fp_count),
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "changed": changed,
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
