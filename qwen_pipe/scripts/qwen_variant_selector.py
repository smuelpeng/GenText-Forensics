#!/usr/bin/env python3
"""Cross-validated per-sample selector between two Qwen-pipe raw variants.

The selector learns when an alternative postprocess variant is likely to help.
GT/eval scores are used only to train cross-validated labels.  Inference
features come from generated reports, parsed fields, stage metadata, language,
and box geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import box_iou, page_area_ratio, read_jsonl, sample_id_from_row  # noqa: E402
from qwen_pair_replace_model import fit_ridge, predict_ridge, stable_fold  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path  # noqa: E402


LANGS = ["ar", "en", "id", "ms", "th", "zh"]


def load_eval(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(s.get("sample_id") or ""): s for s in data.get("samples") or []}


def sample_proxy(sample: dict[str, Any]) -> float:
    det = float(sample.get("det_correct") or 0.0)
    loc = float(sample.get("loc_score") or 0.0)
    exp = float(sample.get("exp_score") or 0.0)
    rep = float(sample.get("rep_score") or 0.0)
    return 0.30 * det + 0.20 * loc + 0.15 * exp + 0.35 * rep


def metric_value(sample: dict[str, Any], metric: str) -> float:
    if metric == "fin":
        return sample_proxy(sample)
    keys = {
        "loc": "loc_score",
        "rep": "rep_score",
        "exp": "exp_score",
        "det": "det_correct",
    }
    return float(sample.get(keys[metric]) or 0.0)


def selection_score(preds: dict[str, np.ndarray], idx: int, args: argparse.Namespace) -> float:
    if args.objective == "fin":
        return float(preds["fin"][idx])
    if args.objective == "loc":
        return float(preds["loc"][idx])
    rep_loss = max(0.0, -float(preds["rep"][idx]))
    exp_loss = max(0.0, -float(preds["exp"][idx]))
    return float(preds["loc"][idx]) - float(args.rep_loss_weight) * rep_loss - float(args.exp_loss_weight) * exp_loss


def box_stats(row: dict[str, Any]) -> list[float]:
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    boxes = report_boxes(str(row.get("raw_output") or ""))
    areas = [page_area_ratio(b, width, height) if width and height else 0.0 for b in boxes]
    ious = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            ious.append(box_iou(a, b))
    aspects = []
    for b in boxes:
        bw = max(1, b[2] - b[0])
        bh = max(1, b[3] - b[1])
        aspects.append(math.log1p(bw / bh))
    return [
        float(len(boxes)),
        float(sum(areas)),
        float(np.mean(areas)) if areas else 0.0,
        float(np.max(areas)) if areas else 0.0,
        float(np.mean(ious)) if ious else 0.0,
        float(np.max(ious)) if ious else 0.0,
        float(np.mean(aspects)) if aspects else 0.0,
    ]


def stage_features(row: dict[str, Any]) -> list[float]:
    stage = (row.get("stage_outputs") or {}).get("qwen_pipe_pair_replace_model") or {}
    selected = stage.get("selected") or []
    if not isinstance(selected, list):
        selected = []
    pred_delta = []
    pred_pos = []
    pred_gain = []
    scores = []
    family_counts = {k: 0.0 for k in ["token", "evidence", "patch", "ocr", "linegrid", "row", "grid"]}
    replace_count = 0.0
    append_count = 0.0
    for item in selected:
        if not isinstance(item, dict):
            continue
        cand = item.get("candidate") or {}
        family = str(cand.get("family") or "")
        if family in family_counts:
            family_counts[family] += 1.0
        if str(item.get("action") or "") == "replace":
            replace_count += 1.0
        if str(item.get("action") or "") == "append":
            append_count += 1.0
        for arr, key in [(pred_delta, "pred_delta"), (pred_pos, "pred_pos"), (pred_gain, "pred_gain"), (scores, "selection_score")]:
            try:
                arr.append(float(item.get(key) or 0.0))
            except (TypeError, ValueError):
                pass
    recall_stage = (row.get("stage_outputs") or {}).get("qwen_pipe_exhaustive_recall_apply") or {}
    recall_selected = recall_stage.get("selected") or []
    if not isinstance(recall_selected, list):
        recall_selected = []
    recall_family_counts = {k: 0.0 for k in ["token", "evidence", "patch", "ocr", "linegrid", "row", "grid", "paragraph"]}
    recall_source_counts = {
        "stage_evidence": 0.0,
        "ocr_token": 0.0,
        "ocr_linegrid": 0.0,
        "ocr_span": 0.0,
        "grid": 0.0,
    }
    recall_areas = []
    recall_widths = []
    recall_heights = []
    for item in recall_selected:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family") or "")
        source = str(item.get("source") or "")
        if family in recall_family_counts:
            recall_family_counts[family] += 1.0
        if source.startswith("stage_evidence"):
            recall_source_counts["stage_evidence"] += 1.0
        if source.startswith("ocr_token"):
            recall_source_counts["ocr_token"] += 1.0
        if source.startswith("ocr_linegrid"):
            recall_source_counts["ocr_linegrid"] += 1.0
        if source.startswith("ocr_span"):
            recall_source_counts["ocr_span"] += 1.0
        if family == "grid":
            recall_source_counts["grid"] += 1.0
        box = item.get("box")
        width = int(row.get("width") or 0)
        height = int(row.get("height") or 0)
        if isinstance(box, list) and len(box) >= 4 and width and height:
            try:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                recall_areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, width * height))
                recall_widths.append(max(0.0, x2 - x1) / max(1.0, width))
                recall_heights.append(max(0.0, y2 - y1) / max(1.0, height))
            except (TypeError, ValueError):
                pass
    relation_stage = (row.get("stage_outputs") or {}).get("qwen_pipe_v91_candidate_delta_model") or {}
    relation_selected = relation_stage.get("selected") or []
    if not isinstance(relation_selected, list):
        relation_selected = []
    relation_family_counts = {k: 0.0 for k in ["token", "evidence", "patch", "ocr", "linegrid", "row", "grid", "paragraph"]}
    relation_source_counts = {
        "stage_evidence": 0.0,
        "ocr_token": 0.0,
        "ocr_linegrid": 0.0,
        "ocr_span": 0.0,
        "ocr_row": 0.0,
        "patch": 0.0,
    }
    relation_category_counts = {
        "semantic": 0.0,
        "render": 0.0,
        "layout": 0.0,
        "style": 0.0,
        "visual": 0.0,
    }
    relation_pred_delta = []
    relation_risk = []
    relation_areas = []
    relation_text_lens = []
    relation_conf = []
    for item in relation_selected:
        if not isinstance(item, dict):
            continue
        cand = item.get("candidate") or {}
        family = str(cand.get("family") or "")
        source = str(cand.get("source") or "")
        meta = cand.get("meta") or {}
        category = str(meta.get("category") or "").lower()
        if family in relation_family_counts:
            relation_family_counts[family] += 1.0
        if source.startswith("stage_evidence"):
            relation_source_counts["stage_evidence"] += 1.0
        if source.startswith("ocr_token"):
            relation_source_counts["ocr_token"] += 1.0
        if source.startswith("ocr_linegrid"):
            relation_source_counts["ocr_linegrid"] += 1.0
        if source.startswith("ocr_span"):
            relation_source_counts["ocr_span"] += 1.0
        if source.startswith("ocr_row"):
            relation_source_counts["ocr_row"] += 1.0
        if source.startswith("patch"):
            relation_source_counts["patch"] += 1.0
        for key in relation_category_counts:
            if key in category:
                relation_category_counts[key] += 1.0
        for arr, key in [(relation_pred_delta, "pred_delta"), (relation_risk, "risk_score")]:
            try:
                arr.append(float(item.get(key) or 0.0))
            except (TypeError, ValueError):
                pass
        try:
            relation_conf.append(float(meta.get("confidence") or 0.0))
        except (TypeError, ValueError):
            pass
        text = str(cand.get("text") or "")
        relation_text_lens.append(min(1.0, len(text) / 800.0))
        box = cand.get("box")
        width = int(row.get("width") or 0)
        height = int(row.get("height") or 0)
        if isinstance(box, list) and len(box) >= 4 and width and height:
            try:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                relation_areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, width * height))
            except (TypeError, ValueError):
                pass
    return [
        float(stage.get("selected_actions") or len(selected)),
        float(stage.get("replaced_count") or replace_count),
        float(stage.get("appended_count") or append_count),
        float(np.mean(pred_delta)) if pred_delta else 0.0,
        float(np.max(pred_delta)) if pred_delta else 0.0,
        float(np.mean(pred_pos)) if pred_pos else 0.0,
        float(np.max(pred_pos)) if pred_pos else 0.0,
        float(np.mean(pred_gain)) if pred_gain else 0.0,
        float(np.max(pred_gain)) if pred_gain else 0.0,
        float(np.mean(scores)) if scores else 0.0,
        float(np.max(scores)) if scores else 0.0,
        *[family_counts[k] for k in ["token", "evidence", "patch", "ocr", "linegrid", "row", "grid"]],
        float(recall_stage.get("boxes_added") or len(recall_selected)),
        *[recall_family_counts[k] for k in ["token", "evidence", "patch", "ocr", "linegrid", "row", "grid", "paragraph"]],
        *[recall_source_counts[k] for k in ["stage_evidence", "ocr_token", "ocr_linegrid", "ocr_span", "grid"]],
        float(np.mean(recall_areas)) if recall_areas else 0.0,
        float(np.max(recall_areas)) if recall_areas else 0.0,
        float(np.mean(recall_widths)) if recall_widths else 0.0,
        float(np.mean(recall_heights)) if recall_heights else 0.0,
        float(relation_stage.get("selected_candidates") or len(relation_selected)),
        float(relation_stage.get("risk_rejected") or 0.0),
        float(relation_stage.get("enable_report_relation_features") or False),
        float(np.mean(relation_pred_delta)) if relation_pred_delta else 0.0,
        float(np.max(relation_pred_delta)) if relation_pred_delta else 0.0,
        float(np.mean(relation_risk)) if relation_risk else 0.0,
        float(np.max(relation_risk)) if relation_risk else 0.0,
        float(np.mean(relation_areas)) if relation_areas else 0.0,
        float(np.max(relation_areas)) if relation_areas else 0.0,
        float(np.mean(relation_text_lens)) if relation_text_lens else 0.0,
        float(np.mean(relation_conf)) if relation_conf else 0.0,
        *[relation_family_counts[k] for k in ["token", "evidence", "patch", "ocr", "linegrid", "row", "grid", "paragraph"]],
        *[relation_source_counts[k] for k in ["stage_evidence", "ocr_token", "ocr_linegrid", "ocr_span", "ocr_row", "patch"]],
        *[relation_category_counts[k] for k in ["semantic", "render", "layout", "style", "visual"]],
    ]


def row_features(row: dict[str, Any], language: str) -> list[float]:
    parsed = row.get("parsed") or {}
    anomalies = parsed.get("anomalies") or []
    raw = str(row.get("raw_output") or "")
    return [
        float(str(parsed.get("conclusion") or "").upper() == "FORGED"),
        float(parsed.get("risk_score") or 0.0) / 100.0,
        min(1.0, len(raw) / 6000.0),
        float(len(anomalies) if isinstance(anomalies, list) else 0),
        *box_stats(row),
        *stage_features(row),
        *[1.0 if language == lang else 0.0 for lang in LANGS],
    ]


def paired_features(base: dict[str, Any], alt: dict[str, Any], language: str) -> list[float]:
    fa = row_features(base, language)
    fb = row_features(alt, language)
    return fa + fb + [b - a for a, b in zip(fa, fb)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-raw", required=True)
    parser.add_argument("--base-eval", required=True)
    parser.add_argument("--alt-raw", required=True)
    parser.add_argument("--alt-eval", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=3.0)
    parser.add_argument("--select-threshold", type=float, default=0.0)
    parser.add_argument("--objective", choices=["fin", "loc", "loc_rep_guard"], default="fin")
    parser.add_argument("--rep-loss-weight", type=float, default=1.0)
    parser.add_argument("--exp-loss-weight", type=float, default=0.5)
    parser.add_argument("--min-pred-loc-delta", type=float, default=-1e9)
    parser.add_argument("--min-pred-rep-delta", type=float, default=-1e9)
    parser.add_argument("--min-pred-exp-delta", type=float, default=-1e9)
    parser.add_argument("--min-pred-fin-delta", type=float, default=-1e9)
    args = parser.parse_args()

    base_rows = {sample_id_from_row(r): r for r in read_jsonl(resolve_pipe_path(args.base_raw))}
    alt_rows = {sample_id_from_row(r): r for r in read_jsonl(resolve_pipe_path(args.alt_raw))}
    base_eval = load_eval(resolve_pipe_path(args.base_eval))
    alt_eval = load_eval(resolve_pipe_path(args.alt_eval))
    sample_ids = sorted(set(base_rows) & set(alt_rows) & set(base_eval) & set(alt_eval))
    features = []
    targets: dict[str, list[float]] = {key: [] for key in ["fin", "loc", "rep", "exp", "det"]}
    for sid in sample_ids:
        language = str(base_eval[sid].get("language_code") or base_rows[sid].get("language_code") or "")
        features.append(paired_features(base_rows[sid], alt_rows[sid], language))
        for metric in targets:
            targets[metric].append(metric_value(alt_eval[sid], metric) - metric_value(base_eval[sid], metric))

    x = np.asarray(features, dtype=np.float64)
    y_by_metric = {metric: np.asarray(values, dtype=np.float64) for metric, values in targets.items()}
    pred_by_metric = {metric: np.zeros(len(sample_ids), dtype=np.float64) for metric in targets}
    for fold in range(args.folds):
        train = [i for i, sid in enumerate(sample_ids) if stable_fold(sid, args.folds) != fold]
        test = [i for i, sid in enumerate(sample_ids) if stable_fold(sid, args.folds) == fold]
        if not train or not test:
            continue
        for metric, y in y_by_metric.items():
            model = fit_ridge(x[train], y[train], args.ridge_alpha)
            pred_by_metric[metric][test] = predict_ridge(x[test], model)

    chosen_alt = set()
    selector_pred = np.zeros(len(sample_ids), dtype=np.float64)
    for idx, sid in enumerate(sample_ids):
        score = selection_score(pred_by_metric, idx, args)
        selector_pred[idx] = score
        if float(score) <= args.select_threshold:
            continue
        if float(pred_by_metric["loc"][idx]) < float(args.min_pred_loc_delta):
            continue
        if float(pred_by_metric["rep"][idx]) < float(args.min_pred_rep_delta):
            continue
        if float(pred_by_metric["exp"][idx]) < float(args.min_pred_exp_delta):
            continue
        if float(pred_by_metric["fin"][idx]) < float(args.min_pred_fin_delta):
            continue
        chosen_alt.add(sid)

    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for sid in sorted(base_rows):
            row = alt_rows[sid] if sid in chosen_alt and sid in alt_rows else base_rows[sid]
            row = dict(row)
            stage_outputs = dict(row.get("stage_outputs") or {})
            stage_outputs["qwen_pipe_variant_selector"] = {
                "selected_variant": "alt" if sid in chosen_alt else "base",
                "pred_delta": float(selector_pred[sample_ids.index(sid)]) if sid in sample_ids else 0.0,
                "pred_fin_delta": float(pred_by_metric["fin"][sample_ids.index(sid)]) if sid in sample_ids else 0.0,
                "pred_loc_delta": float(pred_by_metric["loc"][sample_ids.index(sid)]) if sid in sample_ids else 0.0,
                "pred_rep_delta": float(pred_by_metric["rep"][sample_ids.index(sid)]) if sid in sample_ids else 0.0,
                "pred_exp_delta": float(pred_by_metric["exp"][sample_ids.index(sid)]) if sid in sample_ids else 0.0,
                "objective": args.objective,
                "base_raw": str(resolve_pipe_path(args.base_raw)),
                "alt_raw": str(resolve_pipe_path(args.alt_raw)),
                "policy": "Sample-level CV ridge predicts whether alt raw improves weighted local proxy or localization with report/exp guards from GT-free report and stage features.",
            }
            row["stage_outputs"] = stage_outputs
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    actual = [y_by_metric["fin"][i] for i, sid in enumerate(sample_ids) if sid in chosen_alt]
    summary = {
        "sample_count": len(sample_ids),
        "selected_alt": len(chosen_alt),
        "target_delta_mean": float(y_by_metric["fin"].mean()) if len(sample_ids) else 0.0,
        "target_delta_positive_rate": float((y_by_metric["fin"] > 0).mean()) if len(sample_ids) else 0.0,
        "pred_delta_mean": float(selector_pred.mean()) if len(selector_pred) else 0.0,
        "pred_delta_max": float(selector_pred.max()) if len(selector_pred) else 0.0,
        "pred_metric_means": {metric: float(values.mean()) if len(values) else 0.0 for metric, values in pred_by_metric.items()},
        "pred_metric_max": {metric: float(values.max()) if len(values) else 0.0 for metric, values in pred_by_metric.items()},
        "target_metric_means": {metric: float(values.mean()) if len(values) else 0.0 for metric, values in y_by_metric.items()},
        "target_metric_positive_rates": {metric: float((values > 0).mean()) if len(values) else 0.0 for metric, values in y_by_metric.items()},
        "selected_actual_delta_mean": float(np.mean(actual)) if actual else 0.0,
        "selected_actual_delta_positive_rate": float(np.mean([v > 0 for v in actual])) if actual else 0.0,
        "selected_actual_metric_means": {
            metric: float(np.mean([values[i] for i, sid in enumerate(sample_ids) if sid in chosen_alt])) if chosen_alt else 0.0
            for metric, values in y_by_metric.items()
        },
        "base_raw": str(resolve_pipe_path(args.base_raw)),
        "alt_raw": str(resolve_pipe_path(args.alt_raw)),
        "output_jsonl": str(out_path),
        "select_threshold": args.select_threshold,
        "objective": args.objective,
        "rep_loss_weight": args.rep_loss_weight,
        "exp_loss_weight": args.exp_loss_weight,
        "min_pred_loc_delta": args.min_pred_loc_delta,
        "min_pred_rep_delta": args.min_pred_rep_delta,
        "min_pred_exp_delta": args.min_pred_exp_delta,
        "min_pred_fin_delta": args.min_pred_fin_delta,
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
