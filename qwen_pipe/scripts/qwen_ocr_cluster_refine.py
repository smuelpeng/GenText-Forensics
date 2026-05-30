#!/usr/bin/env python3
"""OCR row/paragraph/table-cluster localization refinement.

This is a GT-blind local postprocess on top of qwen_pipe_v71. It borrows the
FakeShield idea of explanation-first localization: use the current anomaly
reason as the description, then ground relationship-like anomalies to OCR
rows, paragraphs, or table clusters instead of a single text span.

GT eval files may be used only for selecting diagnostic subsets. They are never
used to construct prompts or to choose boxes for a full inference run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v71_text_crop_local_gated_300.jsonl"
DEFAULT_EVAL = PIPE_ROOT / "outputs/eval/qwen_pipe_v71_text_crop_local_gated_300.json"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_text_crop_verify import (  # noqa: E402
    Anomaly,
    Span,
    area,
    clamp_box,
    collect_spans,
    expand_box,
    extract_text_queries,
    make_anomalies,
    overlap,
    replace_groundings,
    resolve_image_path,
    resolve_pipe_path,
    setup_debug_import,
    span_match_score,
    union_boxes,
)


BODY_CUES = (
    "body",
    "paragraph",
    "lower",
    "section",
    "question",
    "questions",
    "smearing",
    "ghosting",
    "bleed",
    "bleeding",
    "merged",
    "text block",
    "正文",
    "下半",
    "段落",
    "小节",
    "问题",
    "题目",
    "涂抹",
    "重影",
    "模糊",
    "ซ้อน",
    "ย่อหน้า",
    "ด้านล่าง",
    "คำถาม",
    "نص الجسم",
    "السفلي",
    "أسئلة",
    "تشوي",
    "تكرار",
    "بقع",
)

TABLE_CUES = (
    "table",
    "row",
    "column",
    "cell",
    "total",
    "sum",
    "subtotal",
    "duplicate",
    "duplicated",
    "repeated",
    "missing",
    "sequence",
    "numbering",
    "skipping",
    "list structure",
    "表",
    "表格",
    "行",
    "列",
    "单元格",
    "合计",
    "总计",
    "小计",
    "重复",
    "缺失",
    "编号",
    "序号",
    "ลำดับ",
    "ตาราง",
    "แถว",
    "คอลัมน์",
    "รวม",
    "ซ้ำ",
    "ขาด",
    "جدول",
    "صف",
    "عمود",
    "مجموع",
    "تكرار",
    "مفقود",
)


@dataclass
class RowCluster:
    index: int
    spans: list[Span]
    box: list[int]


@dataclass
class ClusterDecision:
    box: list[int] | None
    reason: str
    meta: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_low_loc_selection(eval_json: Path, threshold: float, limit: int) -> set[str]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    samples = [
        s
        for s in data.get("samples") or []
        if s.get("gt_label") == "FORGED"
        and s.get("pred_label") == "FORGED"
        and float(s.get("loc_score") or 0.0) < threshold
    ]
    samples.sort(key=lambda s: (float(s.get("loc_score") or 0.0), str(s.get("sample_id") or "")))
    if limit > 0:
        samples = samples[:limit]
    return {str(s.get("sample_id")) for s in samples}


def load_worsened_selection(current_eval: Path, previous_eval: Path, min_delta: float, limit: int) -> set[str]:
    current = json.loads(current_eval.read_text(encoding="utf-8"))
    previous = json.loads(previous_eval.read_text(encoding="utf-8"))
    prev = {str(s.get("sample_id")): s for s in previous.get("samples") or []}
    rows: list[tuple[float, str]] = []
    for sample in current.get("samples") or []:
        sid = str(sample.get("sample_id") or "")
        old = float((prev.get(sid) or {}).get("loc_score") or 0.0)
        new = float(sample.get("loc_score") or 0.0)
        if (
            sample.get("gt_label") == "FORGED"
            and sample.get("pred_label") == "FORGED"
            and old - new >= min_delta
        ):
            rows.append((old - new, sid))
    rows.sort(reverse=True)
    if limit > 0:
        rows = rows[:limit]
    return {sid for _, sid in rows}


def has_any(text: str, cues: tuple[str, ...]) -> bool:
    folded = text.lower()
    return any(cue.lower() in folded for cue in cues)


def explode_multiline_spans(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for span in spans:
        text = span.text or ""
        lines = [line.strip() for line in re.split(r"\n+", text) if line.strip()]
        if len(lines) <= 1:
            out.append(span)
            continue
        height = max(1, span.box[3] - span.box[1])
        if height < 10:
            out.append(span)
            continue
        step = height / len(lines)
        for idx, line in enumerate(lines):
            y1 = span.box[1] + idx * step
            y2 = span.box[1] + (idx + 1) * step
            box = clamp_box([span.box[0], y1, span.box[2], y2], 10**9, 10**9)
            if box:
                out.append(Span(f"{span.span_id}.{idx + 1}", line, box, span.role))
        out.append(span)
    return out


def y_overlap(a: list[int], b: list[int]) -> float:
    y1 = max(a[1], b[1])
    y2 = min(a[3], b[3])
    inter = max(0, y2 - y1)
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def build_rows(spans: list[Span]) -> list[RowCluster]:
    line_spans = sorted(explode_multiline_spans(spans), key=lambda s: ((s.box[1] + s.box[3]) / 2.0, s.box[0]))
    rows: list[list[Span]] = []
    for span in line_spans:
        if not rows:
            rows.append([span])
            continue
        current_box = union_boxes([s.box for s in rows[-1]]) or span.box
        current_h = max(1, current_box[3] - current_box[1])
        span_h = max(1, span.box[3] - span.box[1])
        center_gap = abs((span.box[1] + span.box[3] - current_box[1] - current_box[3]) / 2.0)
        if y_overlap(span.box, current_box) >= 0.25 or center_gap <= max(8, 0.65 * max(current_h, span_h)):
            rows[-1].append(span)
        else:
            rows.append([span])

    clusters: list[RowCluster] = []
    for idx, row in enumerate(rows):
        box = union_boxes([s.box for s in row])
        if box:
            clusters.append(RowCluster(idx, row, box))
    return clusters


def rows_overlapping_box(rows: list[RowCluster], box: list[int], width: int, height: int) -> list[int]:
    expanded = expand_box(box, width, height, 0.05, 0.10, min_pad=8)
    return [row.index for row in rows if overlap(row.box, expanded) > 0]


def matched_row_indices(rows: list[RowCluster], anomaly: Anomaly) -> list[int]:
    queries = extract_text_queries(anomaly.reason)
    scored: list[tuple[float, int]] = []
    for row in rows:
        row_text = " ".join(span.text for span in row.spans)
        pseudo = Span(f"row{row.index}", row_text, row.box)
        score = span_match_score(pseudo, queries)
        if score > 0:
            scored.append((score, row.index))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:6]]


def contiguous_indices(seed_indices: list[int], max_index: int, radius: int) -> list[int]:
    selected: set[int] = set()
    for idx in seed_indices:
        for j in range(max(0, idx - radius), min(max_index, idx + radius) + 1):
            selected.add(j)
    return sorted(selected)


def candidate_from_rows(rows: list[RowCluster], indices: list[int], width: int, height: int) -> list[int] | None:
    selected = [rows[i].box for i in indices if 0 <= i < len(rows)]
    box = union_boxes(selected)
    if not box:
        return None
    return clamp_box(box, width, height)


def get_history_box(row: dict[str, Any], anomaly_index: int, width: int, height: int) -> list[int] | None:
    meta = ((row.get("stage_outputs") or {}).get("qwen_pipe_text_crop_verify") or {}).get("anomalies") or []
    if anomaly_index >= len(meta):
        return None
    old_box = meta[anomaly_index].get("old_box")
    if isinstance(old_box, list) and len(old_box) >= 4:
        return clamp_box([float(v) for v in old_box[:4]], width, height)
    return None


def box_ratio(box: list[int], width: int, height: int) -> float:
    return area(box) / max(1, width * height)


def choose_cluster_box(
    *,
    row: dict[str, Any],
    anomaly: Anomaly,
    rows: list[RowCluster],
    width: int,
    height: int,
    min_expand_factor: float,
    max_candidate_ratio: float,
    max_degenerate_candidate_ratio: float,
    degenerate_height: int,
    max_degenerate_expand: float,
    max_nondegenerate_expand: float,
) -> ClusterDecision:
    if not rows:
        return ClusterDecision(None, "no_rows", {})

    reason = anomaly.reason or ""
    body_cue = has_any(reason, BODY_CUES)
    table_cue = has_any(reason, TABLE_CUES)
    if anomaly.target_type not in {"logical_numeric", "render_pixel_blur", "layout_table"} and not (body_cue or table_cue):
        return ClusterDecision(None, "unsupported_type", {"target_type": anomaly.target_type})

    current = anomaly.box
    history = get_history_box(row, anomaly.index, width, height) or current
    current_ratio = box_ratio(current, width, height)
    history_ratio = box_ratio(history, width, height)
    broad_history = history_ratio > current_ratio * 2.0 and (history_ratio >= 0.025 or (history[3] - history[1]) / max(1, height) >= 0.12)

    seed_indices = matched_row_indices(rows, anomaly)
    if not seed_indices:
        seed_indices = rows_overlapping_box(rows, history if broad_history else current, width, height)

    if not seed_indices:
        return ClusterDecision(None, "no_seed_rows", {"body_cue": body_cue, "table_cue": table_cue})

    if body_cue and broad_history and anomaly.target_type in {"logical_numeric", "render_pixel_blur", "layout_table"}:
        ref = expand_box(history, width, height, 0.02, 0.08, min_pad=8)
        indices = [r.index for r in rows if overlap(r.box, ref) > 0]
        mode = "body_history_rows"
    elif table_cue:
        radius = 2 if broad_history else 1
        indices = contiguous_indices(seed_indices, len(rows) - 1, radius)
        if broad_history:
            allowed = set(rows_overlapping_box(rows, history, width, height))
            if allowed:
                indices = [idx for idx in indices if idx in allowed]
        mode = "table_neighbor_rows"
    elif anomaly.target_type == "logical_numeric" and broad_history and len(seed_indices) >= 2:
        lo, hi = min(seed_indices), max(seed_indices)
        indices = list(range(lo, hi + 1))
        mode = "multi_value_rows"
    else:
        return ClusterDecision(None, "no_cluster_cue", {"body_cue": body_cue, "table_cue": table_cue})

    candidate = candidate_from_rows(rows, indices, width, height)
    if not candidate:
        return ClusterDecision(None, "empty_candidate", {"indices": indices, "mode": mode})

    cand_ratio = box_ratio(candidate, width, height)
    expand_factor = area(candidate) / max(1, area(current))
    degenerate_current = (current[3] - current[1]) <= degenerate_height
    ratio_limit = max_degenerate_candidate_ratio if degenerate_current else max_candidate_ratio
    if cand_ratio > ratio_limit:
        return ClusterDecision(
            None,
            "candidate_too_large_strict",
            {
                "candidate": candidate,
                "ratio": cand_ratio,
                "ratio_limit": ratio_limit,
                "degenerate_current": degenerate_current,
                "mode": mode,
            },
        )
    if degenerate_current and expand_factor > max_degenerate_expand:
        return ClusterDecision(
            None,
            "degenerate_expand_factor_too_large",
            {
                "candidate": candidate,
                "expand_factor": expand_factor,
                "max_degenerate_expand": max_degenerate_expand,
                "mode": mode,
            },
        )
    if (not degenerate_current) and expand_factor > max_nondegenerate_expand:
        return ClusterDecision(
            None,
            "expand_factor_too_large",
            {
                "candidate": candidate,
                "expand_factor": expand_factor,
                "max_nondegenerate_expand": max_nondegenerate_expand,
                "mode": mode,
            },
        )
    if area(candidate) < area(current) * min_expand_factor and not (broad_history and body_cue):
        return ClusterDecision(None, "not_expanding_enough", {"candidate": candidate, "mode": mode})
    if broad_history and area(candidate) > area(history) * 1.25:
        return ClusterDecision(None, "exceeds_history", {"candidate": candidate, "history": history, "mode": mode})
    if not overlap(candidate, expand_box(history, width, height, 0.2, 0.2, min_pad=16)):
        return ClusterDecision(None, "history_distance_gate", {"candidate": candidate, "history": history, "mode": mode})
    if mode == "body_history_rows":
        current_ref = expand_box(current, width, height, 0.25, 0.35, min_pad=24)
        current_tall = (current[3] - current[1]) / max(1, height) >= 0.15
        if not current_tall and not overlap(candidate, current_ref):
            return ClusterDecision(
                None,
                "current_distance_gate",
                {"candidate": candidate, "current": current, "mode": mode, "current_tall": current_tall},
            )

    return ClusterDecision(
        candidate,
        mode,
        {
            "target_type": anomaly.target_type,
            "body_cue": body_cue,
            "table_cue": table_cue,
            "seed_rows": seed_indices,
            "selected_rows": indices,
            "current_box": current,
            "history_box": history,
            "candidate_ratio": cand_ratio,
            "expand_factor": expand_factor,
            "degenerate_current": degenerate_current,
            "ratio_limit": ratio_limit,
        },
    )


def process_row(
    row: dict[str, Any],
    *,
    debug_root: Path,
    min_expand_factor: float,
    max_candidate_ratio: float,
    max_degenerate_candidate_ratio: float,
    degenerate_height: int,
    max_degenerate_expand: float,
    max_nondegenerate_expand: float,
    coord_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(row)
    parsed = row.get("parsed") or {}
    if str(parsed.get("conclusion") or "").upper() != "FORGED":
        return out, {"applied": False, "reason": "non_forged"}
    image_path = resolve_image_path(row, debug_root)
    if not image_path:
        return out, {"applied": False, "reason": "missing_image"}
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    anomalies = make_anomalies(row, width, height)
    if not anomalies:
        return out, {"applied": False, "reason": "no_anomalies"}
    spans = collect_spans(row, width, height, coord_mode)
    rows = build_rows(spans)

    new_boxes = [a.box for a in anomalies]
    meta: list[dict[str, Any]] = []
    replaced = 0
    for anomaly in anomalies:
        decision = choose_cluster_box(
            row=row,
            anomaly=anomaly,
            rows=rows,
            width=width,
            height=height,
            min_expand_factor=min_expand_factor,
            max_candidate_ratio=max_candidate_ratio,
            max_degenerate_candidate_ratio=max_degenerate_candidate_ratio,
            degenerate_height=degenerate_height,
            max_degenerate_expand=max_degenerate_expand,
            max_nondegenerate_expand=max_nondegenerate_expand,
        )
        if decision.box is not None and decision.box != anomaly.box:
            new_boxes[anomaly.index] = decision.box
            replaced += 1
        meta.append(
            {
                "index": anomaly.index,
                "target_type": anomaly.target_type,
                "old_box": anomaly.box,
                "new_box": new_boxes[anomaly.index],
                "decision": decision.reason,
                "meta": decision.meta,
            }
        )

    report = str(row.get("raw_output") or "")
    new_report, count = replace_groundings(report, new_boxes)
    if replaced and count == len(new_boxes) and new_report != report:
        setup_debug_import(debug_root)
        from postprocess import parse_cct_report  # type: ignore

        out["raw_output"] = new_report
        out["parsed"] = parse_cct_report(new_report)

    stage_outputs = dict(out.get("stage_outputs") or {})
    stage_outputs["qwen_pipe_ocr_cluster_refine"] = {
        "applied": bool(replaced),
        "boxes_replaced": replaced,
        "row_count": len(rows),
        "anomalies": meta,
        "policy": "FakeShield-inspired explanation-first OCR row/paragraph/table cluster localization. GT is not used.",
    }
    out["stage_outputs"] = stage_outputs
    return out, stage_outputs["qwen_pipe_ocr_cluster_refine"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL), help="Used only for optional diagnostic subset selection.")
    parser.add_argument("--previous-eval-json", default="", help="Used only to select v71 regressions for diagnostic subsets.")
    parser.add_argument("--select-low-loc-limit", type=int, default=0)
    parser.add_argument("--low-loc-threshold", type=float, default=0.02)
    parser.add_argument("--select-worsened-limit", type=int, default=0)
    parser.add_argument("--worsened-min-delta", type=float, default=0.005)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--write-subset-only", action="store_true")
    parser.add_argument("--min-expand-factor", type=float, default=1.35)
    parser.add_argument("--max-candidate-ratio", type=float, default=0.08)
    parser.add_argument("--max-degenerate-candidate-ratio", type=float, default=0.18)
    parser.add_argument("--degenerate-height", type=int, default=20)
    parser.add_argument("--max-degenerate-expand", type=float, default=20.0)
    parser.add_argument("--max-nondegenerate-expand", type=float, default=15.0)
    parser.add_argument("--coord-mode", choices=["auto", "normalized-1000", "pixel"], default="normalized-1000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected: set[str] = set(args.sample_id or [])
    eval_path = resolve_pipe_path(args.eval_json)
    if args.select_low_loc_limit > 0:
        selected.update(load_low_loc_selection(eval_path, args.low_loc_threshold, args.select_low_loc_limit))
    if args.select_worsened_limit > 0:
        if not args.previous_eval_json:
            raise SystemExit("--select-worsened-limit requires --previous-eval-json")
        selected.update(
            load_worsened_selection(
                eval_path,
                resolve_pipe_path(args.previous_eval_json),
                args.worsened_min_delta,
                args.select_worsened_limit,
            )
        )

    stats: dict[str, Any] = {
        "rows": 0,
        "rows_written": 0,
        "rows_processed": 0,
        "rows_changed": 0,
        "boxes_replaced": 0,
        "selected_count": len(selected),
        "write_subset_only": args.write_subset_only,
        "decision_counts": {},
        "target_changed": {},
    }

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            sample_id = str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)
            should_process = not selected or sample_id in selected
            if should_process:
                out, meta = process_row(
                    row,
                    debug_root=debug_root,
                    min_expand_factor=args.min_expand_factor,
                    max_candidate_ratio=args.max_candidate_ratio,
                    max_degenerate_candidate_ratio=args.max_degenerate_candidate_ratio,
                    degenerate_height=args.degenerate_height,
                    max_degenerate_expand=args.max_degenerate_expand,
                    max_nondegenerate_expand=args.max_nondegenerate_expand,
                    coord_mode=args.coord_mode,
                )
                stats["rows_processed"] += 1
                if meta.get("applied"):
                    stats["rows_changed"] += 1
                    stats["boxes_replaced"] += int(meta.get("boxes_replaced") or 0)
                for anomaly in meta.get("anomalies") or []:
                    decision = str(anomaly.get("decision") or "unknown")
                    stats["decision_counts"][decision] = stats["decision_counts"].get(decision, 0) + 1
                    if anomaly.get("old_box") != anomaly.get("new_box"):
                        target = str(anomaly.get("target_type") or "unknown")
                        stats["target_changed"][target] = stats["target_changed"].get(target, 0) + 1
            else:
                out = row
            if (not args.write_subset_only) or should_process:
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
