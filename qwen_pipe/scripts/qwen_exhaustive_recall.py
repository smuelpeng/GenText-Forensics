#!/usr/bin/env python3
"""Exhaustive local candidate-pool diagnostics for low-S_Loc samples.

v86 is intentionally not a verifier run.  It asks one narrow question:

    Can a GT-blind OCR/visual candidate generator place *any* plausible region
    over the GT tamper mask/boxes?

The inference-side candidate generation reads only the current raw prediction,
image pixels, OCR/layout caches, and stage outputs.  GT labels, reports, masks,
and eval fields are used only after candidates are generated to measure oracle
coverage for a diagnostic subset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v85_embedding_v4_basekey_stablecache_k8_300.jsonl"
DEFAULT_EVAL = PIPE_ROOT / "outputs/eval/qwen_pipe_v85_embedding_v4_basekey_stablecache_k8_300.json"
DEFAULT_GT = DEFAULT_DEBUG_ROOT / "data/val_300.jsonl"
DEFAULT_OCR_LAYOUT_CACHE = DEFAULT_DEBUG_ROOT / "outputs/cache/ocr_layouts"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_issue_refine import (  # noqa: E402
    evidence_candidates,
    parse_box,
    project_box,
)
from qwen_ocr_cluster_refine import build_rows  # noqa: E402
from qwen_text_crop_verify import (  # noqa: E402
    Span,
    area,
    clamp_box,
    collect_spans,
    expand_box,
    extract_text_queries,
    overlap,
    resolve_image_path,
    resolve_pipe_path,
    setup_debug_import,
    union_boxes,
)


GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[([^\[\]]+)\]", re.IGNORECASE)
NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,:/-]\d+)*(?:\s?%|\s?万元|\s?万|年|月|日|\s?PM|\s?AM)?", re.IGNORECASE)
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff\u0e00-\u0e7f\u0600-\u06ff]{2,}", re.UNICODE)
VISUAL_TERMS = (
    "blur",
    "blurry",
    "pixel",
    "pixelated",
    "jagged",
    "artifact",
    "smudge",
    "black",
    "red",
    "yellow",
    "highlight",
    "font",
    "bold",
    "weight",
    "layout",
    "spacing",
    "table",
    "row",
    "cell",
    "overlap",
    "misalign",
    "模糊",
    "锯齿",
    "黑块",
    "颜色",
    "高亮",
    "字体",
    "加粗",
    "排版",
    "间距",
    "表格",
    "错位",
)


@dataclass
class RecallCandidate:
    label: str
    box: list[int]
    source: str
    family: str
    score: float
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_eval_selection(eval_json: Path, *, threshold: float, limit: int) -> list[str]:
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
    return [str(s.get("sample_id") or "") for s in samples]


def load_eval_samples(eval_json: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    return {str(s.get("sample_id") or ""): s for s in data.get("samples") or []}


def sample_id_from_row(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)


def resolve_debug_path(path: str | Path, debug_root: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return debug_root / p


def safe_cache_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def box_iou(a: list[int] | list[float], b: list[int] | list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    bb = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / max(1e-9, aa + bb - inter)


def contain_recall(gt_box: list[int] | list[float], cand_box: list[int] | list[float]) -> float:
    inter = overlap(
        [int(round(v)) for v in gt_box],
        [int(round(v)) for v in cand_box],
    )
    return inter / max(1, area([int(round(v)) for v in gt_box]))


def parse_grounding_boxes(report: str, width: int, height: int) -> list[list[int]]:
    boxes: list[list[int]] = []
    for match in GROUNDING_RE.finditer(report or ""):
        nums = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        if len(nums) < 4:
            continue
        try:
            box = clamp_box([float(v) for v in nums[:4]], width, height)
        except ValueError:
            continue
        if box:
            boxes.append(box)
    return boxes


def report_grounding_contexts(report: str, width: int, height: int) -> list[tuple[list[int], str]]:
    matches = list(GROUNDING_RE.finditer(report or ""))
    out: list[tuple[list[int], str]] = []
    for idx, match in enumerate(matches):
        nums = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        if len(nums) < 4:
            continue
        box = clamp_box([float(v) for v in nums[:4]], width, height)
        if not box:
            continue
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(report or "")
        block_start = (report or "").rfind("###", 0, match.start())
        if block_start < 0:
            block_start = max(0, (report or "").rfind("\n", 0, match.start()))
        context = (report or "")[block_start:next_start]
        out.append((box, re.sub(r"\s+", " ", context).strip()))
    return out


def boxes_to_mask(boxes: Iterable[list[int]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        clamped = clamp_box([float(v) for v in box[:4]], width, height)
        if not clamped:
            continue
        x1, y1, x2, y2 = clamped
        mask[y1:y2, x1:x2] = True
    return mask


def read_gt_mask(gt_row: dict[str, Any], width: int, height: int, debug_root: Path) -> np.ndarray | None:
    raw = str(gt_row.get("mask_path") or "")
    if not raw:
        return None
    path = resolve_debug_path(raw, debug_root)
    if not path.exists():
        return None
    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != (width, height):
            im = im.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(im) > 0


def mask_stats(gt_mask: np.ndarray | None, boxes: list[list[int]], width: int, height: int) -> dict[str, float | None]:
    if gt_mask is None:
        return {"mask_precision": None, "mask_recall": None, "mask_f1": None, "mask_iou": None, "mask_area_ratio": None}
    pred = boxes_to_mask(boxes, width, height)
    tp = float(np.logical_and(gt_mask, pred).sum())
    fp = float(np.logical_and(~gt_mask, pred).sum())
    fn = float(np.logical_and(gt_mask, ~pred).sum())
    union = float(np.logical_or(gt_mask, pred).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn > 0 else 0.0
    iou = tp / union if union > 0 else 0.0
    return {
        "mask_precision": precision,
        "mask_recall": recall,
        "mask_f1": f1,
        "mask_iou": iou,
        "mask_area_ratio": float(pred.sum()) / max(1, width * height),
    }


def box_coverage(gt_boxes: list[list[int]], boxes: list[list[int]]) -> dict[str, float | int]:
    if not gt_boxes:
        return {
            "gt_box_count": 0,
            "mean_best_iou": 0.0,
            "gt_recall_iou_0p1": 0.0,
            "gt_recall_iou_0p3": 0.0,
            "gt_recall_cover_0p3": 0.0,
            "gt_recall_cover_0p5": 0.0,
        }
    best_iou: list[float] = []
    best_cover: list[float] = []
    for gt in gt_boxes:
        if boxes:
            best_iou.append(max(box_iou(gt, box) for box in boxes))
            best_cover.append(max(contain_recall(gt, box) for box in boxes))
        else:
            best_iou.append(0.0)
            best_cover.append(0.0)
    return {
        "gt_box_count": len(gt_boxes),
        "mean_best_iou": float(np.mean(best_iou)),
        "gt_recall_iou_0p1": float(np.mean([v >= 0.10 for v in best_iou])),
        "gt_recall_iou_0p3": float(np.mean([v >= 0.30 for v in best_iou])),
        "gt_recall_cover_0p3": float(np.mean([v >= 0.30 for v in best_cover])),
        "gt_recall_cover_0p5": float(np.mean([v >= 0.50 for v in best_cover])),
    }


def load_native_ocr_spans(sample_id: str, cache_dir: Path, model: str, width: int, height: int) -> list[Span]:
    path = cache_dir / safe_cache_name(model) / f"{safe_cache_name(sample_id)}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    spans: list[Span] = []
    parsed = data.get("parsed") or {}
    for idx, span in enumerate(parsed.get("text_spans") or [], start=1):
        if not isinstance(span, dict):
            continue
        box = clamp_box([float(v) for v in (span.get("bbox") or [])[:4]], width, height)
        text = str(span.get("text") or "")
        if box and text:
            spans.append(Span(str(span.get("id") or f"q{idx}"), text, box, str(span.get("role") or "qwen_ocr")))
    return spans


def dedupe_spans(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for span in spans:
        key = (re.sub(r"\s+", " ", span.text).strip().lower(), tuple(span.box))
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out


def row_text(row: Any) -> str:
    return " ".join(str(span.text or "") for span in row.spans)


def page_area_ratio(box: list[int], width: int, height: int) -> float:
    return area(box) / max(1, width * height)


def add_candidate(
    candidates: list[RecallCandidate],
    *,
    label: str,
    box: list[int] | None,
    source: str,
    family: str,
    score: float,
    width: int,
    height: int,
    text: str = "",
    meta: dict[str, Any] | None = None,
    max_area_ratio: float = 0.35,
) -> None:
    if not box:
        return
    clamped = clamp_box([float(v) for v in box], width, height)
    if not clamped:
        return
    if area(clamped) < 20:
        return
    if page_area_ratio(clamped, width, height) > max_area_ratio:
        return
    candidates.append(
        RecallCandidate(
            label=label,
            box=clamped,
            source=source,
            family=family,
            score=float(score),
            text=text[:500],
            meta=meta or {},
        )
    )


def dedupe_candidates(candidates: list[RecallCandidate], *, duplicate_iou: float, limit: int) -> list[RecallCandidate]:
    out: list[RecallCandidate] = []
    for cand in sorted(candidates, key=lambda c: (c.score, -area(c.box)), reverse=True):
        if any(box_iou(cand.box, kept.box) >= duplicate_iou for kept in out):
            continue
        out.append(cand)
        if limit > 0 and len(out) >= limit:
            break
    return out


def visual_patch_candidates(image: Image.Image, width: int, height: int) -> list[RecallCandidate]:
    try:
        from qwen_crop_patch_refine import detect_patches  # type: ignore
    except Exception:
        return []

    candidates: list[RecallCandidate] = []
    for kind in ("dark", "red", "yellow", "pale_yellow"):
        for idx, patch in enumerate(detect_patches(image, kind), start=1):
            score = 7.0 + math.log1p(max(1.0, float(patch.score))) / 4.0
            add_candidate(
                candidates,
                label=f"patch_{kind}_{idx}",
                box=patch.box,
                source=f"patch:{kind}",
                family="patch",
                score=score,
                width=width,
                height=height,
                meta={"fill": patch.fill, "pixels": patch.pixels},
                max_area_ratio=0.10,
            )
            add_candidate(
                candidates,
                label=f"patch_{kind}_{idx}_pad",
                box=expand_box(patch.box, width, height, 0.50, 0.80, min_pad=10),
                source=f"patch:{kind}:expanded",
                family="patch",
                score=score - 0.2,
                width=width,
                height=height,
                meta={"fill": patch.fill, "pixels": patch.pixels},
                max_area_ratio=0.14,
            )
    return candidates


def dense_grid_candidates(width: int, height: int) -> list[RecallCandidate]:
    candidates: list[RecallCandidate] = []
    specs = [
        ("grid_3x4", 3, 4, 0.50, 3.0),
        ("grid_4x6", 4, 6, 0.50, 3.4),
        ("grid_6x8", 6, 8, 0.50, 3.8),
        ("grid_8x10", 8, 10, 0.50, 4.0),
    ]
    for name, cols, rows, stride_frac, base_score in specs:
        win_w = max(8, int(math.ceil(width / cols)))
        win_h = max(8, int(math.ceil(height / rows)))
        stride_x = max(4, int(round(win_w * stride_frac)))
        stride_y = max(4, int(round(win_h * stride_frac)))
        xs = list(range(0, max(1, width - win_w + 1), stride_x))
        ys = list(range(0, max(1, height - win_h + 1), stride_y))
        if not xs or xs[-1] != max(0, width - win_w):
            xs.append(max(0, width - win_w))
        if not ys or ys[-1] != max(0, height - win_h):
            ys.append(max(0, height - win_h))
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                add_candidate(
                    candidates,
                    label=f"{name}_{xi}_{yi}",
                    box=[x, y, min(width, x + win_w), min(height, y + win_h)],
                    source=name,
                    family="grid",
                    score=base_score,
                    width=width,
                    height=height,
                    max_area_ratio=0.09,
                )
    return candidates


def evidence_stage_candidates(row: dict[str, Any], spans: list[Span], width: int, height: int, coord_mode: str) -> list[RecallCandidate]:
    candidates: list[RecallCandidate] = []
    span_by_id = {span.span_id: span for span in spans if span.span_id}
    for idx, cand in enumerate(evidence_candidates(row), start=1):
        raw = parse_box(cand.get("bbox"))
        text = " ".join(str(cand.get(k) or "") for k in ("category", "evidence", "notes", "reason"))
        try:
            conf = float(cand.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if not raw:
            box = None
        else:
            box = project_box(raw, width, height, coord_mode)
        if box:
            add_candidate(
                candidates,
                label=f"evidence_{idx}",
                box=box,
                source="stage_evidence_candidate",
                family="evidence",
                score=5.0 + conf,
                width=width,
                height=height,
                text=text,
                meta={"category": cand.get("category"), "confidence": cand.get("confidence")},
                max_area_ratio=0.22,
            )
        span_ids = [str(v) for v in cand.get("span_ids") or []]
        span_boxes = [span_by_id[sid].box for sid in span_ids if sid in span_by_id]
        if span_boxes:
            union = union_boxes(span_boxes)
            add_candidate(
                candidates,
                label=f"evidence_{idx}_span_union",
                box=union,
                source="stage_evidence_span_ids",
                family="evidence",
                score=6.8 + conf,
                width=width,
                height=height,
                text=text,
                meta={"category": cand.get("category"), "confidence": cand.get("confidence"), "span_ids": span_ids},
                max_area_ratio=0.18,
            )
            if union:
                add_candidate(
                    candidates,
                    label=f"evidence_{idx}_span_union_pad",
                    box=expand_box(union, width, height, 0.12, 0.45, min_pad=6),
                    source="stage_evidence_span_ids:expanded",
                    family="evidence",
                    score=6.4 + conf,
                    width=width,
                    height=height,
                    text=text,
                    meta={"category": cand.get("category"), "confidence": cand.get("confidence"), "span_ids": span_ids},
                    max_area_ratio=0.20,
                )
            for sid in span_ids[:8]:
                span = span_by_id.get(sid)
                if not span:
                    continue
                add_candidate(
                    candidates,
                    label=f"evidence_{idx}_span_{sid}",
                    box=span.box,
                    source="stage_evidence_span_id",
                    family="evidence",
                    score=6.2 + conf,
                    width=width,
                    height=height,
                    text=f"{text} {span.text}",
                    meta={"category": cand.get("category"), "confidence": cand.get("confidence"), "span_id": sid},
                    max_area_ratio=0.10,
                )
    return candidates


def ocr_candidates(spans: list[Span], width: int, height: int) -> list[RecallCandidate]:
    candidates: list[RecallCandidate] = []
    for idx, span in enumerate(spans, start=1):
        add_candidate(
            candidates,
            label=f"span_{span.span_id or idx}",
            box=span.box,
            source=f"ocr_span:{span.role or 'unknown'}",
            family="ocr",
            score=6.0,
            width=width,
            height=height,
            text=span.text,
            max_area_ratio=0.08,
        )
        add_candidate(
            candidates,
            label=f"span_{span.span_id or idx}_pad",
            box=expand_box(span.box, width, height, 0.18, 0.45, min_pad=4),
            source=f"ocr_span:{span.role or 'unknown'}:expanded",
            family="ocr",
            score=5.6,
            width=width,
            height=height,
            text=span.text,
            max_area_ratio=0.10,
        )

    rows = build_rows(spans)
    for row in rows:
        text = row_text(row)
        add_candidate(
            candidates,
            label=f"row_{row.index}",
            box=row.box,
            source="ocr_row",
            family="row",
            score=5.4,
            width=width,
            height=height,
            text=text,
            max_area_ratio=0.10,
        )
        add_candidate(
            candidates,
            label=f"row_{row.index}_pad",
            box=expand_box(row.box, width, height, 0.03, 0.55, min_pad=6),
            source="ocr_row:expanded",
            family="row",
            score=5.2,
            width=width,
            height=height,
            text=text,
            max_area_ratio=0.12,
        )

    for radius in (1, 2, 4):
        for idx in range(len(rows)):
            lo = max(0, idx - radius)
            hi = min(len(rows) - 1, idx + radius)
            box = union_boxes([rows[j].box for j in range(lo, hi + 1)])
            text = " ".join(row_text(rows[j]) for j in range(lo, hi + 1))
            add_candidate(
                candidates,
                label=f"row_window_{lo}_{hi}_r{radius}",
                box=box,
                source=f"ocr_row_window_r{radius}",
                family="row",
                score=4.7 - 0.15 * radius,
                width=width,
                height=height,
                text=text,
                max_area_ratio=0.16,
            )

    # Paragraph/table-like clusters: consecutive OCR rows with small vertical gaps.
    if rows:
        heights = [max(1, r.box[3] - r.box[1]) for r in rows]
        med_h = float(np.median(heights))
        clusters: list[list[Any]] = []
        current: list[Any] = [rows[0]]
        for prev, cur in zip(rows, rows[1:]):
            gap = cur.box[1] - prev.box[3]
            same_band = gap <= max(12.0, 1.25 * med_h)
            if same_band and len(current) < 9:
                current.append(cur)
            else:
                clusters.append(current)
                current = [cur]
        clusters.append(current)
        for idx, cluster in enumerate(clusters, start=1):
            if len(cluster) < 2:
                continue
            box = union_boxes([r.box for r in cluster])
            text = " ".join(row_text(r) for r in cluster)
            add_candidate(
                candidates,
                label=f"paragraph_cluster_{idx}",
                box=box,
                source="ocr_paragraph_cluster",
                family="paragraph",
                score=4.1 + min(1.5, len(cluster) * 0.1),
                width=width,
                height=height,
                text=text,
                max_area_ratio=0.22,
            )
    return candidates


def current_report_candidates(row: dict[str, Any], width: int, height: int) -> list[RecallCandidate]:
    candidates: list[RecallCandidate] = []
    contexts = report_grounding_contexts(str(row.get("raw_output") or ""), width, height)
    if not contexts:
        contexts = [(box, "") for box in report_boxes(str(row.get("raw_output") or ""))]
    for idx, (box, context) in enumerate(contexts, start=1):
        clamped = clamp_box([float(v) for v in box], width, height)
        if not clamped:
            continue
        add_candidate(
            candidates,
            label=f"v85_final_{idx}",
            box=clamped,
            source="current_final_report",
            family="existing",
            score=8.0,
            width=width,
            height=height,
            text=context,
            max_area_ratio=0.28,
        )
        add_candidate(
            candidates,
            label=f"v85_final_{idx}_pad",
            box=expand_box(clamped, width, height, 0.20, 0.35, min_pad=12),
            source="current_final_report:expanded",
            family="existing",
            score=7.4,
            width=width,
            height=height,
            text=context,
            max_area_ratio=0.32,
        )
    return candidates


TOKEN_CANDIDATE_STOPWORDS = {
    "analysis",
    "anomaly",
    "authentic",
    "candidate",
    "document",
    "evidence",
    "forged",
    "forgery",
    "grounding",
    "layout",
    "reason",
    "report",
    "summary",
    "visual",
}


def subspan_box_for_match(span: Span, start: int, end: int, width: int, height: int) -> list[int] | None:
    text = span.text or ""
    if not text or start < 0 or end <= start:
        return None
    x1, y1, x2, y2 = span.box
    span_w = max(1, x2 - x1)
    span_h = max(1, y2 - y1)
    n = max(1, len(text))
    # Character-position projection is intentionally simple.  It turns coarse
    # OCR line boxes into small candidate anchors without needing word OCR.
    sx1 = x1 + span_w * max(0.0, min(1.0, start / n))
    sx2 = x1 + span_w * max(0.0, min(1.0, end / n))
    pad_x = max(4.0, min(18.0, span_w * 0.018))
    pad_y = max(3.0, min(12.0, span_h * 0.30))
    if sx2 - sx1 < max(8.0, span_w * 0.012):
        center = (sx1 + sx2) / 2.0
        half = max(4.0, span_w * 0.010)
        sx1, sx2 = center - half, center + half
    return clamp_box([sx1 - pad_x, y1 - pad_y, sx2 + pad_x, y2 + pad_y], width, height)


def token_candidate_terms(row: dict[str, Any]) -> list[str]:
    context = rank_context(row)
    terms: list[str] = []
    for term in list(context["numbers"]) + list(context["queries"]):
        cleaned = re.sub(r"\s+", " ", str(term or "")).strip()
        if not cleaned:
            continue
        low = cleaned.lower()
        if low in TOKEN_CANDIDATE_STOPWORDS:
            continue
        # Prefer terms that can plausibly identify a tampered token/cell.
        if not (any(ch.isdigit() for ch in cleaned) or len(cleaned) >= 5 or "." in cleaned or "/" in cleaned):
            continue
        if len(cleaned) > 80:
            continue
        terms.append(cleaned)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = re.sub(r"\s+", "", term).lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(term)
    return deduped[:80]


def ocr_token_subspan_candidates(row: dict[str, Any], spans: list[Span], width: int, height: int) -> list[RecallCandidate]:
    terms = token_candidate_terms(row)
    if not terms:
        return []
    candidates: list[RecallCandidate] = []
    for span_idx, span in enumerate(spans, start=1):
        text = span.text or ""
        if len(text) < 2:
            continue
        lower = text.lower()
        compact_lower = re.sub(r"\s+", "", lower)
        emitted_for_span = 0
        for term_idx, term in enumerate(terms, start=1):
            low = term.lower()
            compact = re.sub(r"\s+", "", low)
            starts: list[tuple[int, int, str]] = []
            pos = lower.find(low)
            if pos >= 0:
                starts.append((pos, pos + len(low), "exact"))
            elif compact and len(compact) >= 4 and compact in compact_lower:
                # Map compact match back to approximate raw character offsets.
                raw_positions: list[int] = []
                cpos = 0
                for i, ch in enumerate(lower):
                    if not ch.isspace():
                        if cpos >= compact_lower.find(compact) and cpos < compact_lower.find(compact) + len(compact):
                            raw_positions.append(i)
                        cpos += 1
                if raw_positions:
                    starts.append((raw_positions[0], raw_positions[-1] + 1, "compact"))
            if not starts:
                continue
            for start, end, match_kind in starts[:2]:
                box = subspan_box_for_match(span, start, end, width, height)
                if not box:
                    continue
                is_numeric = any(ch.isdigit() for ch in term)
                base = 7.9 if is_numeric else 7.1
                score = base + min(1.2, len(term) / 40.0)
                add_candidate(
                    candidates,
                    label=f"token_{span.span_id or span_idx}_{term_idx}_{match_kind}",
                    box=box,
                    source=f"ocr_token_subspan:{match_kind}",
                    family="token",
                    score=score,
                    width=width,
                    height=height,
                    text=f"{term} :: {span.text}",
                    meta={"term": term, "span_id": span.span_id, "match_kind": match_kind, "span_box": span.box},
                    max_area_ratio=0.035,
                )
                emitted_for_span += 1
                if emitted_for_span >= 8:
                    break
            if emitted_for_span >= 8:
                break
    return candidates


def ocr_linegrid_candidates(row: dict[str, Any], spans: list[Span], width: int, height: int) -> list[RecallCandidate]:
    """Generate OCR-line internal sliding-window anchors.

    Token candidates need a text match between the report and OCR span.  The
    remaining low-S_Loc cases often only know "this line/cell looks wrong" but
    not the exact altered token.  These line-grid windows provide GT-blind
    sub-line anchors, especially for Thai/Arabic or OCR spans where word
    segmentation is unreliable.
    """
    context = rank_context(row)
    language = str(row.get("language_code") or ((row.get("metadata") or {}).get("language_code")) or "").lower()
    dense_language = language in {"th", "ar", "zh", "ja", "ko"}
    candidates: list[RecallCandidate] = []
    emitted_total = 0
    for span_idx, span in enumerate(spans, start=1):
        text = span.text or ""
        if len(text) < 4:
            continue
        x1, y1, x2, y2 = span.box
        span_w = max(1, x2 - x1)
        span_h = max(1, y2 - y1)
        area_ratio = page_area_ratio(span.box, width, height)
        if area_ratio > 0.045 or span_w < 24 or span_h < 6:
            continue
        match_score, match_meta = text_match_score(text, context)
        # Keep all reasonably long text lines for dense scripts; for space-
        # separated scripts, require either report relevance or visual language
        # in the current report to avoid flooding the candidate pool.
        if not dense_language and match_score <= 0 and not context["has_visual_terms"]:
            continue
        n = max(1, len(text))
        fractions = [(0.0, 0.5), (0.5, 1.0), (0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)]
        if dense_language or span_w >= 240 or match_score > 0:
            fractions.extend([(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)])
        if dense_language and span_w >= 180:
            fractions.extend([(0.0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.0)])
        emitted_for_span = 0
        for frac_idx, (start_frac, end_frac) in enumerate(fractions, start=1):
            start = int(round(n * start_frac))
            end = int(round(n * end_frac))
            if end <= start:
                continue
            box = subspan_box_for_match(span, start, end, width, height)
            if not box:
                continue
            win_ratio = page_area_ratio(box, width, height)
            if win_ratio > 0.030:
                continue
            snippet = text[start:end].strip()
            score = 5.2 + min(2.0, match_score * 0.35)
            if dense_language:
                score += 0.55
            if any(ch.isdigit() for ch in snippet):
                score += 0.45
            add_candidate(
                candidates,
                label=f"linegrid_{span.span_id or span_idx}_{frac_idx}",
                box=box,
                source="ocr_linegrid:subline_window",
                family="linegrid",
                score=score,
                width=width,
                height=height,
                text=f"{snippet} :: {text}",
                meta={
                    "span_id": span.span_id,
                    "span_box": span.box,
                    "start_frac": start_frac,
                    "end_frac": end_frac,
                    "language": language,
                    "match_score": match_score,
                    **match_meta,
                },
                max_area_ratio=0.035,
            )
            emitted_for_span += 1
            emitted_total += 1
            if emitted_for_span >= (9 if dense_language else 6):
                break
        if emitted_total >= 220:
            break
    return candidates


def ocr_scriptgrid_candidates(
    row: dict[str, Any],
    spans: list[Span],
    width: int,
    height: int,
    *,
    language_code: str = "",
) -> list[RecallCandidate]:
    """Generate physical sub-line windows for scripts with weak char geometry.

    ``ocr_linegrid_candidates`` maps text character offsets to OCR-line x
    positions.  That is fragile for right-to-left Arabic and Thai OCR where the
    returned string order can diverge from visual glyph positions.  Script-grid
    candidates ignore character offsets and slice the OCR line by physical x
    fractions, giving the ranker independent anchors for dense-script low-S_Loc
    failures.
    """
    context = rank_context(row)
    language = str(
        language_code
        or row.get("language_code")
        or ((row.get("metadata") or {}).get("language_code"))
        or ""
    ).lower()
    if language not in {"ar", "th"}:
        return []

    candidates: list[RecallCandidate] = []
    fractions = [
        (0.0, 1 / 3),
        (1 / 3, 2 / 3),
        (2 / 3, 1.0),
        (0.0, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 1.0),
        (0.0, 1 / 6),
        (1 / 6, 2 / 6),
        (2 / 6, 3 / 6),
        (3 / 6, 4 / 6),
        (4 / 6, 5 / 6),
        (5 / 6, 1.0),
        (0.0, 0.20),
        (0.80, 1.0),
        (0.15, 0.50),
        (0.50, 0.85),
    ]
    emitted_total = 0
    for span_idx, span in enumerate(spans, start=1):
        text = span.text or ""
        if len(text) < 4:
            continue
        x1, y1, x2, y2 = span.box
        span_w = max(1, x2 - x1)
        span_h = max(1, y2 - y1)
        if span_w < 36 or span_h < 6:
            continue
        area_ratio = page_area_ratio(span.box, width, height)
        if area_ratio > 0.065:
            continue
        match_score, match_meta = text_match_score(text, context)
        # For ar/th, retain long OCR lines even without report-term hits.  The
        # purpose is coverage diagnosis for scripts where term-to-position
        # projection has already proven unreliable.
        if match_score <= 0 and len(text) < (9 if language == "th" else 7) and not context["has_visual_terms"]:
            continue
        pad_x = max(5.0, min(20.0, span_w * 0.020))
        pad_y = max(3.0, min(12.0, span_h * 0.28))
        emitted_for_span = 0
        for frac_idx, (start_frac, end_frac) in enumerate(fractions, start=1):
            sx1 = x1 + span_w * start_frac
            sx2 = x1 + span_w * end_frac
            if sx2 <= sx1:
                continue
            box = clamp_box([sx1 - pad_x, y1 - pad_y, sx2 + pad_x, y2 + pad_y], width, height)
            if not box:
                continue
            ratio = page_area_ratio(box, width, height)
            if ratio > 0.025:
                continue
            score = 6.1 + min(1.5, match_score * 0.25)
            if any(ch.isdigit() for ch in text):
                score += 0.35
            if context["has_visual_terms"]:
                score += 0.25
            # Edge windows are especially useful for RTL lines and stamped
            # identifiers; keep a slight bonus without reversing coordinates.
            if start_frac <= 0.001 or end_frac >= 0.999:
                score += 0.18
            add_candidate(
                candidates,
                label=f"scriptgrid_{span.span_id or span_idx}_{frac_idx}",
                box=box,
                source="ocr_scriptgrid:physical_window",
                family="linegrid",
                score=score,
                width=width,
                height=height,
                text=text,
                meta={
                    "span_id": span.span_id,
                    "span_box": span.box,
                    "physical_start_frac": start_frac,
                    "physical_end_frac": end_frac,
                    "language": language,
                    "match_score": match_score,
                    **match_meta,
                },
                max_area_ratio=0.030,
            )
            emitted_for_span += 1
            emitted_total += 1
            if emitted_for_span >= 10:
                break
        if emitted_total >= 260:
            break
    return candidates


def generate_candidates(
    row: dict[str, Any],
    image: Image.Image,
    debug_root: Path,
    ocr_layout_cache_dir: Path,
    ocr_layout_model: str,
    coord_mode: str,
    max_candidates: int,
    enable_token_candidates: bool = False,
    enable_linegrid_candidates: bool = False,
    enable_scriptgrid_candidates: bool = False,
    language_code: str = "",
) -> list[RecallCandidate]:
    width, height = image.size
    sample_id = sample_id_from_row(row)
    spans = collect_spans(row, width, height, coord_mode)
    spans.extend(load_native_ocr_spans(sample_id, ocr_layout_cache_dir, ocr_layout_model, width, height))
    spans = dedupe_spans(spans)

    candidates: list[RecallCandidate] = []
    candidates.extend(current_report_candidates(row, width, height))
    candidates.extend(evidence_stage_candidates(row, spans, width, height, coord_mode))
    candidates.extend(ocr_candidates(spans, width, height))
    if enable_token_candidates:
        candidates.extend(ocr_token_subspan_candidates(row, spans, width, height))
    if enable_linegrid_candidates:
        candidates.extend(ocr_linegrid_candidates(row, spans, width, height))
    if enable_scriptgrid_candidates:
        candidates.extend(ocr_scriptgrid_candidates(row, spans, width, height, language_code=language_code))
    candidates.extend(visual_patch_candidates(image, width, height))
    candidates.extend(dense_grid_candidates(width, height))
    return dedupe_candidates(candidates, duplicate_iou=0.92, limit=max_candidates)


def choose_topk_diverse(candidates: list[RecallCandidate], top_k: int) -> list[RecallCandidate]:
    selected: list[RecallCandidate] = []
    per_family: Counter[str] = Counter()
    for cand in sorted(candidates, key=lambda c: (c.score, -area(c.box)), reverse=True):
        soft_cap = {
            "existing": 3,
            "patch": 4,
            "ocr": 5,
            "token": 5,
            "linegrid": 5,
            "row": 6,
            "paragraph": 3,
            "evidence": 4,
            "grid": 4,
        }.get(cand.family, 4)
        if per_family[cand.family] >= soft_cap and len(selected) < top_k // 2:
            continue
        if any(box_iou(cand.box, prev.box) >= 0.75 for prev in selected):
            continue
        selected.append(cand)
        per_family[cand.family] += 1
        if len(selected) >= top_k:
            break
    return selected


def rank_context(row: dict[str, Any]) -> dict[str, Any]:
    raw_report = str(row.get("raw_output") or "")
    pieces = [raw_report]
    parsed = (((row.get("stage_outputs") or {}).get("evidence_candidates") or {}).get("parsed") or {})
    for key in ("visual_candidates", "logical_candidates"):
        for cand in parsed.get(key) or []:
            if isinstance(cand, dict):
                pieces.extend(str(cand.get(k) or "") for k in ("category", "evidence", "notes", "reason"))
    text = "\n".join(pieces)
    queries = extract_text_queries(text)
    nums = NUMERIC_RE.findall(text)
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "report",
        "analysis",
        "forgery",
        "anomaly",
        "visual",
        "clumsy",
        "reason",
        "summary",
        "document",
    }
    token_counts = Counter(t for t in tokens if len(t) >= 3 and t not in stop)
    salient_tokens = [t for t, _ in token_counts.most_common(80)]
    return {
        "text": text,
        "queries": queries[:32],
        "numbers": list(dict.fromkeys(nums))[:48],
        "tokens": salient_tokens,
        "has_visual_terms": any(term.lower() in text.lower() for term in VISUAL_TERMS),
    }


def text_match_score(candidate_text: str, context: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    text = (candidate_text or "").lower()
    compact = re.sub(r"\s+", "", text)
    score = 0.0
    hits: list[str] = []
    for query in context["queries"]:
        q = query.lower()
        qc = re.sub(r"\s+", "", q)
        if not q or len(q) < 2:
            continue
        if q in text or (qc and qc in compact):
            add = 3.0 + min(2.5, len(q) / 18.0)
            score += add
            hits.append(query)
        else:
            parts = [p for p in TOKEN_RE.findall(q) if len(p) >= 3]
            part_hits = sum(1 for p in parts if p.lower() in text)
            if part_hits:
                score += min(2.0, 0.55 * part_hits)
                hits.append(query)
    number_hits = [n for n in context["numbers"] if n and n.lower() in text]
    if number_hits:
        score += min(5.0, 1.2 + 0.8 * len(number_hits))
    token_hits = [t for t in context["tokens"][:50] if t in text]
    if token_hits:
        score += min(4.0, 0.18 * len(token_hits))
    visual_hits = [term for term in VISUAL_TERMS if term.lower() in text]
    if visual_hits:
        score += min(2.0, 0.5 * len(visual_hits))
    return score, {
        "query_hits": hits[:8],
        "number_hits": number_hits[:8],
        "token_hit_count": len(token_hits),
        "visual_hits": visual_hits[:8],
    }


def size_prior(candidate: RecallCandidate, width: int, height: int) -> float:
    ratio = page_area_ratio(candidate.box, width, height)
    if candidate.family == "grid":
        return 0.6 if 0.025 <= ratio <= 0.10 else -1.0
    if candidate.family == "token":
        return 1.4 if ratio <= 0.010 else 0.5
    if candidate.family == "linegrid":
        return 1.0 if ratio <= 0.012 else 0.4
    if ratio < 0.00001:
        return -0.5
    if ratio <= 0.015:
        return 1.2
    if ratio <= 0.06:
        return 0.7
    if ratio <= 0.14:
        return 0.1
    return -1.2


def score_candidate_v87(candidate: RecallCandidate, context: dict[str, Any], width: int, height: int) -> float:
    base = {
        "evidence": 4.4,
        "token": 4.0,
        "linegrid": 3.9,
        "ocr": 3.7,
        "row": 3.4,
        "paragraph": 2.8,
        "patch": 2.6,
        "existing": 2.4,
        "grid": 1.4,
    }.get(candidate.family, 2.0)
    query_score, query_meta = text_match_score(candidate.text, context)
    source_bonus = 0.0
    if "span_ids" in candidate.source:
        source_bonus += 2.2
    if candidate.source == "stage_evidence_span_id":
        source_bonus += 1.3
    if candidate.source.startswith("ocr_span:qwen_ocr"):
        source_bonus += 0.4
    if candidate.source.startswith("ocr_token_subspan"):
        source_bonus += 1.0
    if candidate.source.startswith("ocr_linegrid"):
        source_bonus += 0.55
    if candidate.source.startswith("ocr_scriptgrid"):
        source_bonus += 0.75
    if candidate.family == "patch" and context["has_visual_terms"]:
        source_bonus += 0.8
    if candidate.family == "existing" and query_score <= 0:
        source_bonus -= 1.6
    if candidate.family == "grid":
        source_bonus -= 0.8
    score = base + query_score + source_bonus + size_prior(candidate, width, height)
    candidate.meta["v87_rank"] = {
        "score": score,
        "base": base,
        "query_score": query_score,
        "source_bonus": source_bonus,
        "size_prior": size_prior(candidate, width, height),
        **query_meta,
    }
    return score


def choose_topk_v87(row: dict[str, Any], candidates: list[RecallCandidate], top_k: int, width: int, height: int) -> list[RecallCandidate]:
    context = rank_context(row)
    ranked = sorted(
        candidates,
        key=lambda c: (score_candidate_v87(c, context, width, height), -page_area_ratio(c.box, width, height)),
        reverse=True,
    )
    selected: list[RecallCandidate] = []
    per_family: Counter[str] = Counter()
    family_caps = {
        "evidence": 5,
        "token": 7,
        "linegrid": 7,
        "ocr": 7,
        "row": 5,
        "paragraph": 2,
        "patch": 2,
        "existing": 2,
        "grid": 3,
    }
    # First pass: take ranked evidence-bearing boxes with stronger caps and
    # spatial diversity.  This prevents ungrounded patches/current boxes from
    # occupying the budget before OCR text anchors are tried.
    for cand in ranked:
        score = float((cand.meta.get("v87_rank") or {}).get("score") or 0.0)
        query_score = float((cand.meta.get("v87_rank") or {}).get("query_score") or 0.0)
        if score < 4.0 and query_score <= 0 and cand.family != "grid":
            continue
        if per_family[cand.family] >= family_caps.get(cand.family, 3):
            continue
        if any(box_iou(cand.box, prev.box) >= 0.68 for prev in selected):
            continue
        selected.append(cand)
        per_family[cand.family] += 1
        if len(selected) >= top_k:
            return selected

    # Second pass: fill remaining budget with broad spatial fallback.  Grid
    # candidates are allowed late, as recall insurance, but cannot dominate.
    for cand in ranked:
        if cand in selected:
            continue
        if per_family[cand.family] >= family_caps.get(cand.family, 3):
            continue
        if any(box_iou(cand.box, prev.box) >= 0.72 for prev in selected):
            continue
        selected.append(cand)
        per_family[cand.family] += 1
        if len(selected) >= top_k:
            break
    return selected


def candidate_group_key(candidate: RecallCandidate) -> str:
    evidence_match = re.match(r"(evidence_\d+)", candidate.label)
    if evidence_match:
        return evidence_match.group(1)
    final_match = re.match(r"(v85_final_\d+)", candidate.label)
    if final_match:
        return final_match.group(1)
    row_match = re.match(r"row(?:_window)?_(\d+)", candidate.label)
    if row_match:
        return f"row_{row_match.group(1)}"
    if candidate.family == "grid":
        return candidate.label.rsplit("_", 2)[0]
    return candidate.label.replace("_pad", "").replace(":expanded", "")


def compact_adjustment(candidate: RecallCandidate, width: int, height: int) -> float:
    ratio = page_area_ratio(candidate.box, width, height)
    adj = 0.0
    if candidate.family == "ocr" and ratio <= 0.006:
        adj += 2.4
    if candidate.family == "token" and ratio <= 0.012:
        adj += 2.8
    if candidate.family == "evidence" and candidate.source == "stage_evidence_span_id":
        adj += 1.8
    if candidate.family == "evidence" and "span_ids" in candidate.source and ratio > 0.025:
        adj -= min(6.0, 1.8 + ratio * 28.0)
    if candidate.source.endswith(":expanded"):
        adj -= min(4.5, 0.8 + ratio * 22.0)
    if "row_window" in candidate.source:
        adj -= min(5.0, 0.9 + ratio * 18.0)
    if candidate.family == "existing":
        adj -= 2.0 if ratio > 0.01 else 0.8
    if candidate.family == "paragraph":
        adj -= min(4.0, 1.2 + ratio * 12.0)
    if candidate.family == "grid":
        adj -= 0.6
    if NUMERIC_RE.search(candidate.text or ""):
        adj += 0.8
    return adj


def choose_topk_v87b(row: dict[str, Any], candidates: list[RecallCandidate], top_k: int, width: int, height: int) -> list[RecallCandidate]:
    context = rank_context(row)
    for cand in candidates:
        base_score = score_candidate_v87(cand, context, width, height)
        adj = compact_adjustment(cand, width, height)
        cand.meta["v87b_rank"] = {
            **(cand.meta.get("v87_rank") or {}),
            "score": base_score + adj,
            "compact_adjustment": adj,
            "group_key": candidate_group_key(cand),
        }
    ranked = sorted(
        candidates,
        key=lambda c: (
            float((c.meta.get("v87b_rank") or {}).get("score") or 0.0),
            -page_area_ratio(c.box, width, height),
        ),
        reverse=True,
    )
    selected: list[RecallCandidate] = []
    per_family: Counter[str] = Counter()
    per_group: Counter[str] = Counter()
    area_budget = 0.24
    family_caps = {
        "evidence": 5,
        "token": 8,
        "linegrid": 8,
        "ocr": 8,
        "row": 3,
        "paragraph": 1,
        "patch": 2,
        "existing": 1,
        "grid": 3,
    }

    def can_take(cand: RecallCandidate, *, fallback: bool) -> bool:
        rank = cand.meta.get("v87b_rank") or {}
        score = float(rank.get("score") or 0.0)
        query_score = float(rank.get("query_score") or 0.0)
        ratio = page_area_ratio(cand.box, width, height)
        group = str(rank.get("group_key") or candidate_group_key(cand))
        if per_family[cand.family] >= family_caps.get(cand.family, 3):
            return False
        group_cap = 2 if cand.family in {"evidence", "ocr", "token", "linegrid", "row"} else 1
        if per_group[group] >= group_cap:
            return False
        if any(box_iou(cand.box, prev.box) >= 0.66 for prev in selected):
            return False
        current_area = sum(page_area_ratio(prev.box, width, height) for prev in selected)
        if not fallback:
            if ratio > 0.055 and cand.family not in {"grid"}:
                return False
            if current_area + ratio > area_budget:
                return False
            if score < 5.2 and query_score <= 0 and cand.family != "grid":
                return False
        else:
            if current_area + ratio > area_budget * 1.25 and cand.family != "grid":
                return False
        return True

    # Compact evidence-bearing pass.
    for cand in ranked:
        if not can_take(cand, fallback=False):
            continue
        selected.append(cand)
        per_family[cand.family] += 1
        per_group[candidate_group_key(cand)] += 1
        if len(selected) >= top_k:
            return selected

    # Recall-insurance pass: add a few grid/row candidates after compact anchors.
    for cand in ranked:
        if cand in selected:
            continue
        if cand.family not in {"grid", "row", "ocr", "token", "linegrid", "evidence"}:
            continue
        if not can_take(cand, fallback=True):
            continue
        selected.append(cand)
        per_family[cand.family] += 1
        per_group[candidate_group_key(cand)] += 1
        if len(selected) >= top_k:
            break
    return selected


def append_nonoverlapping(
    selected: list[RecallCandidate],
    additions: list[RecallCandidate],
    *,
    top_k: int,
    duplicate_iou: float,
) -> list[RecallCandidate]:
    out = list(selected)
    for cand in additions:
        if cand in out:
            continue
        if any(box_iou(cand.box, prev.box) >= duplicate_iou for prev in out):
            continue
        out.append(cand)
        if len(out) >= top_k:
            break
    return out


def choose_topk_v87mix(row: dict[str, Any], candidates: list[RecallCandidate], top_k: int, width: int, height: int) -> list[RecallCandidate]:
    # v87b supplies report/span-id aware compact recall anchors.  v86's old
    # local ranking was weaker for coverage but had better mask precision.  The
    # mix keeps them as two independent branches and fuses by non-overlap.
    recall_anchors = choose_topk_v87b(row, candidates, max(6, top_k // 2), width, height)
    precision_anchors = choose_topk_diverse(candidates, top_k)
    selected = append_nonoverlapping(recall_anchors, precision_anchors, top_k=top_k, duplicate_iou=0.62)
    if len(selected) < top_k:
        selected = append_nonoverlapping(
            selected,
            choose_topk_v87b(row, candidates, top_k, width, height),
            top_k=top_k,
            duplicate_iou=0.68,
        )
    return selected


def choose_topk_v89token(row: dict[str, Any], candidates: list[RecallCandidate], top_k: int, width: int, height: int) -> list[RecallCandidate]:
    # v89 is a precision-first variant: token/subspan boxes are cheap in area
    # and empirically carry higher mask F1 than whole OCR rows.  We seed the
    # selection with small token or evidence-span boxes, then keep v87mix as a
    # fallback for recall.
    _ = choose_topk_v87b(row, candidates, top_k, width, height)
    ranked = sorted(
        candidates,
        key=lambda c: (
            float((c.meta.get("v87b_rank") or c.meta.get("v87_rank") or {}).get("score") or c.score),
            -page_area_ratio(c.box, width, height),
        ),
        reverse=True,
    )
    selected: list[RecallCandidate] = []
    per_family: Counter[str] = Counter()
    area_budget = 0.13

    def add_if_ok(cand: RecallCandidate, *, max_ratio: float, min_score: float, duplicate_iou: float) -> bool:
        rank = cand.meta.get("v87b_rank") or cand.meta.get("v87_rank") or {}
        score = float(rank.get("score") or cand.score)
        ratio = page_area_ratio(cand.box, width, height)
        if ratio > max_ratio or score < min_score:
            return False
        if sum(page_area_ratio(prev.box, width, height) for prev in selected) + ratio > area_budget:
            return False
        if any(box_iou(cand.box, prev.box) >= duplicate_iou for prev in selected):
            return False
        selected.append(cand)
        per_family[cand.family] += 1
        return len(selected) >= top_k

    for cand in ranked:
        if cand.family != "token":
            continue
        if per_family["token"] >= 8:
            continue
        if add_if_ok(cand, max_ratio=0.018, min_score=6.5, duplicate_iou=0.55):
            return selected

    for cand in ranked:
        if cand.family != "evidence" or "span_id" not in cand.source:
            continue
        if per_family["evidence"] >= 5:
            continue
        if add_if_ok(cand, max_ratio=0.035, min_score=6.2, duplicate_iou=0.58):
            return selected

    for cand in ranked:
        if cand.family != "linegrid":
            continue
        if per_family["linegrid"] >= 6:
            continue
        # Line-grid anchors are recall insurance for weakly localized text
        # lines, so admit slightly lower scores than exact token matches while
        # keeping them compact and non-duplicative.
        if add_if_ok(cand, max_ratio=0.020, min_score=5.6, duplicate_iou=0.50):
            return selected

    fallback = choose_topk_v87mix(row, candidates, top_k, width, height)
    selected = append_nonoverlapping(selected, fallback, top_k=top_k, duplicate_iou=0.60)
    if len(selected) < top_k:
        selected = append_nonoverlapping(selected, ranked, top_k=top_k, duplicate_iou=0.68)
    return selected


def _rank_score(candidate: RecallCandidate) -> float:
    rank = candidate.meta.get("v87b_rank") or candidate.meta.get("v87_rank") or {}
    return float(rank.get("score") or candidate.score)


def _spatial_bin(candidate: RecallCandidate, width: int, height: int, cols: int = 4, rows: int = 8) -> tuple[int, int]:
    cx = (candidate.box[0] + candidate.box[2]) / 2.0
    cy = (candidate.box[1] + candidate.box[3]) / 2.0
    col = max(0, min(cols - 1, int(cx / max(width, 1) * cols)))
    row = max(0, min(rows - 1, int(cy / max(height, 1) * rows)))
    return col, row


def _candidate_ok_for_diverse(candidate: RecallCandidate, width: int, height: int) -> bool:
    ratio = page_area_ratio(candidate.box, width, height)
    if candidate.family == "existing":
        return False
    if ratio > 0.070 and candidate.family not in {"grid", "row"}:
        return False
    if ratio > 0.095:
        return False
    if area(candidate.box) < 20:
        return False
    return True


def choose_topk_v137diverse(row: dict[str, Any], candidates: list[RecallCandidate], top_k: int, width: int, height: int) -> list[RecallCandidate]:
    """MMR-style GT-blind rerank over the exhaustive pool.

    v136 showed that the all-candidate pool covers most GT regions but top-k
    ranking collapses onto repeated template instances.  This selector keeps
    the existing relevance model, then explicitly trades relevance for spatial
    and family diversity so the first few appended boxes cover different OCR
    rows/page regions.
    """
    _ = choose_topk_v87b(row, candidates, max(top_k, 24), width, height)
    ranked = sorted(
        [c for c in candidates if _candidate_ok_for_diverse(c, width, height)],
        key=lambda c: (_rank_score(c), -page_area_ratio(c.box, width, height)),
        reverse=True,
    )
    selected: list[RecallCandidate] = []
    per_family: Counter[str] = Counter()
    per_group: Counter[str] = Counter()
    per_bin: Counter[tuple[int, int]] = Counter()
    area_budget = 0.18
    family_caps = {
        "token": 5,
        "linegrid": 5,
        "ocr": 5,
        "evidence": 4,
        "row": 3,
        "grid": 2,
        "paragraph": 1,
        "patch": 1,
    }

    def can_take(cand: RecallCandidate, *, seed: bool = False) -> bool:
        ratio = page_area_ratio(cand.box, width, height)
        if per_family[cand.family] >= family_caps.get(cand.family, 2):
            return False
        group = candidate_group_key(cand)
        if per_group[group] >= (1 if not seed else 2):
            return False
        if any(box_iou(cand.box, prev.box) >= 0.52 for prev in selected):
            return False
        if sum(page_area_ratio(prev.box, width, height) for prev in selected) + ratio > area_budget:
            return False
        return True

    def take(cand: RecallCandidate) -> None:
        cand.meta["v137diverse_rank"] = {
            **(cand.meta.get("v87b_rank") or cand.meta.get("v87_rank") or {}),
            "score": _rank_score(cand),
            "spatial_bin": list(_spatial_bin(cand, width, height)),
            "group_key": candidate_group_key(cand),
        }
        selected.append(cand)
        per_family[cand.family] += 1
        per_group[candidate_group_key(cand)] += 1
        per_bin[_spatial_bin(cand, width, height)] += 1

    # Seed with the strongest compact text/evidence anchor; these have the
    # best precision and keep report count growth controlled.
    for cand in ranked:
        if cand.family not in {"token", "linegrid", "ocr", "evidence"}:
            continue
        if _rank_score(cand) < 5.2:
            continue
        if page_area_ratio(cand.box, width, height) > 0.040:
            continue
        if can_take(cand, seed=True):
            take(cand)
            break

    if not selected and ranked:
        take(ranked[0])

    while len(selected) < top_k:
        best: RecallCandidate | None = None
        best_score = -1e9
        for cand in ranked:
            if cand in selected or not can_take(cand):
                continue
            ratio = page_area_ratio(cand.box, width, height)
            same_bin = per_bin[_spatial_bin(cand, width, height)] > 0
            same_family = per_family[cand.family] > 0
            same_group = per_group[candidate_group_key(cand)] > 0
            max_iou = max((box_iou(cand.box, prev.box) for prev in selected), default=0.0)
            score = _rank_score(cand)
            # MMR: high relevance, low overlap, different page bin, and varied
            # family.  Grid/row are allowed as late recall insurance but not as
            # first-class precision anchors.
            mmr = score
            mmr -= 4.5 * max_iou
            mmr -= 1.25 if same_bin else 0.0
            mmr -= 0.65 if same_family else 0.0
            mmr -= 1.10 if same_group else 0.0
            mmr -= min(2.5, ratio * 25.0)
            if cand.family in {"ocr", "linegrid", "evidence"}:
                mmr += 0.35
            if cand.family == "grid":
                mmr -= 0.45
            if mmr > best_score:
                best = cand
                best_score = mmr
        if best is None:
            break
        best.meta["v137diverse_mmr"] = {
            "mmr": best_score,
            "rank_score": _rank_score(best),
            "spatial_bin": list(_spatial_bin(best, width, height)),
        }
        take(best)

    if len(selected) < top_k:
        selected = append_nonoverlapping(selected, ranked, top_k=top_k, duplicate_iou=0.62)
    return selected


def choose_topk_v139hybrid(row: dict[str, Any], candidates: list[RecallCandidate], top_k: int, width: int, height: int) -> list[RecallCandidate]:
    """Blend v137 spatial diversity with one precision text anchor.

    v137 fixed repeated-instance collapse, but failure analysis showed several
    regressions when two broad stage-evidence boxes displaced the compact
    token/linegrid anchors that v136 had selected.  This hybrid keeps v137's
    first diverse anchor and reserves one early slot for the best compact text
    candidate, then fills the rest with v137 non-overlapping candidates.
    """
    diverse = choose_topk_v137diverse(row, candidates, max(top_k, 24), width, height)
    # Ensure v87b scores exist for all candidates, then reuse the precision
    # ordering that v89token used successfully before diversity reranking.
    _ = choose_topk_v87b(row, candidates, max(top_k, 24), width, height)
    ranked = sorted(
        [c for c in candidates if _candidate_ok_for_diverse(c, width, height)],
        key=lambda c: (_rank_score(c), -page_area_ratio(c.box, width, height)),
        reverse=True,
    )
    precision_families = {"token", "linegrid", "ocr"}
    selected: list[RecallCandidate] = []

    def add(cand: RecallCandidate, duplicate_iou: float = 0.54) -> bool:
        if cand in selected:
            return False
        if any(box_iou(cand.box, prev.box) >= duplicate_iou for prev in selected):
            return False
        selected.append(cand)
        return True

    # Keep a strong diverse anchor if available; it often catches large style
    # or rendering artifacts missed by token-only ranking.
    for cand in diverse:
        if cand.family in {"evidence", "linegrid", "ocr", "token"}:
            add(cand, duplicate_iou=0.60)
            break
    if not selected and diverse:
        add(diverse[0], duplicate_iou=0.60)

    # Reserve an early slot for one compact text-derived anchor.  This is the
    # rollback mechanism for evidence-only selections without needing GT.
    has_precision = any(c.family in precision_families for c in selected)
    if not has_precision or (top_k >= 2 and len(selected) < 2):
        for cand in ranked:
            if cand.family not in precision_families:
                continue
            if page_area_ratio(cand.box, width, height) > 0.045:
                continue
            if _rank_score(cand) < 5.0 and cand.family != "linegrid":
                continue
            if add(cand, duplicate_iou=0.50):
                break

    selected = append_nonoverlapping(selected, diverse, top_k=top_k, duplicate_iou=0.55)
    if len(selected) < top_k:
        selected = append_nonoverlapping(selected, ranked, top_k=top_k, duplicate_iou=0.62)
    for cand in selected:
        cand.meta["v139hybrid_rank"] = {
            **(cand.meta.get("v137diverse_rank") or cand.meta.get("v87b_rank") or cand.meta.get("v87_rank") or {}),
            "score": _rank_score(cand),
            "selected_family": cand.family,
        }
    return selected[:top_k]


def choose_topk_v143dense(
    row: dict[str, Any],
    candidates: list[RecallCandidate],
    top_k: int,
    width: int,
    height: int,
    *,
    language_code: str = "",
) -> list[RecallCandidate]:
    """Language-aware ranking for dense scripts.

    Failure analysis of v142 showed that ar/th low-S_Loc cases are not missing
    candidates: whole OCR spans cover GT well, but token/subspan projections are
    ranked too early despite weak oracle coverage.  This selector keeps v137 for
    other languages, and for ar/th shifts the first slots toward physical OCR
    spans/rows/line windows that are less dependent on character-to-position
    ordering.
    """
    language = str(
        language_code
        or row.get("language_code")
        or ((row.get("metadata") or {}).get("language_code"))
        or ""
    ).lower()
    if language not in {"ar", "th"}:
        return choose_topk_v137diverse(row, candidates, top_k, width, height)

    _ = choose_topk_v87b(row, candidates, max(top_k, 24), width, height)
    ranked = [c for c in candidates if _candidate_ok_for_diverse(c, width, height)]

    def dense_score(cand: RecallCandidate) -> float:
        score = _rank_score(cand)
        source = cand.source or ""
        ratio = page_area_ratio(cand.box, width, height)
        if source.startswith("ocr_span:"):
            score += 2.2
        if cand.family == "row":
            score += 1.2
        if source.startswith("ocr_linegrid"):
            score += 0.8
        if source.startswith("ocr_scriptgrid"):
            score += 0.15
        if cand.family == "token":
            score -= 2.6
        if cand.family == "grid":
            score -= 0.20
        if source.startswith("stage_evidence"):
            score += 0.45
        if ratio > 0.055 and cand.family not in {"grid", "row"}:
            score -= min(2.4, ratio * 18.0)
        cand.meta["v143dense_rank"] = {
            **(cand.meta.get("v87b_rank") or cand.meta.get("v87_rank") or {}),
            "score": score,
            "dense_language": language,
            "selected_family": cand.family,
            "source": source,
        }
        return score

    ranked.sort(key=lambda c: (dense_score(c), -page_area_ratio(c.box, width, height)), reverse=True)
    selected: list[RecallCandidate] = []
    per_family: Counter[str] = Counter()
    per_group: Counter[str] = Counter()
    per_bin: Counter[tuple[int, int]] = Counter()
    family_caps = {
        "ocr": 6,
        "row": 4,
        "linegrid": 5,
        "evidence": 4,
        "token": 2,
        "grid": 2,
        "paragraph": 1,
        "patch": 1,
    }
    area_budget = 0.20

    def can_take(cand: RecallCandidate, *, seed: bool = False) -> bool:
        ratio = page_area_ratio(cand.box, width, height)
        if per_family[cand.family] >= family_caps.get(cand.family, 2):
            return False
        group = candidate_group_key(cand)
        if per_group[group] >= (2 if seed and cand.family in {"ocr", "row", "linegrid"} else 1):
            return False
        if any(box_iou(cand.box, prev.box) >= 0.54 for prev in selected):
            return False
        if sum(page_area_ratio(prev.box, width, height) for prev in selected) + ratio > area_budget:
            return False
        return True

    def take(cand: RecallCandidate) -> None:
        selected.append(cand)
        per_family[cand.family] += 1
        per_group[candidate_group_key(cand)] += 1
        per_bin[_spatial_bin(cand, width, height)] += 1

    # Force the first anchor away from token projections when possible.
    for cand in ranked:
        if cand.family not in {"ocr", "row", "linegrid", "evidence"}:
            continue
        if dense_score(cand) < 4.8:
            continue
        if can_take(cand, seed=True):
            take(cand)
            break
    if not selected and ranked:
        take(ranked[0])

    while len(selected) < top_k:
        best: RecallCandidate | None = None
        best_score = -1e9
        for cand in ranked:
            if cand in selected or not can_take(cand):
                continue
            ratio = page_area_ratio(cand.box, width, height)
            max_iou = max((box_iou(cand.box, prev.box) for prev in selected), default=0.0)
            same_bin = per_bin[_spatial_bin(cand, width, height)] > 0
            same_family = per_family[cand.family] > 0
            score = dense_score(cand)
            mmr = score
            mmr -= 4.2 * max_iou
            mmr -= 1.0 if same_bin else 0.0
            mmr -= 0.55 if same_family else 0.0
            mmr -= min(2.4, ratio * 22.0)
            if cand.family in {"ocr", "row", "linegrid"}:
                mmr += 0.35
            if cand.family == "token":
                mmr -= 0.75
            if mmr > best_score:
                best = cand
                best_score = mmr
        if best is None:
            break
        best.meta["v143dense_mmr"] = {
            "mmr": best_score,
            "dense_score": dense_score(best),
            "spatial_bin": list(_spatial_bin(best, width, height)),
        }
        take(best)

    if len(selected) < top_k:
        selected = append_nonoverlapping(selected, ranked, top_k=top_k, duplicate_iou=0.62)
    return selected


def choose_topk_v145ocranchor(
    row: dict[str, Any],
    candidates: list[RecallCandidate],
    top_k: int,
    width: int,
    height: int,
    *,
    language_code: str = "",
) -> list[RecallCandidate]:
    """Recall-first selector seeded by physical OCR spans.

    v144 residual diagnostics showed a consistent failure across languages:
    token/subspan boxes dominate the first two slots, while whole OCR spans have
    much stronger oracle coverage on the remaining low-S_Loc samples.  This
    selector explicitly seeds with OCR-span/row/line anchors, then applies the
    same MMR-style diversity constraints to avoid repeated template instances.
    """
    language = str(
        language_code
        or row.get("language_code")
        or ((row.get("metadata") or {}).get("language_code"))
        or ""
    ).lower()
    _ = choose_topk_v87b(row, candidates, max(top_k, 24), width, height)
    ranked = [c for c in candidates if _candidate_ok_for_diverse(c, width, height)]

    def anchor_score(cand: RecallCandidate) -> float:
        score = _rank_score(cand)
        source = cand.source or ""
        ratio = page_area_ratio(cand.box, width, height)
        if source.startswith("ocr_span:"):
            score += 2.8
        if cand.family == "row":
            score += 1.5
        if source.startswith("ocr_linegrid"):
            score += 1.0
        if source.startswith("ocr_scriptgrid"):
            score += 0.35 if language in {"ar", "th"} else 0.05
        if cand.family == "token":
            score -= 3.2
        if source.startswith("stage_evidence"):
            score += 0.35
        if cand.family == "grid":
            score -= 0.35
        if ratio > 0.060 and cand.family not in {"grid", "row"}:
            score -= min(3.0, ratio * 20.0)
        cand.meta["v145ocranchor_rank"] = {
            **(cand.meta.get("v87b_rank") or cand.meta.get("v87_rank") or {}),
            "score": score,
            "language": language,
            "source": source,
            "selected_family": cand.family,
        }
        return score

    ranked.sort(key=lambda c: (anchor_score(c), -page_area_ratio(c.box, width, height)), reverse=True)
    selected: list[RecallCandidate] = []
    per_family: Counter[str] = Counter()
    per_group: Counter[str] = Counter()
    per_bin: Counter[tuple[int, int]] = Counter()
    family_caps = {
        "ocr": 7,
        "row": 5,
        "linegrid": 6,
        "evidence": 3,
        "token": 1,
        "grid": 2,
        "paragraph": 1,
        "patch": 1,
    }
    area_budget = 0.22

    def can_take(cand: RecallCandidate, *, seed: bool = False) -> bool:
        ratio = page_area_ratio(cand.box, width, height)
        if per_family[cand.family] >= family_caps.get(cand.family, 2):
            return False
        group = candidate_group_key(cand)
        group_cap = 2 if seed and cand.family in {"ocr", "row", "linegrid"} else 1
        if per_group[group] >= group_cap:
            return False
        if any(box_iou(cand.box, prev.box) >= 0.54 for prev in selected):
            return False
        if sum(page_area_ratio(prev.box, width, height) for prev in selected) + ratio > area_budget:
            return False
        return True

    def take(cand: RecallCandidate) -> None:
        selected.append(cand)
        per_family[cand.family] += 1
        per_group[candidate_group_key(cand)] += 1
        per_bin[_spatial_bin(cand, width, height)] += 1

    # First slot: prefer physical OCR anchors over projected token boxes.
    for cand in ranked:
        if cand.family not in {"ocr", "row", "linegrid"}:
            continue
        if anchor_score(cand) < 4.7:
            continue
        if page_area_ratio(cand.box, width, height) > 0.065 and cand.family != "row":
            continue
        if can_take(cand, seed=True):
            take(cand)
            break
    if not selected and ranked:
        take(ranked[0])

    while len(selected) < top_k:
        best: RecallCandidate | None = None
        best_score = -1e9
        for cand in ranked:
            if cand in selected or not can_take(cand):
                continue
            ratio = page_area_ratio(cand.box, width, height)
            max_iou = max((box_iou(cand.box, prev.box) for prev in selected), default=0.0)
            same_bin = per_bin[_spatial_bin(cand, width, height)] > 0
            same_family = per_family[cand.family] > 0
            score = anchor_score(cand)
            mmr = score
            mmr -= 4.0 * max_iou
            mmr -= 0.95 if same_bin else 0.0
            mmr -= 0.50 if same_family else 0.0
            mmr -= min(2.5, ratio * 20.0)
            if cand.family in {"ocr", "row", "linegrid"}:
                mmr += 0.30
            if cand.family == "token":
                mmr -= 0.90
            if mmr > best_score:
                best = cand
                best_score = mmr
        if best is None:
            break
        best.meta["v145ocranchor_mmr"] = {
            "mmr": best_score,
            "anchor_score": anchor_score(best),
            "spatial_bin": list(_spatial_bin(best, width, height)),
        }
        take(best)

    if len(selected) < top_k:
        selected = append_nonoverlapping(selected, ranked, top_k=top_k, duplicate_iou=0.62)
    return selected


def choose_topk_v446physdiverse(
    row: dict[str, Any],
    candidates: list[RecallCandidate],
    top_k: int,
    width: int,
    height: int,
    *,
    language_code: str = "",
) -> list[RecallCandidate]:
    """Physical OCR-span coverage selector for dense-script low-loc samples.

    v446 coverage diagnostics on th/ms showed that the exhaustive pool covers
    GT well, while top-k ranking collapses around semantically salient but
    spatially wrong text.  This selector is still GT-free: it shifts priority
    toward physical OCR spans, line windows, and row boxes, and forces vertical
    diversity before token projections are admitted.
    """
    language = str(
        language_code
        or row.get("language_code")
        or ((row.get("metadata") or {}).get("language_code"))
        or ""
    ).lower()
    if language not in {"th", "ms", "ar"}:
        return choose_topk_v145ocranchor(row, candidates, top_k, width, height, language_code=language)

    _ = choose_topk_v87b(row, candidates, max(top_k, 32), width, height)
    pool = [c for c in candidates if _candidate_ok_for_diverse(c, width, height)]

    def phys_score(cand: RecallCandidate) -> float:
        score = _rank_score(cand)
        source = cand.source or ""
        ratio = page_area_ratio(cand.box, width, height)
        if source.startswith("ocr_span:"):
            score += 3.5
        if source.startswith("ocr_linegrid"):
            score += 2.0
        if source.startswith("ocr_scriptgrid"):
            score += 1.6
        if cand.family == "row":
            score += 1.8
        if source.startswith("stage_evidence_span_id"):
            score += 0.8
        if cand.family == "token":
            score -= 4.0
        if cand.family == "existing":
            score -= 2.2
        if cand.family == "grid":
            score -= 0.6
        if ratio > 0.070 and cand.family not in {"row", "grid"}:
            score -= min(3.0, ratio * 18.0)
        cand.meta["v446physdiverse_rank"] = {
            **(cand.meta.get("v87b_rank") or cand.meta.get("v87_rank") or {}),
            "score": score,
            "language": language,
            "source": source,
            "selected_family": cand.family,
            "spatial_bin": list(_spatial_bin(cand, width, height)),
        }
        return score

    ranked = sorted(pool, key=lambda c: (phys_score(c), -page_area_ratio(c.box, width, height)), reverse=True)
    selected: list[RecallCandidate] = []
    per_family: Counter[str] = Counter()
    per_group: Counter[str] = Counter()
    per_bin: Counter[tuple[int, int]] = Counter()
    family_caps = {
        "ocr": 8,
        "linegrid": 8,
        "row": 5,
        "evidence": 4,
        "grid": 3,
        "token": 1,
        "paragraph": 1,
        "patch": 1,
    }
    area_budget = 0.26

    def can_take(cand: RecallCandidate, *, seed: bool = False) -> bool:
        ratio = page_area_ratio(cand.box, width, height)
        if per_family[cand.family] >= family_caps.get(cand.family, 2):
            return False
        group = candidate_group_key(cand)
        group_cap = 2 if seed and cand.family in {"ocr", "linegrid", "row"} else 1
        if per_group[group] >= group_cap:
            return False
        if any(box_iou(cand.box, prev.box) >= 0.50 for prev in selected):
            return False
        if sum(page_area_ratio(prev.box, width, height) for prev in selected) + ratio > area_budget:
            return False
        return True

    def take(cand: RecallCandidate) -> None:
        selected.append(cand)
        per_family[cand.family] += 1
        per_group[candidate_group_key(cand)] += 1
        per_bin[_spatial_bin(cand, width, height)] += 1

    # Seed with up to three physical OCR anchors in different vertical bands.
    for cand in ranked:
        if cand.family not in {"ocr", "linegrid", "row"}:
            continue
        if phys_score(cand) < 4.8:
            continue
        if per_bin[_spatial_bin(cand, width, height)] > 0:
            continue
        if can_take(cand, seed=True):
            take(cand)
        if len(selected) >= min(3, top_k):
            break

    if not selected and ranked:
        take(ranked[0])

    while len(selected) < top_k:
        best: RecallCandidate | None = None
        best_score = -1e9
        for cand in ranked:
            if cand in selected or not can_take(cand):
                continue
            ratio = page_area_ratio(cand.box, width, height)
            max_iou = max((box_iou(cand.box, prev.box) for prev in selected), default=0.0)
            bin_key = _spatial_bin(cand, width, height)
            same_bin = per_bin[bin_key] > 0
            same_family = per_family[cand.family] > 0
            same_group = per_group[candidate_group_key(cand)] > 0
            score = phys_score(cand)
            mmr = score
            mmr -= 4.8 * max_iou
            mmr -= 1.6 if same_bin else 0.0
            mmr -= 0.55 if same_family else 0.0
            mmr -= 1.15 if same_group else 0.0
            mmr -= min(2.8, ratio * 20.0)
            if cand.family in {"ocr", "linegrid", "row"}:
                mmr += 0.5
            if cand.family == "grid":
                mmr -= 0.4
            if cand.family == "token":
                mmr -= 1.2
            if mmr > best_score:
                best = cand
                best_score = mmr
        if best is None:
            break
        best.meta["v446physdiverse_mmr"] = {
            "mmr": best_score,
            "phys_score": phys_score(best),
            "spatial_bin": list(_spatial_bin(best, width, height)),
        }
        take(best)

    if len(selected) < top_k:
        selected = append_nonoverlapping(selected, ranked, top_k=top_k, duplicate_iou=0.60)
    return selected


def coverage_for_candidates(
    *,
    gt_boxes: list[list[int]],
    gt_mask: np.ndarray | None,
    candidates: list[RecallCandidate],
    width: int,
    height: int,
) -> dict[str, Any]:
    boxes = [c.box for c in candidates]
    out: dict[str, Any] = {}
    out.update(mask_stats(gt_mask, boxes, width, height))
    out.update(box_coverage(gt_boxes, boxes))
    out["candidate_count"] = len(candidates)
    out["candidate_area_ratio_sum"] = sum(page_area_ratio(b, width, height) for b in boxes)
    return out


def _script_bucket(text: str) -> str:
    counts: dict[str, int] = {"zh": 0, "th": 0, "ar": 0, "latin": 0}
    for ch in text or "":
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            counts["zh"] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["th"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["ar"] += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            counts["latin"] += 1
    bucket, count = max(counts.items(), key=lambda item: item[1])
    return bucket if count > 0 else "latin"


def _extra_reason_text(cand: dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", str(cand.get("text") or "")).strip()
    clipped = text[:220]
    bucket = _script_bucket(text)
    if bucket == "zh":
        if clipped:
            return (
                f"补充定位区域来自OCR行和局部网格检索，相关文本片段为：{clipped}。"
                "该区域与已报告的数字、日期或文本篡改线索相关，因此作为分散异常的补充定位框保留。"
            )
        return "补充定位区域来自OCR行和局部网格检索，作为分散异常的补充定位框保留。"
    if bucket == "th":
        if clipped:
            return (
                "\u0e01\u0e23\u0e2d\u0e1a\u0e15\u0e33\u0e41\u0e2b\u0e19\u0e48\u0e07\u0e40\u0e1e\u0e34\u0e48\u0e21\u0e40\u0e15\u0e34\u0e21\u0e19\u0e35\u0e49\u0e21\u0e32\u0e08\u0e32\u0e01 OCR "
                f"\u0e41\u0e25\u0e30\u0e01\u0e32\u0e23\u0e04\u0e49\u0e19\u0e2b\u0e32\u0e41\u0e1a\u0e1a\u0e01\u0e23\u0e34\u0e14 \u0e02\u0e49\u0e2d\u0e04\u0e27\u0e32\u0e21\u0e17\u0e35\u0e48\u0e40\u0e01\u0e35\u0e48\u0e22\u0e27\u0e02\u0e49\u0e2d\u0e07: {clipped} "
                "\u0e08\u0e36\u0e07\u0e40\u0e01\u0e47\u0e1a\u0e44\u0e27\u0e49\u0e40\u0e1b\u0e47\u0e19\u0e01\u0e23\u0e2d\u0e1a\u0e15\u0e33\u0e41\u0e2b\u0e19\u0e48\u0e07\u0e40\u0e2a\u0e23\u0e34\u0e21\u0e02\u0e2d\u0e07\u0e08\u0e38\u0e14\u0e1c\u0e34\u0e14\u0e1b\u0e01\u0e15\u0e34"
            )
        return (
            "\u0e01\u0e23\u0e2d\u0e1a\u0e15\u0e33\u0e41\u0e2b\u0e19\u0e48\u0e07\u0e40\u0e1e\u0e34\u0e48\u0e21\u0e40\u0e15\u0e34\u0e21\u0e19\u0e35\u0e49\u0e21\u0e32\u0e08\u0e32\u0e01 OCR "
            "\u0e41\u0e25\u0e30\u0e01\u0e32\u0e23\u0e04\u0e49\u0e19\u0e2b\u0e32\u0e41\u0e1a\u0e1a\u0e01\u0e23\u0e34\u0e14 \u0e08\u0e36\u0e07\u0e40\u0e01\u0e47\u0e1a\u0e44\u0e27\u0e49\u0e40\u0e1b\u0e47\u0e19\u0e01\u0e23\u0e2d\u0e1a\u0e40\u0e2a\u0e23\u0e34\u0e21"
        )
    if bucket == "ar":
        if clipped:
            return (
                "\u062a\u0645 \u062a\u062d\u062f\u064a\u062f \u0647\u0630\u0647 \u0627\u0644\u0645\u0646\u0637\u0642\u0629 \u0628\u0648\u0627\u0633\u0637\u0629 OCR "
                f"\u0648\u0628\u062d\u062b \u0634\u0628\u0643\u064a \u0645\u062d\u0644\u064a\u060c \u0648\u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0631\u062a\u0628\u0637 \u0647\u0648: {clipped}. "
                "\u0644\u0630\u0644\u0643 \u062a\u064f\u062d\u0641\u0638 \u0643\u0625\u0637\u0627\u0631 \u062a\u0645\u0648\u0636\u0639 \u0625\u0636\u0627\u0641\u064a \u0644\u0623\u062b\u0631 \u062a\u0644\u0627\u0639\u0628 \u0645\u062a\u0641\u0631\u0642."
            )
        return (
            "\u062a\u0645 \u062a\u062d\u062f\u064a\u062f \u0647\u0630\u0647 \u0627\u0644\u0645\u0646\u0637\u0642\u0629 \u0628\u0648\u0627\u0633\u0637\u0629 OCR "
            "\u0648\u0628\u062d\u062b \u0634\u0628\u0643\u064a \u0645\u062d\u0644\u064a\u060c \u0644\u0630\u0644\u0643 \u062a\u064f\u062d\u0641\u0638 \u0643\u0625\u0637\u0627\u0631 \u062a\u0645\u0648\u0636\u0639 \u0625\u0636\u0627\u0641\u064a."
        )
    evidence = f"Additional localization candidate from OCR line and local grid search."
    if clipped:
        evidence += f" Related OCR/evidence text: {clipped}"
    evidence += " This box is retained to cover a dispersed tamper region already indicated by the report."
    return evidence


def insert_extra_anomalies(report: str, extras: list[dict[str, Any]]) -> str:
    if not extras:
        return report
    block: list[str] = []
    for idx, cand in enumerate(extras, start=1):
        box = cand["box"]
        source = str(cand.get("source") or "")
        family = str(cand.get("family") or "")
        evidence = _extra_reason_text(cand)
        block.extend(
            [
                f"### ANOMALY_V87_EXTRA_{idx:03d}: Exhaustive Recall Candidate ({family})",
                f"[GROUNDING]:{box}",
                f"[REASON]: {evidence}",
                "",
            ]
        )
    extra_text = "\n".join(block)
    marker = re.search(r"\n\s*-{3,}\s*\n\s*##\s*SUMMARY|\n\s*##\s*SUMMARY", report, re.IGNORECASE)
    if marker:
        return report[: marker.start()] + "\n\n" + extra_text + report[marker.start():]
    return report.rstrip() + "\n\n" + extra_text


def replace_groundings(report: str, replacements: dict[int, list[int]]) -> tuple[str, int]:
    if not replacements:
        return report, 0
    idx = 0
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx, count
        cur_idx = idx
        idx += 1
        box = replacements.get(cur_idx)
        if not box:
            return match.group(0)
        count += 1
        return f"[GROUNDING]:{box}"

    return GROUNDING_RE.sub(repl, report), count


def select_apply_extras(
    top_candidates: list[dict[str, Any]],
    existing_boxes: list[list[int]],
    *,
    width: int,
    height: int,
    limit: int,
    max_area_ratio: float,
    duplicate_iou: float,
) -> list[dict[str, Any]]:
    extras: list[dict[str, Any]] = []
    occupied = list(existing_boxes)
    for cand in top_candidates:
        if cand.get("family") == "existing" or str(cand.get("source") or "").startswith("current_final_report"):
            continue
        box = cand.get("box")
        if not isinstance(box, list) or len(box) < 4:
            continue
        clamped = clamp_box([float(v) for v in box[:4]], width, height)
        if not clamped:
            continue
        if page_area_ratio(clamped, width, height) > max_area_ratio:
            continue
        if any(box_iou(clamped, prev) >= duplicate_iou for prev in occupied):
            continue
        out = dict(cand)
        out["box"] = clamped
        extras.append(out)
        occupied.append(clamped)
        if len(extras) >= limit:
            break
    return extras


def select_apply_replacements(
    top_candidates: list[dict[str, Any]],
    existing_boxes: list[list[int]],
    *,
    width: int,
    height: int,
    limit: int,
    max_area_ratio: float,
    duplicate_iou: float,
    replace_policy: str,
) -> dict[int, dict[str, Any]]:
    if not existing_boxes:
        return {}
    replacements: dict[int, dict[str, Any]] = {}
    occupied = list(existing_boxes)
    support_boxes: list[list[int]] = []
    for support_cand in top_candidates:
        if support_cand.get("family") == "existing" or str(support_cand.get("source") or "").startswith("current_final_report"):
            continue
        raw_box = support_cand.get("box")
        if not isinstance(raw_box, list) or len(raw_box) < 4:
            continue
        support_box = clamp_box([float(v) for v in raw_box[:4]], width, height)
        if not support_box:
            continue
        if page_area_ratio(support_box, width, height) > max_area_ratio:
            continue
        support_boxes.append(support_box)

    def support_iou(existing_box: list[int]) -> float:
        return max((box_iou(existing_box, support_box) for support_box in support_boxes), default=0.0)

    def choose_index(box: list[int]) -> int | None:
        available = [idx for idx in range(len(existing_boxes)) if idx not in replacements]
        if not available:
            return None
        if replace_policy == "largest":
            return max(available, key=lambda idx: area(existing_boxes[idx]))
        if replace_policy == "large_unmatched":
            # Replace large outlier boxes, but protect existing boxes already
            # spatially supported by the current OCR-anchor candidate set.
            unmatched = [idx for idx in available if support_iou(existing_boxes[idx]) <= 0.12]
            if unmatched:
                return max(unmatched, key=lambda idx: area(existing_boxes[idx]))
            return max(
                available,
                key=lambda idx: page_area_ratio(existing_boxes[idx], width, height) - 0.035 * support_iou(existing_boxes[idx]),
            )
        if replace_policy == "closest":
            return max(available, key=lambda idx: box_iou(box, existing_boxes[idx]))
        # Default: remove the currently least-supported spatial outlier with
        # respect to the proposed OCR anchor.
        return min(available, key=lambda idx: box_iou(box, existing_boxes[idx]))

    for cand in top_candidates:
        if cand.get("family") == "existing" or str(cand.get("source") or "").startswith("current_final_report"):
            continue
        box = cand.get("box")
        if not isinstance(box, list) or len(box) < 4:
            continue
        clamped = clamp_box([float(v) for v in box[:4]], width, height)
        if not clamped:
            continue
        if page_area_ratio(clamped, width, height) > max_area_ratio:
            continue
        replace_idx = choose_index(clamped)
        if replace_idx is None:
            break
        if any(
            idx != replace_idx and idx not in replacements and box_iou(clamped, prev) >= duplicate_iou
            for idx, prev in enumerate(occupied)
        ):
            continue
        out = dict(cand)
        out["box"] = clamped
        out["replace_index"] = replace_idx
        out["replaced_box"] = existing_boxes[replace_idx]
        replacements[replace_idx] = out
        occupied[replace_idx] = clamped
        if len(replacements) >= limit:
            break
    return replacements


def family_coverage(
    *,
    gt_boxes: list[list[int]],
    gt_mask: np.ndarray | None,
    candidates: list[RecallCandidate],
    width: int,
    height: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_family: dict[str, list[RecallCandidate]] = defaultdict(list)
    for cand in candidates:
        by_family[cand.family].append(cand)
    for family, rows in sorted(by_family.items()):
        out[family] = coverage_for_candidates(
            gt_boxes=gt_boxes,
            gt_mask=gt_mask,
            candidates=rows,
            width=width,
            height=height,
        )
    return out


def process_sample(
    row: dict[str, Any],
    *,
    gt_row: dict[str, Any],
    eval_sample: dict[str, Any] | None,
    debug_root: Path,
    ocr_layout_cache_dir: Path,
    ocr_layout_model: str,
    coord_mode: str,
    max_candidates: int,
    top_k: int,
    rank_mode: str,
    enable_token_candidates: bool = False,
    enable_linegrid_candidates: bool = False,
    enable_scriptgrid_candidates: bool = False,
) -> dict[str, Any]:
    image_path = resolve_image_path(row, debug_root)
    if not image_path:
        raise RuntimeError(f"missing image for {sample_id_from_row(row)}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    language_code = str((eval_sample or {}).get("language_code") or "")
    candidates = generate_candidates(
        row,
        image,
        debug_root,
        ocr_layout_cache_dir,
        ocr_layout_model,
        coord_mode,
        max_candidates,
        enable_token_candidates=enable_token_candidates,
        enable_linegrid_candidates=enable_linegrid_candidates,
        enable_scriptgrid_candidates=enable_scriptgrid_candidates,
        language_code=language_code,
    )
    if rank_mode == "v139hybrid":
        top = choose_topk_v139hybrid(row, candidates, top_k, width, height)
    elif rank_mode == "v145ocranchor":
        top = choose_topk_v145ocranchor(row, candidates, top_k, width, height, language_code=language_code)
    elif rank_mode == "v446physdiverse":
        top = choose_topk_v446physdiverse(row, candidates, top_k, width, height, language_code=language_code)
    elif rank_mode == "v143dense":
        top = choose_topk_v143dense(row, candidates, top_k, width, height, language_code=language_code)
    elif rank_mode == "v137diverse":
        top = choose_topk_v137diverse(row, candidates, top_k, width, height)
    elif rank_mode == "v89token":
        top = choose_topk_v89token(row, candidates, top_k, width, height)
    elif rank_mode == "v87mix":
        top = choose_topk_v87mix(row, candidates, top_k, width, height)
    elif rank_mode == "v87b":
        top = choose_topk_v87b(row, candidates, top_k, width, height)
    elif rank_mode == "v87":
        top = choose_topk_v87(row, candidates, top_k, width, height)
    else:
        top = choose_topk_diverse(candidates, top_k)
    gt_report = str(gt_row.get("report_text") or gt_row.get("report") or "")
    gt_boxes = parse_grounding_boxes(gt_report, width, height)
    gt_mask = read_gt_mask(gt_row, width, height, debug_root)
    all_cov = coverage_for_candidates(gt_boxes=gt_boxes, gt_mask=gt_mask, candidates=candidates, width=width, height=height)
    top_cov = coverage_for_candidates(gt_boxes=gt_boxes, gt_mask=gt_mask, candidates=top, width=width, height=height)
    by_family = family_coverage(gt_boxes=gt_boxes, gt_mask=gt_mask, candidates=candidates, width=width, height=height)
    source_counts = Counter(c.family for c in candidates)
    return {
        "sample_id": sample_id_from_row(row),
        "image_name": row.get("image_name"),
        "width": width,
        "height": height,
        "baseline": {
            "loc_score": (eval_sample or {}).get("loc_score"),
            "pred_boxes": (eval_sample or {}).get("pred_boxes"),
            "gt_boxes": (eval_sample or {}).get("gt_boxes"),
            "language_code": (eval_sample or {}).get("language_code") or gt_row.get("language_code"),
        },
        "candidate_count": len(candidates),
        "top_k": top_k,
        "rank_mode": rank_mode,
        "source_counts": dict(source_counts),
        "coverage_all": all_cov,
        "coverage_topk": top_cov,
        "coverage_by_family": by_family,
        "top_candidates": [asdict(c) for c in top],
        "candidates": [asdict(c) for c in candidates],
    }


def mean_numeric(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    vals: list[float] = []
    for row in rows:
        cur: Any = row
        for key in path:
            cur = cur.get(key) if isinstance(cur, dict) else None
        if cur is not None:
            vals.append(float(cur))
    return float(np.mean(vals)) if vals else 0.0


def summarize(rows: list[dict[str, Any]], *, name: str) -> dict[str, Any]:
    weak_all = [
        r["sample_id"]
        for r in rows
        if float(r["coverage_all"].get("mask_recall") or 0.0) < 0.20
        and float(r["coverage_all"].get("gt_recall_cover_0p3") or 0.0) < 0.30
    ]
    weak_top = [
        r["sample_id"]
        for r in rows
        if float(r["coverage_topk"].get("mask_recall") or 0.0) < 0.20
        and float(r["coverage_topk"].get("gt_recall_cover_0p3") or 0.0) < 0.30
    ]
    families = sorted({fam for row in rows for fam in row.get("coverage_by_family", {})})
    family_summary = {}
    for fam in families:
        sub = []
        for row in rows:
            cov = row.get("coverage_by_family", {}).get(fam)
            if cov:
                sub.append({"coverage_all": cov})
        family_summary[fam] = {
            "n": len(sub),
            "mask_recall": mean_numeric(sub, ("coverage_all", "mask_recall")),
            "mask_f1": mean_numeric(sub, ("coverage_all", "mask_f1")),
            "gt_recall_cover_0p3": mean_numeric(sub, ("coverage_all", "gt_recall_cover_0p3")),
            "mean_best_iou": mean_numeric(sub, ("coverage_all", "mean_best_iou")),
        }
    return {
        "name": name,
        "n": len(rows),
        "coverage_all": {
            "mask_recall": mean_numeric(rows, ("coverage_all", "mask_recall")),
            "mask_precision": mean_numeric(rows, ("coverage_all", "mask_precision")),
            "mask_f1": mean_numeric(rows, ("coverage_all", "mask_f1")),
            "mask_iou": mean_numeric(rows, ("coverage_all", "mask_iou")),
            "mean_best_iou": mean_numeric(rows, ("coverage_all", "mean_best_iou")),
            "gt_recall_iou_0p1": mean_numeric(rows, ("coverage_all", "gt_recall_iou_0p1")),
            "gt_recall_iou_0p3": mean_numeric(rows, ("coverage_all", "gt_recall_iou_0p3")),
            "gt_recall_cover_0p3": mean_numeric(rows, ("coverage_all", "gt_recall_cover_0p3")),
            "gt_recall_cover_0p5": mean_numeric(rows, ("coverage_all", "gt_recall_cover_0p5")),
            "candidate_count": mean_numeric(rows, ("coverage_all", "candidate_count")),
            "candidate_area_ratio_sum": mean_numeric(rows, ("coverage_all", "candidate_area_ratio_sum")),
        },
        "coverage_topk": {
            "mask_recall": mean_numeric(rows, ("coverage_topk", "mask_recall")),
            "mask_precision": mean_numeric(rows, ("coverage_topk", "mask_precision")),
            "mask_f1": mean_numeric(rows, ("coverage_topk", "mask_f1")),
            "mask_iou": mean_numeric(rows, ("coverage_topk", "mask_iou")),
            "mean_best_iou": mean_numeric(rows, ("coverage_topk", "mean_best_iou")),
            "gt_recall_iou_0p1": mean_numeric(rows, ("coverage_topk", "gt_recall_iou_0p1")),
            "gt_recall_iou_0p3": mean_numeric(rows, ("coverage_topk", "gt_recall_iou_0p3")),
            "gt_recall_cover_0p3": mean_numeric(rows, ("coverage_topk", "gt_recall_cover_0p3")),
            "gt_recall_cover_0p5": mean_numeric(rows, ("coverage_topk", "gt_recall_cover_0p5")),
            "candidate_count": mean_numeric(rows, ("coverage_topk", "candidate_count")),
            "candidate_area_ratio_sum": mean_numeric(rows, ("coverage_topk", "candidate_area_ratio_sum")),
        },
        "family_summary": family_summary,
        "weak_all_count": len(weak_all),
        "weak_all_sample_ids": weak_all[:30],
        "weak_topk_count": len(weak_top),
        "weak_topk_sample_ids": weak_top[:30],
        "interpretation": "GT is used only here for coverage diagnostics. If coverage_all is weak, candidate generation is the bottleneck; if coverage_all is high but topk is weak, ranking/fusion is the bottleneck; if both are high but final S_Loc remains weak, verifier/gating/report insertion is the bottleneck.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL))
    parser.add_argument("--gt-jsonl", default=str(DEFAULT_GT))
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--select-low-loc-limit", type=int, default=50)
    parser.add_argument("--low-loc-threshold", type=float, default=0.02)
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel", "auto"], default="normalized-1000")
    parser.add_argument("--ocr-layout-cache-dir", default=str(DEFAULT_OCR_LAYOUT_CACHE))
    parser.add_argument("--ocr-layout-model", default="qwen-vl-ocr")
    parser.add_argument("--max-candidates", type=int, default=900)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--rank-mode", choices=["v86", "v87", "v87b", "v87mix", "v89token", "v137diverse", "v139hybrid", "v143dense", "v145ocranchor", "v446physdiverse"], default="v86")
    parser.add_argument("--enable-token-candidates", action="store_true", help="Add OCR token/subspan candidates projected inside OCR line boxes.")
    parser.add_argument("--enable-linegrid-candidates", action="store_true", help="Add OCR line internal sliding-window candidates for weak word segmentation.")
    parser.add_argument("--enable-scriptgrid-candidates", action="store_true", help="Add ar/th physical OCR-line slice candidates independent of character order.")
    parser.add_argument("--applied-raw-jsonl", default="", help="Optional raw JSONL with selected top candidates appended to reports.")
    parser.add_argument("--apply-mode", choices=["append", "replace"], default="append", help="Append new anomaly boxes or replace existing [GROUNDING] boxes.")
    parser.add_argument("--apply-top-n", type=int, default=8)
    parser.add_argument("--apply-max-area-ratio", type=float, default=0.06)
    parser.add_argument("--apply-duplicate-iou", type=float, default=0.50)
    parser.add_argument("--replace-policy", choices=["largest", "large_unmatched", "closest", "farthest"], default="largest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    input_path = resolve_pipe_path(args.input_jsonl)
    eval_path = resolve_pipe_path(args.eval_json)
    gt_path = Path(args.gt_jsonl).expanduser()
    if not gt_path.is_absolute():
        gt_path = debug_root / gt_path
    output_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = {sample_id_from_row(r): r for r in read_jsonl(input_path)}
    eval_samples = load_eval_samples(eval_path)
    gt_rows = {str(r.get("sample_id") or Path(str(r.get("image_file") or "")).stem): r for r in read_jsonl(gt_path)}
    selected = list(args.sample_id or [])
    if args.select_low_loc_limit > 0:
        for sid in load_eval_selection(eval_path, threshold=args.low_loc_threshold, limit=args.select_low_loc_limit):
            if sid not in selected:
                selected.append(sid)
    if not selected:
        selected = [sid for sid, s in eval_samples.items() if s.get("gt_label") == "FORGED" and s.get("pred_label") == "FORGED"]

    rows: list[dict[str, Any]] = []
    results_by_sample: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    with output_path.open("w", encoding="utf-8") as out_fh:
        for sid in selected:
            row = raw_rows.get(sid)
            gt_row = gt_rows.get(sid)
            if not row or not gt_row:
                missing.append(sid)
                continue
            result = process_sample(
                row,
                gt_row=gt_row,
                eval_sample=eval_samples.get(sid),
                debug_root=debug_root,
                ocr_layout_cache_dir=Path(args.ocr_layout_cache_dir).expanduser().resolve(),
                ocr_layout_model=args.ocr_layout_model,
                coord_mode=args.coord_mode,
                max_candidates=args.max_candidates,
                top_k=args.top_k,
                rank_mode=args.rank_mode,
                enable_token_candidates=args.enable_token_candidates,
                enable_linegrid_candidates=args.enable_linegrid_candidates,
                enable_scriptgrid_candidates=args.enable_scriptgrid_candidates,
            )
            rows.append(result)
            results_by_sample[sid] = result
            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    run_name = f"qwen_pipe_{args.rank_mode}_exhaustive_recall_local_lowloc{args.select_low_loc_limit}"
    summary = summarize(rows, name=run_name)
    summary["missing_sample_ids"] = missing
    summary["input_jsonl"] = str(input_path)
    summary["eval_json"] = str(eval_path)
    summary["gt_jsonl"] = str(gt_path)
    summary["low_loc_threshold"] = args.low_loc_threshold
    summary["selected_count_requested"] = len(selected)
    summary["rank_mode"] = args.rank_mode
    summary["enable_token_candidates"] = bool(args.enable_token_candidates)
    summary["enable_linegrid_candidates"] = bool(args.enable_linegrid_candidates)
    summary["enable_scriptgrid_candidates"] = bool(args.enable_scriptgrid_candidates)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.applied_raw_jsonl:
        setup_debug_import(debug_root)
        from postprocess import parse_cct_report  # type: ignore

        applied_path = resolve_pipe_path(args.applied_raw_jsonl)
        applied_path.parent.mkdir(parents=True, exist_ok=True)
        apply_stats = {
            "rows": 0,
            "changed": 0,
            "boxes_added": 0,
            "rank_mode": args.rank_mode,
            "apply_mode": args.apply_mode,
            "apply_top_n": args.apply_top_n,
            "apply_max_area_ratio": args.apply_max_area_ratio,
            "diagnostic_selection_count": len(results_by_sample),
        }
        with input_path.open("r", encoding="utf-8") as src, applied_path.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                apply_stats["rows"] += 1
                sid = sample_id_from_row(row)
                result = results_by_sample.get(sid)
                if result:
                    width = int(result.get("width") or row.get("width") or 0)
                    height = int(result.get("height") or row.get("height") or 0)
                    existing = report_boxes(str(row.get("raw_output") or ""))
                    selected_boxes: list[dict[str, Any]] = []
                    if args.apply_mode == "replace":
                        replacements = select_apply_replacements(
                            result.get("top_candidates") or [],
                            existing,
                            width=width,
                            height=height,
                            limit=args.apply_top_n,
                            max_area_ratio=args.apply_max_area_ratio,
                            duplicate_iou=args.apply_duplicate_iou,
                            replace_policy=args.replace_policy,
                        )
                        if replacements:
                            row = dict(row)
                            new_report, replaced_count = replace_groundings(
                                str(row.get("raw_output") or ""),
                                {idx: cand["box"] for idx, cand in replacements.items()},
                            )
                            row["raw_output"] = new_report
                            row["parsed"] = parse_cct_report(new_report)
                            selected_boxes = list(replacements.values())
                            stage_outputs = dict(row.get("stage_outputs") or {})
                            stage_outputs["qwen_pipe_exhaustive_recall_apply"] = {
                                "applied": True,
                                "rank_mode": args.rank_mode,
                                "apply_mode": args.apply_mode,
                                "boxes_replaced": replaced_count,
                                "selected": selected_boxes,
                                "policy": "Replace existing [GROUNDING] boxes with GT-blind ranked OCR/evidence/patch/grid candidates. GT was not used to rank or select boxes; low-loc selection is diagnostic when enabled.",
                            }
                            row["stage_outputs"] = stage_outputs
                            apply_stats["changed"] += 1
                            apply_stats["boxes_added"] += replaced_count
                    else:
                        extras = select_apply_extras(
                            result.get("top_candidates") or [],
                            existing,
                            width=width,
                            height=height,
                            limit=args.apply_top_n,
                            max_area_ratio=args.apply_max_area_ratio,
                            duplicate_iou=args.apply_duplicate_iou,
                        )
                        selected_boxes = extras
                    if args.apply_mode == "append" and selected_boxes:
                        row = dict(row)
                        new_report = insert_extra_anomalies(str(row.get("raw_output") or ""), selected_boxes)
                        row["raw_output"] = new_report
                        row["parsed"] = parse_cct_report(new_report)
                        stage_outputs = dict(row.get("stage_outputs") or {})
                        stage_outputs["qwen_pipe_exhaustive_recall_apply"] = {
                            "applied": True,
                            "rank_mode": args.rank_mode,
                            "apply_mode": args.apply_mode,
                            "boxes_added": len(selected_boxes),
                            "selected": selected_boxes,
                            "policy": "Append GT-blind ranked OCR/evidence/patch/grid candidates from exhaustive recall. GT was not used to rank or select boxes; low-loc selection is diagnostic when enabled.",
                        }
                        row["stage_outputs"] = stage_outputs
                        apply_stats["changed"] += 1
                        apply_stats["boxes_added"] += len(selected_boxes)
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["applied_raw_jsonl"] = str(applied_path)
        summary["apply_stats"] = apply_stats
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
