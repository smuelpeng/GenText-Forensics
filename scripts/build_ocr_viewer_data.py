#!/usr/bin/env python3
"""Build compact data for the local OCR/grounding coordinate viewer."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[([^\]]+)\]", re.IGNORECASE)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_key(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("image_name") or Path(str(row.get("image_path") or "")).stem)


def safe_cache_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe[:180] or "unknown"


def raw_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def clamp_box(value: list[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = value
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def detect_coord_space(boxes: list[list[float]], width: int, height: int, *, model_like: bool = False) -> str:
    """Infer whether model boxes are pixels or a 0-1000 visual grid.

    Qwen-VL often emits layout coordinates on a 0-1000 canvas even when the
    prompt asks for native pixels. GT report boxes and masks in this dataset are
    native pixels, so the viewer projects model-grid boxes before drawing them.
    """

    if not boxes:
        return "pixel"
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    if max_x <= 1.5 and max_y <= 1.5:
        return "normalized_0_1"
    if width > 1200 and height > 1200 and max_x <= 1100 and max_y <= 1100:
        return "normalized_1000"
    if model_like and width > 1600 and height > 1600 and max_x <= 1200 and max_y <= 1200:
        return "normalized_1000"
    if model_like and width > 1600 and height > 1600 and max_x <= 1600 and max_y <= 1250:
        return "normalized_1000"
    return "pixel"


def project_box(value: list[float], width: int, height: int, coord_space: str) -> list[int] | None:
    if coord_space == "normalized_0_1":
        value = [value[0] * width, value[1] * height, value[2] * width, value[3] * height]
    elif coord_space == "normalized_1000":
        value = [value[0] * width / 1000.0, value[1] * height / 1000.0, value[2] * width / 1000.0, value[3] * height / 1000.0]
    return clamp_box(value, width, height)


def parse_grounding_boxes(
    report: str,
    width: int,
    height: int,
    *,
    role: str,
    id_prefix: str,
    coord_space: str | None = None,
    model_like: bool = False,
) -> list[dict[str, Any]]:
    raw_boxes: list[list[float]] = []
    for match in GROUNDING_RE.finditer(report or ""):
        nums = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        if len(nums) < 4:
            continue
        box = raw_box([float(v) for v in nums[:4]])
        if box:
            raw_boxes.append(box)
    detected = coord_space or detect_coord_space(raw_boxes, width, height, model_like=model_like)
    boxes: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_boxes, start=1):
        box = project_box(raw, width, height, detected)
        if not box:
            continue
        boxes.append(
            {
                "id": f"{id_prefix}{idx}",
                "bbox": box,
                "raw_bbox": [round(v, 2) for v in raw],
                "coord_space": detected,
                "text": f"{role} {idx}",
                "role": role,
            }
        )
    return boxes


def parse_source_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip() or Path(path).stem, repo_path(path.strip())
    path = repo_path(value)
    return path.stem, path


def load_cache_text(cache_dir: Path, model: str, key: str) -> str:
    path = cache_dir / safe_cache_name(model) / f"{safe_cache_name(key)}.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(data.get("text") or data.get("raw") or "")


def load_gt_rows(gt_jsonl: Path | None) -> dict[str, dict[str, Any]]:
    if not gt_jsonl:
        return {}
    rows = read_jsonl(gt_jsonl)
    gt: dict[str, dict[str, Any]] = {}
    for row in rows:
        keys = {
            str(row.get("sample_id") or ""),
            Path(str(row.get("image_file") or row.get("image_name") or "")).stem,
            Path(str(row.get("image_path") or "")).stem,
        }
        for key in keys:
            if key:
                gt[key] = row
    return gt


def make_mask_overlay(mask_path: str, output_dir: Path, key: str) -> tuple[str, list[int] | None]:
    if not mask_path:
        return "", None
    path = repo_path(mask_path)
    if not path.exists():
        return "", None
    try:
        from PIL import Image

        mask = Image.open(path).convert("L")
        bbox = list(mask.getbbox() or [])
        alpha = mask.point(lambda p: 96 if p > 0 else 0)
        overlay = Image.new("RGBA", mask.size, (22, 128, 86, 0))
        overlay.putalpha(alpha)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{safe_cache_name(key)}_mask.png"
        overlay.save(out_path)
    except Exception:
        return "", None
    return "/" + str(out_path.relative_to(REPO_ROOT)), bbox or None


def build_source(
    label: str,
    path: Path,
    cache_dir: Path,
    cache_model: str,
    gt_rows: dict[str, dict[str, Any]],
    mask_output_dir: Path,
) -> dict[str, Any]:
    records = read_jsonl(path)
    samples: dict[str, Any] = {}
    for rec in records:
        key = sample_key(rec)
        width = int(rec.get("width") or 0)
        height = int(rec.get("height") or 0)
        stage_outputs = rec.get("stage_outputs") or {}
        ocr_layout = (stage_outputs.get("ocr_layout") or {}).get("parsed") or {}
        raw_spans: list[tuple[int, dict[str, Any], list[float]]] = []
        for idx, span in enumerate(ocr_layout.get("text_spans") or [], start=1):
            if isinstance(span, dict):
                box = raw_box(span.get("bbox"))
                if box:
                    raw_spans.append((idx, span, box))
        span_coord_space = detect_coord_space([box for _, _, box in raw_spans], width, height, model_like=True)
        spans: list[dict[str, Any]] = []
        invalid_spans = 0
        for idx, span, raw in raw_spans:
            box = project_box(raw, width, height, span_coord_space)
            if not box:
                invalid_spans += 1
                continue
            spans.append(
                {
                    "id": str(span.get("id") or f"s{idx}"),
                    "bbox": box,
                    "raw_bbox": [round(v, 2) for v in raw],
                    "coord_space": span_coord_space,
                    "text": str(span.get("text") or ""),
                    "role": str(span.get("role") or ""),
                    "confidence": span.get("confidence"),
                }
            )
        invalid_spans += len(ocr_layout.get("text_spans") or []) - len(raw_spans)
        transcript_stage = stage_outputs.get("ocr_transcript") or {}
        transcript = str(transcript_stage.get("text") or "") or load_cache_text(cache_dir, cache_model, key)
        report_raw = str((stage_outputs.get("report") or {}).get("raw") or rec.get("raw_output") or "")
        final_report = str(rec.get("raw_output") or "")
        gt_row = gt_rows.get(key) or gt_rows.get(Path(str(rec.get("image_name") or "")).stem) or {}
        gt_report = str(gt_row.get("report_text") or "")
        if not gt_report and gt_row.get("report_path"):
            report_path = repo_path(str(gt_row.get("report_path")))
            if report_path.exists():
                gt_report = report_path.read_text(encoding="utf-8")
        mask_url, mask_bbox = make_mask_overlay(str(gt_row.get("mask_path") or ""), mask_output_dir, key)
        samples[key] = {
            "sample_id": key,
            "image_name": rec.get("image_name"),
            "image_path": rec.get("image_path"),
            "image_url": "/" + str(rec.get("image_path") or "").lstrip("/"),
            "width": width,
            "height": height,
            "verdict": (rec.get("parsed") or {}).get("conclusion"),
            "risk_score": (rec.get("parsed") or {}).get("risk_score"),
            "document_language": ocr_layout.get("document_language"),
            "document_type": ocr_layout.get("document_type"),
            "global_summary": ocr_layout.get("global_summary"),
            "layout_model": (stage_outputs.get("ocr_layout") or {}).get("model"),
            "ocr_transcript_model": transcript_stage.get("model") or cache_model,
            "ocr_cache_hit": transcript_stage.get("cache_hit"),
            "transcript": transcript,
            "ocr_spans": spans,
            "grounding_boxes": parse_grounding_boxes(
                report_raw,
                width,
                height,
                role="pred_grounding",
                id_prefix="p",
                model_like=True,
            ),
            "final_grounding_boxes": parse_grounding_boxes(
                final_report,
                width,
                height,
                role="final_report",
                id_prefix="f",
                model_like=True,
            ),
            "gt_boxes": parse_grounding_boxes(
                gt_report,
                width,
                height,
                role="gt_report",
                id_prefix="t",
                coord_space="pixel",
            ),
            "gt_mask_url": mask_url,
            "gt_mask_bbox": mask_bbox,
            "gt_label": gt_row.get("label_codalab") or gt_row.get("label"),
            "invalid_spans": invalid_spans,
        }
    return {"label": label, "path": str(path.relative_to(REPO_ROOT)), "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-jsonl",
        action="append",
        required=True,
        help="Source JSONL. Use label=path to control display label. Can be repeated.",
    )
    parser.add_argument("--cache-dir", default="outputs/cache/ocr_transcripts")
    parser.add_argument("--cache-model", default="qwen-vl-ocr")
    parser.add_argument("--gt-jsonl", default="", help="Optional GT JSONL for local diagnostic overlays only.")
    parser.add_argument("--mask-output-dir", default="outputs/ocr_viewer/masks")
    parser.add_argument("--output", default="outputs/ocr_viewer/data.json")
    args = parser.parse_args()

    gt_rows = load_gt_rows(repo_path(args.gt_jsonl) if args.gt_jsonl else None)
    sources = [
        build_source(
            *parse_source_arg(v),
            repo_path(args.cache_dir),
            args.cache_model,
            gt_rows,
            repo_path(args.mask_output_dir),
        )
        for v in args.raw_jsonl
    ]
    sample_order = sorted({key for source in sources for key in source["samples"]})
    payload = {"sources": sources, "sample_order": sample_order}
    out_path = repo_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} with {len(sample_order)} samples across {len(sources)} sources")


if __name__ == "__main__":
    main()
