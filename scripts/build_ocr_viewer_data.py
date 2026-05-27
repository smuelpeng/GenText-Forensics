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


def normalize_box(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_grounding_boxes(report: str, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for idx, match in enumerate(GROUNDING_RE.finditer(report or ""), start=1):
        nums = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        if len(nums) < 4:
            continue
        box = normalize_box([float(v) for v in nums[:4]], width, height)
        if not box:
            continue
        boxes.append({"id": f"g{idx}", "bbox": box, "text": f"GROUNDING {idx}", "role": "grounding"})
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


def build_source(label: str, path: Path, cache_dir: Path, cache_model: str) -> dict[str, Any]:
    records = read_jsonl(path)
    samples: dict[str, Any] = {}
    for rec in records:
        key = sample_key(rec)
        width = int(rec.get("width") or 0)
        height = int(rec.get("height") or 0)
        stage_outputs = rec.get("stage_outputs") or {}
        ocr_layout = (stage_outputs.get("ocr_layout") or {}).get("parsed") or {}
        spans: list[dict[str, Any]] = []
        invalid_spans = 0
        for idx, span in enumerate(ocr_layout.get("text_spans") or [], start=1):
            if not isinstance(span, dict):
                continue
            box = normalize_box(span.get("bbox"), width, height)
            if not box:
                invalid_spans += 1
                continue
            spans.append(
                {
                    "id": str(span.get("id") or f"s{idx}"),
                    "bbox": box,
                    "text": str(span.get("text") or ""),
                    "role": str(span.get("role") or ""),
                    "confidence": span.get("confidence"),
                }
            )
        transcript_stage = stage_outputs.get("ocr_transcript") or {}
        transcript = str(transcript_stage.get("text") or "") or load_cache_text(cache_dir, cache_model, key)
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
            "grounding_boxes": parse_grounding_boxes(rec.get("raw_output") or "", width, height),
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
    parser.add_argument("--output", default="outputs/ocr_viewer/data.json")
    args = parser.parse_args()

    sources = [build_source(*parse_source_arg(v), repo_path(args.cache_dir), args.cache_model) for v in args.raw_jsonl]
    sample_order = sorted({key for source in sources for key in source["samples"]})
    payload = {"sources": sources, "sample_order": sample_order}
    out_path = repo_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} with {len(sample_order)} samples across {len(sources)} sources")


if __name__ == "__main__":
    main()
