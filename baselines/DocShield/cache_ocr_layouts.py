#!/usr/bin/env python3
"""Precompute Qwen-OCR text boxes for local grounding diagnostics.

The cache contains only OCR text and coordinates extracted from images. It does
not read labels, GT reports, masks, or other evaluation-only fields.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

DOC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOC_DIR))

from run_staged_docshield_api import (  # noqa: E402
    ensure_data_available,
    ensure_sample_assets_available,
    load_api_key,
    parse_json_object,
    read_jsonl,
    resolve_repo_path,
    run_text_stage,
    safe_cache_name,
)
from staged_prompts import ocr_layout_only_prompt  # noqa: E402


DASHSCOPE_NATIVE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", default="data/val_300.jsonl")
    p.add_argument("--model", default="qwen-vl-ocr")
    p.add_argument("--api-key-file", default="/Users/penpen/Desktop/api-key.txt")
    p.add_argument("--cache-dir", default="outputs/cache/ocr_layouts")
    p.add_argument(
        "--api-mode",
        choices=["openai-prompt", "dashscope-native"],
        default="openai-prompt",
        help="Use prompt-based OpenAI-compatible OCR, or DashScope native OCR options.",
    )
    p.add_argument("--ocr-task", default="advanced_recognition", help="DashScope native OCR task.")
    p.add_argument("--min-pixels", type=int, default=32 * 32 * 3)
    p.add_argument("--max-pixels", type=int, default=32 * 32 * 8192)
    p.add_argument("--enable-rotate", action="store_true")
    p.add_argument("--sample-id", action="append", default=[], help="Only cache matching sample_id values.")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all rows after filtering.")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("image_file") or row.get("image_path") or "unknown")


def image_name_for_row(row: dict[str, Any]) -> str:
    return str(row.get("image_file") or Path(str(row.get("image_path") or "")).name)


def layout_cache_path(cache_dir: Path, model: str, key: str) -> Path:
    return cache_dir / safe_cache_name(model) / f"{safe_cache_name(key)}.json"


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def clamp_box(box: list[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def convert_numbers(nums: list[float], width: int, height: int, source: str) -> tuple[list[int] | None, str]:
    if len(nums) >= 5:
        cx, cy, size_a, size_b, angle = nums[:5]
        normalized_angle = abs(angle) % 180
        if 45 <= normalized_angle <= 135:
            box_w, box_h = size_b, size_a
        else:
            box_w, box_h = size_a, size_b
        return (
            clamp_box([cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0], width, height),
            "cxcy_size_angle",
        )
    a, b, c, d = nums[:4]
    if "center" in source or "cx" in source:
        h, w = c, d
        return clamp_box([a - w / 2.0, b - h / 2.0, a + w / 2.0, b + h / 2.0], width, height), "cxcyhw"
    if "xyxy" in source or (c > a and d > b and (c - a) > 2 and (d - b) > 2):
        return clamp_box([a, b, c, d], width, height), "xyxy"
    if c <= max(80, height * 0.08) and d <= width:
        h, w = c, d
        return clamp_box([a - w / 2.0, b - h / 2.0, a + w / 2.0, b + h / 2.0], width, height), "cxcyhw"
    return clamp_box([a, b, a + c, b + d], width, height), "xywh"


def numbers_from_value(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", value)]
    if isinstance(value, list):
        nums: list[float] = []
        for part in value:
            nums.extend(numbers_from_value(part))
        return nums
    return []


def parse_text_line(item: Any, idx: int, width: int, height: int, box_format: str) -> dict[str, Any] | None:
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("content") or "")
        raw_box = item.get("bbox") or item.get("box")
        item_format = str(item.get("box_format") or box_format or "")
        if raw_box is None:
            return None
        try:
            nums = numbers_from_value(raw_box)[:5]
        except (TypeError, ValueError):
            return None
        if len(nums) < 4:
            return None
        box, detected_format = convert_numbers(nums, width, height, item_format)
    elif isinstance(item, list):
        if isinstance(item[0], str):
            if len(item) >= 2:
                text = str(item[0])
                raw_nums = item[1:6]
            else:
                text = ""
                raw_nums = item[:1]
        else:
            text = ""
            raw_nums = item[:5]
        try:
            nums = numbers_from_value(raw_nums)[:5]
        except (TypeError, ValueError):
            return None
        if len(nums) < 4:
            return None
        box, detected_format = convert_numbers(nums, width, height, box_format)
    else:
        return None
    if not box:
        return None
    return {
        "id": f"q{idx}",
        "text": text,
        "bbox": box,
        "detected_box_format": detected_format,
        "confidence": None,
    }


def parse_ocr_layout(raw: str, width: int, height: int) -> dict[str, Any]:
    data, error = parse_json_object(raw)
    if not isinstance(data, dict):
        data = {}
    box_format = str(data.get("box_format") or data.get("bbox_format") or data.get("coordinate_format") or "")
    text_lines = data.get("text_lines") or data.get("lines") or data.get("ocr") or []
    if not isinstance(text_lines, list) or not text_lines:
        text_lines = []
        pattern = re.compile(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
            r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        )
        for match in pattern.finditer(raw):
            text_lines.append([float(v) for v in match.groups()])
    spans = []
    for idx, item in enumerate(text_lines, start=1):
        span = parse_text_line(item, idx, width, height, box_format)
        if span:
            spans.append(span)
    return {
        "coordinate_system": data.get("coordinate_system") or data.get("coordinates") or "",
        "box_format": box_format,
        "text_spans": spans,
        "raw_line_count": len(text_lines) if isinstance(text_lines, list) else 0,
        "_parse_error": error,
    }


def box_from_location(location: Any, width: int, height: int) -> list[int] | None:
    nums = numbers_from_value(location)
    if len(nums) < 8:
        return None
    xs = nums[0::2][:4]
    ys = nums[1::2][:4]
    return clamp_box([min(xs), min(ys), max(xs), max(ys)], width, height)


def parse_native_ocr_result(result: dict[str, Any], width: int, height: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    choice = ((result.get("output") or {}).get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or [{}])[0]
    raw = str(content.get("text") or "")
    ocr_result = content.get("ocr_result") or {}
    words_info = ocr_result.get("words_info") or []
    spans: list[dict[str, Any]] = []
    if isinstance(words_info, list):
        for idx, item in enumerate(words_info, start=1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "")
            box = box_from_location(item.get("location"), width, height)
            detected_format = "location"
            if not box:
                nums = numbers_from_value(item.get("rotate_rect"))[:5]
                if len(nums) >= 4:
                    box, detected_format = convert_numbers(nums, width, height, "rotate_rect")
            if not box:
                continue
            spans.append(
                {
                    "id": f"q{idx}",
                    "text": text,
                    "bbox": box,
                    "detected_box_format": detected_format,
                    "confidence": None,
                    "location": item.get("location"),
                    "rotate_rect": item.get("rotate_rect"),
                }
            )
    parsed = {
        "coordinate_system": "native_pixel",
        "box_format": "location_or_rotate_rect",
        "text_spans": spans,
        "raw_line_count": len(words_info) if isinstance(words_info, list) else 0,
        "_parse_error": None,
    }
    usage = result.get("usage") or {}
    return raw, parsed, usage


def call_dashscope_native_ocr(
    *,
    image_path: Path,
    model: str,
    api_key: str,
    ocr_task: str,
    min_pixels: int,
    max_pixels: int,
    enable_rotate: bool,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": image_to_data_url(image_path),
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                            "enable_rotate": enable_rotate,
                        }
                    ],
                }
            ]
        },
        "parameters": {"ocr_options": {"task": ocr_task}},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        DASHSCOPE_NATIVE_URL,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def process_row(
    row: dict[str, Any],
    *,
    model: str,
    api_key: str,
    cache_dir: Path,
    api_mode: str,
    ocr_task: str,
    min_pixels: int,
    max_pixels: int,
    enable_rotate: bool,
    max_tokens: int,
    timeout: int,
    overwrite: bool,
) -> dict[str, Any]:
    from PIL import Image

    key = row_key(row)
    image_name = image_name_for_row(row)
    image_path = resolve_repo_path(str(row.get("image_path") or ""))
    cache_path = layout_cache_path(cache_dir, model, key)
    if cache_path.exists() and not overwrite:
        return {"sample_id": key, "status": "cached", "cache_path": str(cache_path)}

    with Image.open(image_path) as im:
        width, height = im.size

    native_response: dict[str, Any] | None = None
    if api_mode == "dashscope-native":
        native_response = call_dashscope_native_ocr(
            image_path=image_path,
            model=model,
            api_key=api_key,
            ocr_task=ocr_task,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            enable_rotate=enable_rotate,
            timeout=timeout,
        )
        raw, parsed, usage = parse_native_ocr_result(native_response, width, height)
    else:
        raw, usage = run_text_stage(
            prompt=ocr_layout_only_prompt(image_name, width, height),
            image_path=image_path,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=0.01,
            timeout=timeout,
        )
        parsed = parse_ocr_layout(raw, width, height)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "sample_id": row.get("sample_id"),
                "image_name": image_name,
                "model": model,
                "api_mode": api_mode,
                "ocr_task": ocr_task if api_mode == "dashscope-native" else "",
                "width": width,
                "height": height,
                "raw": raw,
                "parsed": parsed,
                "usage": usage,
                "native_response": native_response,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"sample_id": key, "status": "written", "cache_path": str(cache_path)}


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_repo_path(args.input_jsonl)
    ensure_data_available(input_jsonl)
    rows = read_jsonl(input_jsonl)
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [row for row in rows if row_key(row) in wanted]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    ensure_sample_assets_available(rows)

    api_key = load_api_key(args.api_key_file)
    cache_dir = resolve_repo_path(args.cache_dir)
    kwargs = {
        "model": args.model,
        "api_key": api_key,
        "cache_dir": cache_dir,
        "api_mode": args.api_mode,
        "ocr_task": args.ocr_task,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "enable_rotate": args.enable_rotate,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "overwrite": args.overwrite,
    }

    written = cached = errors = 0
    if args.num_workers <= 1:
        iterator = (process_row(row, **kwargs) for row in rows)
        pbar = tqdm(iterator, total=len(rows), desc=f"{args.model} OCR layout", unit="img")
        for rec in pbar:
            written += rec["status"] == "written"
            cached += rec["status"] == "cached"
            pbar.set_postfix(written=written, cached=cached, err=errors)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_row, row, **kwargs): row for row in rows}
            pbar = tqdm(total=len(rows), desc=f"{args.model} OCR layout x{args.num_workers}", unit="img")
            for future in as_completed(futures):
                try:
                    rec = future.result()
                    written += rec["status"] == "written"
                    cached += rec["status"] == "cached"
                except Exception:
                    errors += 1
                pbar.update(1)
                pbar.set_postfix(written=written, cached=cached, err=errors)
            pbar.close()

    print(
        f"[cache_ocr_layouts] done: rows={len(rows)} written={written} cached={cached} "
        f"errors={errors} cache_dir={cache_dir}"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
