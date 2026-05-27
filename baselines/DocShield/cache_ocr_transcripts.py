#!/usr/bin/env python3
"""Precompute OCR transcripts for the staged DocShield baseline.

The cache contains only model-visible OCR text extracted from images. It does
not read labels, GT reports, masks, or other evaluation-only fields.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

DOC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOC_DIR))

from run_staged_docshield_api import (  # noqa: E402
    compact_text,
    ensure_data_available,
    ensure_sample_assets_available,
    load_api_key,
    read_jsonl,
    read_transcript_cache,
    resolve_repo_path,
    run_text_stage,
    transcript_cache_path,
    write_transcript_cache,
)
from staged_prompts import ocr_transcript_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", default="data/val_300.jsonl")
    p.add_argument("--model", default="qwen-vl-ocr")
    p.add_argument("--api-key-file", default="/Users/penpen/Desktop/api-key.txt")
    p.add_argument("--cache-dir", default="outputs/cache/ocr_transcripts")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--seed-jsonl",
        action="append",
        default=[],
        help="Optional staged output JSONL to import existing ocr_transcript stage outputs before calling the API.",
    )
    return p.parse_args()


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("image_file") or row.get("image_path") or "unknown")


def image_name_for_row(row: dict[str, Any]) -> str:
    return str(row.get("image_file") or Path(str(row.get("image_path") or "")).name)


def seed_cache(seed_jsonls: list[str], model: str, cache_dir: Path, max_chars: int) -> int:
    imported = 0
    for raw_path in seed_jsonls:
        path = resolve_repo_path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = ((rec.get("stage_outputs") or {}).get("ocr_transcript") or {})
                if stage.get("model") != model or not isinstance(stage.get("raw"), str):
                    continue
                sample_key = str(rec.get("sample_id") or rec.get("image_name") or "")
                if not sample_key:
                    continue
                cache_path = transcript_cache_path(cache_dir, model, sample_key)
                if cache_path.exists():
                    continue
                raw = str(stage.get("raw") or "")
                write_transcript_cache(
                    cache_path,
                    {
                        "sample_id": rec.get("sample_id"),
                        "image_name": rec.get("image_name"),
                        "model": model,
                        "raw": raw,
                        "text": compact_text(raw, max_chars),
                        "usage": {},
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "source": str(path),
                    },
                )
                imported += 1
    return imported


def process_row(
    row: dict[str, Any],
    *,
    model: str,
    api_key: str,
    cache_dir: Path,
    max_tokens: int,
    timeout: int,
    max_chars: int,
    overwrite: bool,
) -> dict[str, Any]:
    from PIL import Image

    key = row_key(row)
    image_name = image_name_for_row(row)
    image_path = resolve_repo_path(str(row.get("image_path") or ""))
    cache_path = transcript_cache_path(cache_dir, model, key)
    if cache_path.exists() and not overwrite:
        cached = read_transcript_cache(cache_path)
        if cached:
            return {"sample_id": key, "status": "cached", "cache_path": str(cache_path)}

    with Image.open(image_path) as im:
        width, height = im.size

    raw, usage = run_text_stage(
        prompt=ocr_transcript_prompt(image_name, width, height),
        image_path=image_path,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=0.01,
        timeout=timeout,
    )
    write_transcript_cache(
        cache_path,
        {
            "sample_id": row.get("sample_id"),
            "image_name": image_name,
            "model": model,
            "raw": raw,
            "text": compact_text(raw, max_chars),
            "usage": usage,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return {"sample_id": key, "status": "written", "cache_path": str(cache_path)}


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_repo_path(args.input_jsonl)
    ensure_data_available(input_jsonl)
    rows = read_jsonl(input_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    ensure_sample_assets_available(rows)

    cache_dir = resolve_repo_path(args.cache_dir)
    imported = seed_cache(args.seed_jsonl, args.model, cache_dir, args.max_chars)
    if imported:
        print(f"[cache_ocr_transcripts] imported {imported} transcripts from seed JSONL")

    api_key = load_api_key(args.api_key_file)
    written = 0
    cached = 0
    errors = 0

    kwargs = {
        "model": args.model,
        "api_key": api_key,
        "cache_dir": cache_dir,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_chars": args.max_chars,
        "overwrite": args.overwrite,
    }

    if args.num_workers <= 1:
        iterator = (process_row(row, **kwargs) for row in rows)
        pbar = tqdm(iterator, total=len(rows), desc=f"{args.model} OCR cache", unit="img")
        for rec in pbar:
            written += rec["status"] == "written"
            cached += rec["status"] == "cached"
            pbar.set_postfix(written=written, cached=cached, err=errors)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_row, row, **kwargs): row for row in rows}
            pbar = tqdm(total=len(rows), desc=f"{args.model} OCR cache x{args.num_workers}", unit="img")
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
        f"[cache_ocr_transcripts] done: rows={len(rows)} written={written} cached={cached} "
        f"errors={errors} cache_dir={cache_dir}"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
