#!/usr/bin/env python3
"""Audit existing predicted grounding boxes with a Qwen crop verifier.

This is diagnostic by default: it checks whether each already-predicted
localization box contains visible hard manipulation evidence.  The verifier
sees only the crop and the current anomaly text, never GT labels/reports/masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_bad_box_prune import iter_grounding_entries  # noqa: E402
from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import (  # noqa: E402
    call_qwen_crop,
    expand_box,
    load_api_key,
    parse_json_object,
    resolve_image_path,
    resolve_pipe_path,
    setup_debug_import,
)


def compact(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def verifier_prompt(block_text: str, box: list[int], crop_size: tuple[int, int]) -> str:
    return f"""You are auditing one crop from a document-forensics prediction.

Only inspect the crop image. Decide whether this crop contains visible hard evidence of local manipulation or forgery.

Return YES only for visible local evidence such as: altered text with inconsistent font/blur/aliasing, broken or overlapping glyphs, redaction/block/smear, pasted numeric/date/amount field, abnormal table cell alignment that is visible in the crop, or clear image-compositing artifacts.

Return NO for ordinary text, normal tables, normal logos/photos, normal scan/compression noise, or claims that require outside semantic/world knowledge without a visible local artifact.

Current anomaly text, possibly noisy:
{compact(block_text)}

Original page box: {box}
Crop size: width={crop_size[0]}, height={crop_size[1]}

Return strict JSON only:
{{"verdict":"YES|NO","confidence":0.0,"evidence":"short reason based only on visible crop evidence"}}
"""


def cache_key(sample_id: str, index: int, box: list[int], model: str, prompt: str) -> str:
    data = {
        "sample_id": sample_id,
        "index": index,
        "box": box,
        "model": model,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--cache-dir", default=str(PIPE_ROOT / "outputs/cache/existing_box_crop_audit"))
    parser.add_argument("--api-key-file", default="/Users/penpen/Desktop/api-key.txt")
    parser.add_argument("--model", default="qwen-vl-max-latest")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--sample-id-file", default="", help="Optional newline-delimited sample ids.")
    parser.add_argument("--max-boxes-per-sample", type=int, default=3)
    parser.add_argument("--crop-pad", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    api_key = load_api_key(args.api_key_file)
    cache_dir = resolve_pipe_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = set(args.sample_id)
    if args.sample_id_file:
        wanted.update(
            line.strip()
            for line in resolve_pipe_path(args.sample_id_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    rows = []
    api_calls = 0
    verdict_counts: dict[str, int] = {}
    for row in read_jsonl(resolve_pipe_path(args.input_jsonl)):
        sid = sample_id_from_row(row)
        if wanted and sid not in wanted:
            continue
        image_path = resolve_image_path(row, debug_root)
        if not image_path:
            rows.append({"sample_id": sid, "error": "missing_image"})
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        entries = iter_grounding_entries(str(row.get("raw_output") or ""))[: args.max_boxes_per_sample]
        sample_rows = []
        for entry in entries:
            box = [int(v) for v in entry["box"]]
            crop_box = expand_box(box, width, height, args.crop_pad, args.crop_pad, min_pad=32)
            crop_dir = cache_dir / "crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{sid}_{entry['index']}.jpg"
            if not crop_path.exists():
                image.crop(tuple(crop_box)).save(crop_path, quality=92)
            prompt = verifier_prompt(str(entry.get("block_text") or ""), box, Image.open(crop_path).size)
            key = cache_key(sid, int(entry["index"]), box, args.model, prompt)
            response_path = cache_dir / "responses" / f"{key}.json"
            response_path.parent.mkdir(parents=True, exist_ok=True)
            if response_path.exists():
                payload = json.loads(response_path.read_text(encoding="utf-8"))
                payload["cache_hit"] = True
            else:
                raw, usage = call_qwen_crop(
                    crop_path=crop_path,
                    prompt=prompt,
                    model=args.model,
                    api_key=api_key,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                payload = {
                    "raw": raw,
                    "parsed": parse_json_object(raw),
                    "usage": usage,
                    "cache_hit": False,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                api_calls += 1
            parsed = payload.get("parsed") or {}
            verdict = str(parsed.get("verdict") or "UNKNOWN").upper()
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            sample_rows.append(
                {
                    "index": entry["index"],
                    "box": box,
                    "crop_box": crop_box,
                    "crop_path": str(crop_path),
                    "verdict": verdict,
                    "confidence": parsed.get("confidence"),
                    "evidence": parsed.get("evidence"),
                    "cache_hit": payload.get("cache_hit"),
                    "block_preview": compact(str(entry.get("block_text") or ""), 260),
                }
            )
        rows.append({"sample_id": sid, "boxes": sample_rows})

    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "output_jsonl": str(out_path),
        "sample_count": len(rows),
        "api_calls": api_calls,
        "model": args.model,
        "verdict_counts": verdict_counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
