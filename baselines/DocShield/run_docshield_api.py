#!/usr/bin/env python3
"""DashScope API runner for GenText-Forensics.

Calls Qwen-VL models via Aliyun Bailian OpenAI-compatible API.
Output schema matches run_docshield_proxy.py so downstream tools work unchanged.

Usage:
    python baselines/DocShield/run_docshield_api.py \
        --model qwen3-vl-flash \
        --prompt-mode cct \
        --input-jsonl data/splits/debug_50.jsonl \
        --output-jsonl outputs/raw/api_flash_cct/debug_50.jsonl \
        --api-key-file api-key.txt \
        --max-tokens 8192

    # Simple prompt (QWEN_PROMPT from gentext_agent.py):
    python baselines/DocShield/run_docshield_api.py \
        --model qwen3-vl-30b-a3b-instruct \
        --prompt-mode simple \
        --input-jsonl data/splits/val_100.jsonl \
        --output-jsonl outputs/raw/api_30b_simple/val_100.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOC_DIR))
sys.path.insert(0, str(REPO_ROOT / "gentext-forensics" / "scripts"))

from cct_prompt import SYSTEM_PROMPT as CCT_SYSTEM, USER_PROMPT as CCT_USER  # noqa: E402
from postprocess import to_pipeline_record  # noqa: E402

SIMPLE_PROMPT = """You are a document forensics examiner. Analyze this text-rich image and determine if it is FORGED or AUTHENTIC.

IMPORTANT: Most documents are authentic. Only classify as FORGED if you find CLEAR, CONCRETE evidence of tampering such as:
- Visually inconsistent text (different font, color, blur, artifacts at boundaries)
- Logically impossible content (contradictory numbers, impossible dates)
- Obvious copy-paste artifacts (misaligned baselines, repeated patterns)

If the document looks normal with consistent formatting and logical content, classify as AUTHENTIC.

Write your analysis as a Markdown report following this structure:

# FORGERY ANALYSIS  REPORT

**Report ID:** FAR-xxxx-xx-xx
**Date of Examination:** xxxx-xx-xx
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** <FORGED or AUTHENTIC>
    **[RISK_SCORE]:** <0-100>

---

## DETAILED ANOMALY ANALYSIS

<If FORGED, for each anomaly write:>

### ANOMALY_001: <category> (<location>)
[GROUNDING]:[x1, y1, x2, y2]
[REASON]: <Detailed evidence in the document's language, at least 200 chars>

<If AUTHENTIC, write: "No anomalies detected. The document has been thoroughly examined and no signs of tampering, alteration, or forgery were found.">

---

## SUMMARY
The examination of the document has identified <N> anomalies, resulting in a fraud risk score of <score>. <Describe the document content in detail, using the document's language.>

---
**END OF REPORT**

Categories: visual_clumsy, logical_fraud, semantic_subtle, text_tampering
Use pixel coordinates for [GROUNDING]. Write REASON and SUMMARY in the same language as the document text."""


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def image_to_data_url(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_messages(image_path: str, prompt_mode: str) -> list[dict[str, Any]]:
    data_url = image_to_data_url(image_path)

    if prompt_mode == "cct":
        return [
            {"role": "system", "content": CCT_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": CCT_USER},
                ],
            },
        ]
    else:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": SIMPLE_PROMPT},
                ],
            },
        ]


def call_api(
    messages: list[dict],
    model: str,
    api_key: str,
    max_tokens: int = 8192,
    temperature: float = 0,
    enable_thinking: bool = False,
    timeout: int = 180,
) -> tuple[str, dict]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not enable_thinking:
        payload["enable_thinking"] = False

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    return content, usage


def process_row(
    row: dict[str, Any],
    model: str,
    api_key: str,
    prompt_mode: str,
    max_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    image_name = row.get("image_file") or Path(str(row.get("image_path") or "")).name
    image_path = row.get("image_path")
    gold_label = row.get("label")
    language_code = row.get("language_code")

    t0 = time.time()
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            width, height = im.size

        messages = build_messages(image_path, prompt_mode)
        raw_content, usage = call_api(
            messages, model, api_key,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

        record = to_pipeline_record(
            sample_id=sample_id,
            image_name=image_name,
            image_path=image_path,
            width=width,
            height=height,
            gold_label=gold_label,
            language_code=language_code,
            model_id=model,
            raw_full=raw_content,
            elapsed_sec=round(time.time() - t0, 3),
        )
        record["usage"] = usage
        record["api_raw_content"] = raw_content  # Debug: keep original API response

    except Exception as exc:
        record = to_pipeline_record(
            sample_id=sample_id,
            image_name=image_name,
            image_path=image_path,
            width=None,
            height=None,
            gold_label=gold_label,
            language_code=language_code,
            model_id=model,
            raw_full="",
            elapsed_sec=round(time.time() - t0, 3),
            error=repr(exc),
        )

    return record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="DashScope model name (e.g. qwen3-vl-flash)")
    p.add_argument("--prompt-mode", default="simple", choices=["simple", "cct"],
                   help="Prompt strategy: simple (QWEN_PROMPT) or cct (DocShield 6-stage)")
    p.add_argument("--input-jsonl", required=True, help="Path to split JSONL")
    p.add_argument("--output-jsonl", required=True, help="Output JSONL path")
    p.add_argument("--api-key-file", default="api-key.txt", help="Path to API key file")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--enable-thinking", action="store_true", help="Enable model thinking (costs more tokens)")
    p.add_argument("--num-workers", type=int, default=1, help="Parallel workers for API calls")
    p.add_argument("--resume", action="store_true", help="Skip rows already in output")
    p.add_argument("--timeout", type=int, default=180, help="API timeout per request (seconds)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load API key
    key_path = Path(args.api_key_file)
    if not key_path.exists():
        key_path = REPO_ROOT / "api-key.txt"
    if not key_path.exists():
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("Error: No API key found. Provide --api-key-file or set DASHSCOPE_API_KEY")
            sys.exit(1)
    else:
        api_key = key_path.read_text(encoding="utf-8").strip()

    # Load data
    rows = read_jsonl(Path(args.input_jsonl))
    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_keys: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    key = rec.get("sample_id") or rec.get("image_name")
                    if key:
                        done_keys.add(key)
                except json.JSONDecodeError:
                    pass
        print(f"[resume] skipping {len(done_keys)} already processed rows")

    filtered = []
    for row in rows:
        key = row.get("sample_id") or row.get("image_file") or row.get("image_path")
        if key and key in done_keys:
            continue
        filtered.append(row)

    if not filtered:
        print("[run_docshield_api] nothing to do — all rows processed")
        return

    print(f"[run_docshield_api] model={args.model} prompt={args.prompt_mode} rows={len(filtered)} workers={args.num_workers}")

    mode = "a" if args.resume and out_path.exists() else "w"
    written = 0
    errors = 0

    from tqdm import tqdm

    if args.num_workers <= 1:
        with out_path.open(mode, encoding="utf-8") as fout:
            pbar = tqdm(filtered, desc=f"{args.model} ({args.prompt_mode})", unit="img")
            for row in pbar:
                record = process_row(row, args.model, api_key, args.prompt_mode, args.max_tokens, args.enable_thinking)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                if "error" in record:
                    errors += 1
                pred = record.get("parsed", {}).get("conclusion", "?")
                pbar.set_postfix(pred=pred, err=errors, t=f"{record.get('elapsed_sec', 0):.0f}s")
    else:
        with out_path.open(mode, encoding="utf-8") as fout:
            pbar = tqdm(total=len(filtered), desc=f"{args.model} ({args.prompt_mode}) ×{args.num_workers}", unit="img")
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {
                    executor.submit(process_row, row, args.model, api_key, args.prompt_mode, args.max_tokens, args.enable_thinking): row
                    for row in filtered
                }
                for future in as_completed(futures):
                    record = future.result()
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    written += 1
                    if "error" in record:
                        errors += 1
                    pred = record.get("parsed", {}).get("conclusion", "?")
                    pbar.update(1)
                    pbar.set_postfix(pred=pred, err=errors)
            pbar.close()

    print(f"\n[run_docshield_api] done: {written} rows, {errors} errors → {out_path}")


if __name__ == "__main__":
    main()
