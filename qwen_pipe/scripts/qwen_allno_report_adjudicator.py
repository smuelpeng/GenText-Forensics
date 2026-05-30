#!/usr/bin/env python3
"""Second-stage Qwen adjudicator for crop-audit all-NO samples.

This diagnostic script is GT-blind.  It sees the page image, the current
prediction report, and the fact that a crop audit found no hard visual evidence
in the audited boxes.  It does not see GT labels, masks, or evaluation fields.
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

from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import (  # noqa: E402
    call_qwen_crop,
    load_api_key,
    parse_json_object,
    resolve_image_path,
    resolve_pipe_path,
    setup_debug_import,
)


def compact(text: str, limit: int = 5000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def page_preview(image_path: Path, out_path: Path, max_side: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
        image.save(out_path, quality=88)
    return out_path


def all_no_sample_ids(audit_jsonl: Path, min_boxes: int) -> set[str]:
    selected: set[str] = set()
    for line in audit_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        boxes = row.get("boxes") or []
        if len(boxes) < min_boxes:
            continue
        verdicts = [str(box.get("verdict") or "").upper() for box in boxes]
        if verdicts and all(v == "NO" for v in verdicts):
            selected.add(str(row.get("sample_id")))
    return selected


def adjudicator_prompt(report: str, audit_note: str) -> str:
    return f"""You are a conservative document-forensics second-stage reviewer.

Inspect the whole page image and the current predicted anomaly report below.
A previous crop audit checked the predicted boxes and found no visible hard local evidence in those crops.

Task:
- Decide whether the current FORGED prediction should be kept or rejected to AUTHENTIC.
- REJECT_TO_AUTHENTIC only when the report is mostly speculative, ordinary layout/OCR noise, normal table/text rendering, or claims that cannot be verified from the image.
- KEEP_FORGED when the page or report contains concrete, local, image-visible manipulation evidence, or a self-contained document inconsistency that is plainly verifiable from the page itself.
- Use only the image and the report. Do not assume any external ground truth.

Crop-audit summary:
{audit_note}

Current predicted report:
{compact(report)}

Return strict JSON only:
{{"decision":"KEEP_FORGED|REJECT_TO_AUTHENTIC|UNCERTAIN","confidence":0.0,"evidence_type":"visual_hard|logical_self_contained|semantic_external|speculative_noise","reason":"short image/report-grounded reason"}}
"""


def cache_key(sample_id: str, model: str, prompt: str, image_path: Path) -> str:
    payload = {
        "sample_id": sample_id,
        "model": model,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "image_name": image_path.name,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--cache-dir", default="outputs/cache/allno_report_adjudicator")
    parser.add_argument("--api-key-file", default="/Users/penpen/Desktop/api-key.txt")
    parser.add_argument("--model", default="qwen-vl-max-latest")
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--min-boxes", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    api_key = load_api_key(args.api_key_file)
    cache_dir = resolve_pipe_path(args.cache_dir)
    out_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selected = all_no_sample_ids(resolve_pipe_path(args.audit_jsonl), args.min_boxes)

    rows: list[dict[str, Any]] = []
    api_calls = 0
    decisions: dict[str, int] = {}
    for row in read_jsonl(resolve_pipe_path(args.input_jsonl)):
        sid = sample_id_from_row(row)
        if sid not in selected:
            continue
        image_path = resolve_image_path(row, debug_root)
        if not image_path:
            rows.append({"sample_id": sid, "error": "missing_image"})
            continue
        preview = page_preview(image_path, cache_dir / "previews" / f"{sid}.jpg", args.max_side)
        prompt = adjudicator_prompt(str(row.get("raw_output") or ""), "All audited predicted boxes were judged NO by the crop-level hard-evidence verifier.")
        key = cache_key(sid, args.model, prompt, preview)
        response_path = cache_dir / "responses" / f"{key}.json"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        if response_path.exists():
            payload = json.loads(response_path.read_text(encoding="utf-8"))
            payload["cache_hit"] = True
        else:
            raw, usage = call_qwen_crop(
                crop_path=preview,
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
        decision = str(parsed.get("decision") or "UNKNOWN").upper()
        decisions[decision] = decisions.get(decision, 0) + 1
        rows.append(
            {
                "sample_id": sid,
                "decision": decision,
                "confidence": parsed.get("confidence"),
                "evidence_type": parsed.get("evidence_type"),
                "reason": parsed.get("reason"),
                "cache_hit": payload.get("cache_hit"),
                "preview_path": str(preview),
            }
        )

    out_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "audit_jsonl": str(resolve_pipe_path(args.audit_jsonl)),
        "output_jsonl": str(out_path),
        "sample_count": len(rows),
        "api_calls": api_calls,
        "model": args.model,
        "decisions": decisions,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
