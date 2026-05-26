#!/usr/bin/env python3
"""Staged evidence-grounded DocShield-style API baseline.

This runner uses a staged pipeline over a base vision-language model:

1. OCR/Layout
2. Evidence extraction
3. Cross-cue validation
4. Spatial grounding
5. Report synthesis

Ground-truth labels, masks, and reports are never included in model-visible
messages. They are only read by the local evaluator.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOC_DIR))

from postprocess import parse_cct_report  # noqa: E402
from staged_prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    evidence_prompt,
    grounding_prompt,
    ocr_layout_prompt,
    report_prompt,
    validation_prompt,
)


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
GT_ONLY_FIELDS = {"label", "label_codalab", "report_text", "report_path", "mask_path", "has_mask"}


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def safe_extract_data(tar_path: Path, repo_root: Path) -> None:
    """Extract data.tar while rejecting paths outside repo_root/data."""

    if not tar_path.exists():
        raise FileNotFoundError(f"Missing data archive: {tar_path}")

    target_root = (repo_root / "data").resolve()
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        for member in members:
            member_path = (repo_root / member.name).resolve()
            if member_path == target_root:
                continue
            if not str(member_path).startswith(str(target_root) + os.sep):
                raise RuntimeError(f"Refusing to extract unsafe path from {tar_path}: {member.name}")
        tf.extractall(repo_root)


def ensure_data_available(input_jsonl: Path) -> None:
    if input_jsonl.exists():
        return
    archive = REPO_ROOT / "data.tar"
    print(f"[data] {input_jsonl} not found; extracting {archive}")
    safe_extract_data(archive, REPO_ROOT)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Expected split after extraction, but missing: {input_jsonl}")


def ensure_sample_assets_available(rows: list[dict[str, Any]]) -> None:
    missing = []
    for row in rows:
        image_path = row.get("image_path")
        if image_path and not resolve_repo_path(str(image_path)).exists():
            missing.append(str(image_path))
    if not missing:
        return
    archive = REPO_ROOT / "data.tar"
    print(f"[data] {len(missing)} selected images missing; extracting {archive}")
    safe_extract_data(archive, REPO_ROOT)
    still_missing = [path for path in missing if not resolve_repo_path(path).exists()]
    if still_missing:
        sample = ", ".join(still_missing[:3])
        raise FileNotFoundError(f"Missing images after extraction: {sample}")


def image_to_data_url(path: str | Path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_messages(prompt: str, image_path: Path | None = None) -> list[dict[str, Any]]:
    if image_path is None:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def call_api(
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
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
    result: dict[str, Any] | None = None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    if result is None:
        raise RuntimeError(f"API call failed without response: {last_exc!r}")

    content = result["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", part)) for part in content)
    return str(content), result.get("usage", {})


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_json_object(raw: str) -> tuple[dict[str, Any], str | None]:
    text = (raw or "").strip()
    fence = JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value, None
        return {}, "json_root_not_object"
    except json.JSONDecodeError as exc:
        return {}, f"json_decode_error: {exc}"


def normalize_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width, int(round(x1))))
    x2 = max(0, min(width, int(round(x2))))
    y1 = max(0, min(height, int(round(y1))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def union_boxes(boxes: list[list[int]]) -> list[int] | None:
    boxes = [b for b in boxes if b and len(b) == 4]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def collect_spans(ocr_layout: dict[str, Any], width: int, height: int) -> dict[str, list[int]]:
    spans: dict[str, list[int]] = {}
    for span in ocr_layout.get("text_spans") or []:
        if not isinstance(span, dict):
            continue
        span_id = str(span.get("id") or "")
        bbox = normalize_bbox(span.get("bbox"), width, height)
        if span_id and bbox:
            spans[span_id] = bbox
    return spans


def collect_candidate_boxes(evidence: dict[str, Any], width: int, height: int) -> dict[str, list[int]]:
    boxes: dict[str, list[int]] = {}
    for key in ("visual_candidates", "logical_candidates"):
        for candidate in evidence.get(key) or []:
            if not isinstance(candidate, dict):
                continue
            cid = str(candidate.get("id") or "")
            bbox = normalize_bbox(candidate.get("bbox"), width, height)
            if cid and bbox:
                boxes[cid] = bbox
    return boxes


def normalize_validated(
    validated: dict[str, Any],
    ocr_layout: dict[str, Any],
    evidence: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    verdict = str(validated.get("verdict") or "AUTHENTIC").upper()
    if verdict not in {"FORGED", "AUTHENTIC"}:
        verdict = "AUTHENTIC"
    try:
        risk_score = int(validated.get("risk_score") or (80 if verdict == "FORGED" else 0))
    except (TypeError, ValueError):
        risk_score = 80 if verdict == "FORGED" else 0
    risk_score = max(0, min(100, risk_score))

    span_boxes = collect_spans(ocr_layout, width, height)
    candidate_boxes = collect_candidate_boxes(evidence, width, height)

    normalized: list[dict[str, Any]] = []
    if verdict == "FORGED":
        for idx, anomaly in enumerate(validated.get("validated_anomalies") or [], start=1):
            if not isinstance(anomaly, dict):
                continue
            span_ids = [str(v) for v in anomaly.get("span_ids") or [] if str(v) in span_boxes]
            source_ids = [
                str(v) for v in anomaly.get("source_candidate_ids") or [] if str(v) in candidate_boxes
            ]
            boxes = [span_boxes[sid] for sid in span_ids] or [candidate_boxes[cid] for cid in source_ids]
            bbox = union_boxes(boxes) or normalize_bbox(anomaly.get("bbox"), width, height)
            if not bbox:
                continue
            category = str(anomaly.get("category") or "text tampering")
            if category not in {"Visual Clumsy", "Logical Fraud", "semantic_subtle", "text tampering"}:
                category = "text tampering"
            normalized.append(
                {
                    "id": str(anomaly.get("id") or f"a{idx}"),
                    "category": category,
                    "source_candidate_ids": source_ids,
                    "span_ids": span_ids,
                    "bbox": bbox,
                    "visual_support": str(anomaly.get("visual_support") or ""),
                    "logical_support": str(anomaly.get("logical_support") or ""),
                    "reason": str(anomaly.get("reason") or anomaly.get("visual_support") or anomaly.get("logical_support") or ""),
                    "confidence": anomaly.get("confidence", 0.0),
                }
            )

    if verdict == "FORGED" and not normalized:
        fallback_boxes = collect_candidate_boxes(evidence, width, height)
        fallback_candidates: list[dict[str, Any]] = []
        for key in ("visual_candidates", "logical_candidates"):
            for candidate in evidence.get(key) or []:
                if isinstance(candidate, dict):
                    fallback_candidates.append(candidate)
        fallback_candidates.sort(key=lambda c: float(c.get("confidence") or 0.0), reverse=True)
        for idx, candidate in enumerate(fallback_candidates[:3], start=1):
            cid = str(candidate.get("id") or "")
            bbox = fallback_boxes.get(cid)
            if not bbox:
                span_ids = [str(v) for v in candidate.get("span_ids") or [] if str(v) in span_boxes]
                bbox = union_boxes([span_boxes[sid] for sid in span_ids])
            if not bbox:
                continue
            normalized.append(
                {
                    "id": f"fallback_{idx}",
                    "category": "text tampering",
                    "source_candidate_ids": [cid] if cid else [],
                    "span_ids": [str(v) for v in candidate.get("span_ids") or [] if str(v) in span_boxes],
                    "bbox": bbox,
                    "visual_support": str(candidate.get("evidence") or ""),
                    "logical_support": str(candidate.get("evidence") or ""),
                    "reason": str(candidate.get("evidence") or "Validated document-authenticity anomaly."),
                    "confidence": candidate.get("confidence", 0.5),
                }
            )

    if not normalized:
        verdict = "AUTHENTIC"
        risk_score = min(risk_score, 10)

    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "validated_anomalies": normalized,
        "discarded_candidates": validated.get("discarded_candidates") or [],
        "grounding_policy": "Grounding boxes are normalized from OCR span boxes, evidence candidate boxes, or their union.",
    }


def fallback_report(ocr_layout: dict[str, Any], validated: dict[str, Any]) -> str:
    verdict = validated.get("verdict", "AUTHENTIC")
    risk = int(validated.get("risk_score") or (80 if verdict == "FORGED" else 0))
    anomalies = validated.get("validated_anomalies") or []
    language = str(ocr_layout.get("document_language") or "unknown").lower()
    no_anomaly_text = {
        "zh": "未检测到异常。该文档已完成检查，未发现篡改、改动或伪造迹象。",
        "th": "ไม่พบความผิดปกติ เอกสารนี้ได้รับการตรวจสอบแล้ว และไม่พบร่องรอยการแก้ไข ดัดแปลง หรือปลอมแปลง",
        "ar": "لم يتم رصد أي شذوذ. تم فحص المستند بدقة ولم تظهر علامات تلاعب أو تعديل أو تزوير.",
        "id": "Tidak ada anomali yang terdeteksi. Dokumen telah diperiksa dan tidak ditemukan tanda perubahan, manipulasi, atau pemalsuan.",
        "ms": "Tiada anomali dikesan. Dokumen telah diperiksa dan tiada tanda pengubahsuaian, manipulasi, atau pemalsuan ditemui.",
    }.get(
        language,
        "No anomalies detected. The document has been thoroughly examined and no signs of tampering, alteration, or forgery were found.",
    )
    summary_prefix = {
        "zh": f"本次检查识别出 {len(anomalies)} 个异常，伪造风险分数为 {risk}。",
        "th": f"การตรวจสอบพบความผิดปกติ {len(anomalies)} รายการ โดยมีคะแนนความเสี่ยงการปลอมแปลง {risk}.",
        "ar": f"حدد الفحص {len(anomalies)} شذوذات، وكانت درجة خطر التزوير {risk}.",
        "id": f"Pemeriksaan dokumen mengidentifikasi {len(anomalies)} anomali, dengan skor risiko pemalsuan {risk}.",
        "ms": f"Pemeriksaan dokumen mengenal pasti {len(anomalies)} anomali, dengan skor risiko pemalsuan {risk}.",
    }.get(
        language,
        f"The examination of the document has identified {len(anomalies)} anomalies, resulting in a fraud risk score of {risk}.",
    )
    lines = [
        "# FORGERY ANALYSIS  REPORT",
        "",
        "**Report ID:** FAR-xxxx-xx-xx",
        "**Date of Examination:** xxxx-xx-xx",
        "**Case Type:** Document Authentication & Fraud Analysis",
        "",
        "**Overall Assessment:**",
        f"    **[Conclusion]:** {verdict}",
        f"    **[RISK_SCORE]:** {risk}",
        "",
        "---",
        "",
        "## DETAILED ANOMALY ANALYSIS",
        "",
    ]
    if verdict == "FORGED" and anomalies:
        for idx, anomaly in enumerate(anomalies, start=1):
            lines.extend(
                [
                    f"### ANOMALY_{idx:03d}: {anomaly.get('category', 'text tampering')} (validated region)",
                    f"[GROUNDING]:{anomaly.get('bbox')}",
                    f"[REASON]: {anomaly.get('reason') or anomaly.get('visual_support') or anomaly.get('logical_support')}",
                    "",
                ]
            )
    else:
        lines.append(no_anomaly_text)
        lines.append("")
    lines.extend(
        [
            "---",
            "",
            "## SUMMARY",
            f"{summary_prefix} {ocr_layout.get('global_summary', '')}",
            "",
            "---",
            "**END OF REPORT**",
        ]
    )
    return "\n".join(lines)


def ensure_report_structure(report: str) -> str:
    """Normalize lightweight report markers expected by the local evaluator."""

    text = (report or "").strip()
    if not re.search(r"DETAILED\s+ANOMALY\s+ANALYSIS", text, re.IGNORECASE):
        detail_stub = (
            "\n\n---\n\n## DETAILED ANOMALY ANALYSIS\n\n"
            "No additional localized anomaly details were produced beyond the overall assessment.\n"
        )
        summary_match = re.search(r"\n\s*##\s*SUMMARY", text, re.IGNORECASE)
        if summary_match:
            text = text[: summary_match.start()] + detail_stub + text[summary_match.start() :]
        else:
            text += detail_stub
    if not re.search(r"END\s+OF\s+REPORT", text, re.IGNORECASE):
        text = text.rstrip() + "\n\n---\n**END OF REPORT**"
    return text


def authentic_downgrade_report(ocr_layout: dict[str, Any]) -> str:
    return fallback_report(
        ocr_layout,
        {
            "verdict": "AUTHENTIC",
            "risk_score": 5,
            "validated_anomalies": [],
            "discarded_candidates": [],
        },
    )


def run_stage(
    *,
    stage_name: str,
    prompt: str,
    image_path: Path | None,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    timeout: int,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    raw, usage = call_api(
        build_messages(prompt, image_path),
        model,
        api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        timeout=timeout,
    )
    parsed: dict[str, Any] | None = None
    if stage_name != "report":
        parsed, error = parse_json_object(raw)
        if error:
            parsed = {"_parse_error": error}
    return raw, usage, parsed


def process_row(
    row: dict[str, Any],
    *,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    timeout: int,
    forged_risk_threshold: int,
) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    image_name = row.get("image_file") or Path(str(row.get("image_path") or "")).name
    image_path = resolve_repo_path(str(row.get("image_path") or ""))

    t0 = time.time()
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            width, height = im.size

        stage_outputs: dict[str, Any] = {}
        stage_usages: dict[str, Any] = {}

        ocr_raw, usage, ocr_layout = run_stage(
            stage_name="ocr_layout",
            prompt=ocr_layout_prompt(str(image_name), width, height),
            image_path=image_path,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
        stage_outputs["ocr_layout"] = {"raw": ocr_raw, "parsed": ocr_layout}
        stage_usages["ocr_layout"] = usage

        evidence_raw, usage, evidence = run_stage(
            stage_name="evidence",
            prompt=evidence_prompt(ocr_layout or {}, str(image_name), width, height),
            image_path=image_path,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
        stage_outputs["evidence_candidates"] = {"raw": evidence_raw, "parsed": evidence}
        stage_usages["evidence_candidates"] = usage

        validation_raw, usage, validation = run_stage(
            stage_name="validation",
            prompt=validation_prompt(ocr_layout or {}, evidence or {}, str(image_name), width, height),
            image_path=image_path,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
        stage_outputs["validation"] = {
            "raw": validation_raw,
            "parsed": validation,
        }
        stage_usages["validation"] = usage

        grounding_raw, usage, grounding = run_stage(
            stage_name="grounding",
            prompt=grounding_prompt(ocr_layout or {}, evidence or {}, validation or {}, str(image_name), width, height),
            image_path=image_path,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
        normalized_validation = normalize_validated(grounding or validation or {}, ocr_layout or {}, evidence or {}, width, height)
        stage_outputs["grounding"] = {
            "raw": grounding_raw,
            "parsed": grounding,
            "normalized": normalized_validation,
        }
        stage_outputs["validation_grounding"] = {
            "raw": grounding_raw,
            "parsed": grounding,
            "normalized": normalized_validation,
        }
        stage_usages["grounding"] = usage

        report_raw, usage, _ = run_stage(
            stage_name="report",
            prompt=report_prompt(ocr_layout or {}, normalized_validation, str(image_name), width, height),
            image_path=None,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
        final_report = ensure_report_structure(report_raw)
        parsed_report = parse_cct_report(final_report)
        if parsed_report.get("conclusion") == "UNKNOWN":
            final_report = ensure_report_structure(fallback_report(ocr_layout or {}, normalized_validation))
            parsed_report = parse_cct_report(final_report)
        if (
            parsed_report.get("conclusion") == "FORGED"
            and parsed_report.get("risk_score") is not None
            and int(parsed_report.get("risk_score") or 0) < forged_risk_threshold
        ):
            final_report = ensure_report_structure(authentic_downgrade_report(ocr_layout or {}))
            parsed_report = parse_cct_report(final_report)

        stage_outputs["report"] = {"raw": report_raw}
        stage_usages["report"] = usage

        return {
            "sample_id": sample_id,
            "image_name": image_name,
            "image_path": str(row.get("image_path") or ""),
            "width": width,
            "height": height,
            "model": model,
            "pipeline": "staged_evidence_cct",
            "raw_output": final_report,
            "raw_output_full": report_raw,
            "parsed": parsed_report,
            "stage_outputs": stage_outputs,
            "usage": stage_usages,
            "elapsed_sec": round(time.time() - t0, 3),
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "image_name": image_name,
            "image_path": str(row.get("image_path") or ""),
            "model": model,
            "pipeline": "staged_evidence_cct",
            "error": repr(exc),
            "elapsed_sec": round(time.time() - t0, 3),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_api_key(path: str) -> str:
    key_path = resolve_repo_path(path)
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        return api_key
    raise SystemExit("Error: No API key found. Provide --api-key-file or set DASHSCOPE_API_KEY")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3.6-35b-a3b")
    p.add_argument("--input-jsonl", default="data/val_300.jsonl")
    p.add_argument("--output-jsonl", default="outputs/raw/staged_docshield_api_val_60.jsonl")
    p.add_argument("--api-key-file", default="/Users/penpen/Desktop/api-key.txt")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-samples", type=int, default=60, help="0 = all")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--forged-risk-threshold", type=int, default=80)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_repo_path(args.input_jsonl)
    ensure_data_available(input_jsonl)
    api_key = load_api_key(args.api_key_file)

    rows = read_jsonl(input_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    ensure_sample_assets_available(rows)

    out_path = resolve_repo_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("sample_id") or rec.get("image_name")
                if key and "error" not in rec:
                    done_keys.add(str(key))
        print(f"[resume] skipping {len(done_keys)} already processed rows")

    filtered = []
    for row in rows:
        key = row.get("sample_id") or row.get("image_file") or row.get("image_path")
        if key and str(key) in done_keys:
            continue
        filtered.append(row)

    if not filtered:
        print("[run_staged_docshield_api] nothing to do -- all rows processed")
        return

    print(
        f"[run_staged_docshield_api] model={args.model} rows={len(filtered)} "
        f"workers={args.num_workers} thinking={args.enable_thinking}"
    )
    mode = "a" if args.resume and out_path.exists() else "w"
    written = 0
    errors = 0

    from tqdm import tqdm

    kwargs = {
        "model": args.model,
        "api_key": api_key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        "timeout": args.timeout,
        "forged_risk_threshold": args.forged_risk_threshold,
    }

    if args.num_workers <= 1:
        with out_path.open(mode, encoding="utf-8") as fout:
            pbar = tqdm(filtered, desc=f"{args.model} staged", unit="img")
            for row in pbar:
                record = process_row(row, **kwargs)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                if "error" in record:
                    errors += 1
                pred = record.get("parsed", {}).get("conclusion", "?")
                pbar.set_postfix(pred=pred, err=errors, t=f"{record.get('elapsed_sec', 0):.0f}s")
    else:
        with out_path.open(mode, encoding="utf-8") as fout:
            pbar = tqdm(total=len(filtered), desc=f"{args.model} staged x{args.num_workers}", unit="img")
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {executor.submit(process_row, row, **kwargs): row for row in filtered}
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

    print(f"\n[run_staged_docshield_api] done: {written} rows, {errors} errors -> {out_path}")


if __name__ == "__main__":
    main()
