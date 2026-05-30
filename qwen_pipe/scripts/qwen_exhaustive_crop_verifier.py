#!/usr/bin/env python3
"""Verify exhaustive localization candidates with a Qwen-VL crop pass.

The script is GT-blind at inference time.  It reads candidate pools produced by
qwen_exhaustive_recall, crops a small mixed-family candidate set, asks a VLM
whether each crop contains visible document-forensics evidence, and appends
accepted local boxes to the current report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import box_iou, insert_extra_anomalies, page_area_ratio, read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import (  # noqa: E402
    call_qwen_crop,
    clamp_box,
    expand_box,
    load_api_key,
    parse_json_object,
    resolve_image_path,
    resolve_pipe_path,
    setup_debug_import,
)


FAMILY_ORDER = ["linegrid", "grid", "ocr", "evidence", "patch", "row", "token"]
VISUAL_WORDS = (
    "blur",
    "pixel",
    "artifact",
    "font",
    "overlap",
    "misalign",
    "style",
    "color",
    "redaction",
    "block",
    "table",
    "cell",
    "date",
    "amount",
    "number",
    "render",
    "模糊",
    "锯齿",
    "重叠",
    "错位",
    "字体",
    "颜色",
    "数字",
    "日期",
)


def conclusion_is_forged(row: dict[str, Any]) -> bool:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper() == "FORGED"


def compact_report(report: str, limit: int = 1400) -> str:
    text = re.sub(r"\s+", " ", report or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head} ... {tail}"


def candidate_sort_score(cand: dict[str, Any], width: int, height: int) -> float:
    family = str(cand.get("family") or "")
    source = str(cand.get("source") or "")
    ratio = page_area_ratio(cand.get("box") or [0, 0, 0, 0], width, height)
    meta = cand.get("meta") or {}
    rank = meta.get("v446physdiverse_rank") or meta.get("v87b_rank") or meta.get("v87_rank") or {}
    score = float(cand.get("score") or 0.0) * 0.35 + float(rank.get("score") or 0.0) * 0.08
    score += {"linegrid": 3.0, "ocr": 2.8, "grid": 2.5, "evidence": 2.4, "patch": 2.2, "row": 2.0, "token": 1.0}.get(family, 0.5)
    if source.startswith("ocr_scriptgrid"):
        score += 0.8
    elif source.startswith("ocr_linegrid"):
        score += 0.6
    elif source.startswith("grid_"):
        score += 0.7
    elif source.startswith("stage_evidence"):
        score += 0.5
    if ratio < 0.0002:
        score -= 1.5
    if ratio > 0.12:
        score -= 2.5
    elif ratio > 0.07:
        score -= 0.8
    text = str(cand.get("text") or "").lower()
    if any(word in text for word in VISUAL_WORDS):
        score += 0.5
    return score


def select_mixed_candidates(
    diag: dict[str, Any],
    *,
    max_candidates: int,
    max_area_ratio: float,
) -> list[dict[str, Any]]:
    width = int(diag.get("width") or 1)
    height = int(diag.get("height") or 1)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, int, int, int, str]] = set()
    for cand in diag.get("candidates") or []:
        if cand.get("family") == "existing" or str(cand.get("source") or "").startswith("current_final_report"):
            continue
        raw_box = cand.get("box")
        if not isinstance(raw_box, list) or len(raw_box) < 4:
            continue
        box = clamp_box([float(v) for v in raw_box[:4]], width, height)
        if not box:
            continue
        if page_area_ratio(box, width, height) > max_area_ratio:
            continue
        key = (*box, str(cand.get("family") or ""))
        if key in seen:
            continue
        seen.add(key)
        out = dict(cand)
        out["box"] = box
        buckets[str(out.get("family") or "")].append(out)

    for family, rows in buckets.items():
        rows.sort(key=lambda c: candidate_sort_score(c, width, height), reverse=True)

    selected: list[dict[str, Any]] = []
    per_family_cap = max(1, max_candidates // max(1, len(FAMILY_ORDER) - 1))
    for family in FAMILY_ORDER:
        cap = 2 if family in {"linegrid", "grid", "ocr"} else per_family_cap
        for cand in buckets.get(family, [])[:cap]:
            if any(box_iou(cand["box"], prev["box"]) >= 0.62 for prev in selected):
                continue
            selected.append(cand)
            if len(selected) >= max_candidates:
                return selected
    if len(selected) < max_candidates:
        leftovers = [cand for rows in buckets.values() for cand in rows]
        leftovers.sort(key=lambda c: candidate_sort_score(c, width, height), reverse=True)
        for cand in leftovers:
            if cand in selected:
                continue
            if any(box_iou(cand["box"], prev["box"]) >= 0.62 for prev in selected):
                continue
            selected.append(cand)
            if len(selected) >= max_candidates:
                break
    return selected


def verifier_prompt(report_hint: str, cand: dict[str, Any], crop_size: tuple[int, int]) -> str:
    text = re.sub(r"\s+", " ", str(cand.get("text") or "")).strip()[:500]
    return f"""You are checking one cropped candidate region from a document-forensics localization pipeline.

Only inspect the crop image. The current system already predicted the document as forged, but this candidate may be wrong.

Return YES only if the crop itself contains visible local evidence of document manipulation, such as:
- inserted/edited text with different font, weight, color, blur, aliasing, broken rendering, or unnatural overlap;
- abnormal table cell/content alignment, spacing, or pasted-looking numeric/date/amount values;
- redaction, dark block, smear, or localized artifact;
- clearly suspicious local visual inconsistency around OCR text.

Return NO if the crop is merely normal text, a normal table/grid area, or only semantically related to the report without visible local evidence.

Candidate family/source: {cand.get("family")} / {cand.get("source")}
Candidate text hint: {text}
Candidate box in original page: {cand.get("box")}
Crop size: width={crop_size[0]}, height={crop_size[1]}

Current predicted report hint, possibly noisy:
{report_hint}

Return strict JSON only:
{{"verdict":"YES|NO","local_bbox":[x1,y1,x2,y2],"confidence":0.0,"evidence":"short visual reason"}}
"""


def crop_candidate(image: Image.Image, cand: dict[str, Any], cache_dir: Path, sample_id: str, index: int, pad: float) -> tuple[Path, list[int]]:
    width, height = image.size
    crop_box = expand_box(cand["box"], width, height, pad, pad, min_pad=32)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{cand.get('family')}_{cand.get('source')}_{index}")[:80]
    crop_path = cache_dir / "crops" / f"{sample_id}_{label}.jpg"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not crop_path.exists():
        image.crop(tuple(crop_box)).save(crop_path, quality=92)
    return crop_path, crop_box


def cache_key(sample_id: str, cand: dict[str, Any], crop_box: list[int], model: str, prompt: str) -> str:
    payload = {
        "sample_id": sample_id,
        "box": cand.get("box"),
        "source": cand.get("source"),
        "family": cand.get("family"),
        "crop_box": crop_box,
        "model": model,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def map_local_bbox(local_bbox: Any, crop_box: list[int], width: int, height: int) -> list[int] | None:
    if not isinstance(local_bbox, list) or len(local_bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in local_bbox[:4]]
    except Exception:
        return None
    cw = max(1, crop_box[2] - crop_box[0])
    ch = max(1, crop_box[3] - crop_box[1])
    if x2 <= x1 or y2 <= y1 or x1 < -2 or y1 < -2 or x2 > cw + 2 or y2 > ch + 2:
        return None
    return clamp_box([x1 + crop_box[0], y1 + crop_box[1], x2 + crop_box[0], y2 + crop_box[1]], width, height)


def insert_verified_anomalies_compact(report: str, extras: list[dict[str, Any]]) -> str:
    if not extras:
        return report
    block: list[str] = []
    for idx, cand in enumerate(extras, start=1):
        box = cand["box"]
        evidence = re.sub(r"\s+", " ", str(cand.get("text") or "Local visual obstruction confirmed in crop.")).strip()
        evidence = evidence[:120]
        block.extend(
            [
                f"### ANOMALY_CROP_VERIFY_{idx:03d}: Local Visual Obstruction",
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


def verify_one(
    *,
    sample_id: str,
    image: Image.Image,
    cand: dict[str, Any],
    report_hint: str,
    cache_dir: Path,
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: int,
    crop_pad: float,
) -> dict[str, Any]:
    width, height = image.size
    crop_path, crop_box = crop_candidate(image, cand, cache_dir, sample_id, len(str(cand.get("box"))), crop_pad)
    prompt = verifier_prompt(report_hint, cand, Image.open(crop_path).size)
    key = cache_key(sample_id, cand, crop_box, model, prompt)
    response_path = cache_dir / "responses" / f"{key}.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    if response_path.exists():
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        payload["cache_hit"] = True
        return payload
    raw, usage = call_qwen_crop(
        crop_path=crop_path,
        prompt=prompt,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    parsed = parse_json_object(raw)
    mapped = map_local_bbox((parsed or {}).get("local_bbox"), crop_box, width, height) if parsed else None
    payload = {
        "sample_id": sample_id,
        "candidate": cand,
        "crop_path": str(crop_path),
        "crop_box": crop_box,
        "model": model,
        "raw": raw,
        "parsed": parsed,
        "mapped_box": mapped,
        "usage": usage,
        "cache_hit": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def coerce_original_or_candidate_box(
    payload: dict[str, Any],
    candidate_box: list[int] | None,
    width: int,
    height: int,
    *,
    allow_original_bbox: bool,
    allow_candidate_fallback: bool,
) -> list[int] | None:
    """Recover boxes when the VLM ignores the requested crop-local frame.

    In early smoke runs Qwen-VL sometimes returned the candidate's original-page
    coordinates, or a visually correct YES with an invalid local box.  Keep this
    as an explicit opt-in gate so the default verifier remains strict.
    """
    if allow_original_bbox:
        parsed = payload.get("parsed") or {}
        raw = parsed.get("local_bbox")
        if isinstance(raw, list) and len(raw) >= 4:
            try:
                vals = [int(float(v)) for v in raw[:4]]
            except Exception:
                vals = []
            if vals:
                page_box = clamp_box(vals, width, height)
                if page_box and candidate_box and box_iou(page_box, candidate_box) >= 0.03:
                    return page_box
    if allow_candidate_fallback and candidate_box:
        return list(candidate_box)
    return None


def accepted_box(
    payload: dict[str, Any],
    width: int,
    height: int,
    *,
    min_confidence: float,
    max_area_ratio: float,
    candidate_box: list[int] | None = None,
    allow_original_bbox: bool = False,
    allow_candidate_fallback: bool = False,
) -> list[int] | None:
    parsed = payload.get("parsed") or {}
    if str(parsed.get("verdict") or "").upper() != "YES":
        return None
    try:
        conf = float(parsed.get("confidence") or 0.0)
    except Exception:
        conf = 0.0
    if conf < min_confidence:
        return None
    box = payload.get("mapped_box")
    if not isinstance(box, list) or len(box) < 4:
        box = coerce_original_or_candidate_box(
            payload,
            candidate_box,
            width,
            height,
            allow_original_bbox=allow_original_bbox,
            allow_candidate_fallback=allow_candidate_fallback,
        )
    if not isinstance(box, list) or len(box) < 4:
        return None
    if page_area_ratio(box, width, height) > max_area_ratio:
        return None
    if int(box[2]) <= int(box[0]) or int(box[3]) <= int(box[1]):
        return None
    return [int(v) for v in box[:4]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--diag-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(PIPE_ROOT.parent / "debug_distribution"))
    parser.add_argument("--cache-dir", default=str(PIPE_ROOT / "outputs/cache/exhaustive_crop_verifier"))
    parser.add_argument("--api-key-file", default="/Users/penpen/Desktop/api-key.txt")
    parser.add_argument("--model", default="qwen-vl-max-latest")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-candidates-per-sample", type=int, default=6)
    parser.add_argument("--max-accepted-per-sample", type=int, default=2)
    parser.add_argument("--candidate-max-area-ratio", type=float, default=0.12)
    parser.add_argument("--accepted-max-area-ratio", type=float, default=0.08)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--crop-pad", type=float, default=0.28)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--allow-original-bbox", action="store_true")
    parser.add_argument("--allow-candidate-fallback", action="store_true")
    parser.add_argument("--fallback-families", default="")
    parser.add_argument("--accept-families", default="")
    parser.add_argument("--accept-evidence-regex", default="")
    parser.add_argument("--reject-evidence-regex", default="")
    parser.add_argument("--insert-mode", choices=["full", "compact"], default="full")
    parser.add_argument("--write-subset-only", action="store_true")
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    cache_dir = resolve_pipe_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key(args.api_key_file)

    raw_rows = read_jsonl(resolve_pipe_path(args.input_jsonl))
    candidate_rows = {str(r.get("sample_id") or ""): r for r in read_jsonl(resolve_pipe_path(args.candidate_jsonl))}
    selected = set(args.sample_id or [])
    if not selected and args.max_samples > 0:
        selected = set(list(candidate_rows.keys())[: args.max_samples])

    out_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "rows_total": len(raw_rows),
        "selected_count": len(selected),
        "rows_processed": 0,
        "rows_changed": 0,
        "api_calls": 0,
        "accepted_boxes": 0,
        "model": args.model,
        "candidate_counts": {},
        "accepted_family_counts": {},
    }
    fallback_families = {p.strip() for p in str(args.fallback_families or "").split(",") if p.strip()}
    accept_families = {p.strip() for p in str(args.accept_families or "").split(",") if p.strip()}
    accept_evidence_re = re.compile(args.accept_evidence_regex, re.I) if args.accept_evidence_regex else None
    reject_evidence_re = re.compile(args.reject_evidence_regex, re.I) if args.reject_evidence_regex else None
    family_counts: Counter[str] = Counter()
    accepted_family_counts: Counter[str] = Counter()

    for row in raw_rows:
        sid = sample_id_from_row(row)
        should_process = sid in selected
        if not should_process:
            if not args.write_subset_only:
                out_rows.append(row)
            continue
        stats["rows_processed"] += 1
        diag = candidate_rows.get(sid)
        if not diag or not conclusion_is_forged(row):
            out_rows.append(row)
            diag_rows.append({"sample_id": sid, "reason": "missing_diag_or_non_forged", "accepted": []})
            continue
        image_path = resolve_image_path(row, debug_root)
        if not image_path:
            out_rows.append(row)
            diag_rows.append({"sample_id": sid, "reason": "missing_image", "accepted": []})
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        report = str(row.get("raw_output") or "")
        existing = report_boxes(report)
        candidates = select_mixed_candidates(
            diag,
            max_candidates=args.max_candidates_per_sample,
            max_area_ratio=args.candidate_max_area_ratio,
        )
        stats["candidate_counts"][sid] = len(candidates)
        attempts: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        occupied = list(existing)
        for cand in candidates:
            family_counts[str(cand.get("family") or "")] += 1
            payload = verify_one(
                sample_id=sid,
                image=image,
                cand=cand,
                report_hint=compact_report(report),
                cache_dir=cache_dir,
                model=args.model,
                api_key=api_key,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                crop_pad=args.crop_pad,
            )
            if not payload.get("cache_hit"):
                stats["api_calls"] += 1
            box = accepted_box(
                payload,
                width,
                height,
                min_confidence=args.min_confidence,
                max_area_ratio=args.accepted_max_area_ratio,
                candidate_box=cand.get("box"),
                allow_original_bbox=args.allow_original_bbox and (not fallback_families or str(cand.get("family") or "") in fallback_families),
                allow_candidate_fallback=args.allow_candidate_fallback and (not fallback_families or str(cand.get("family") or "") in fallback_families),
            )
            evidence_text = str(((payload.get("parsed") or {}).get("evidence")) or "")
            if box is not None and accept_families and str(cand.get("family") or "") not in accept_families:
                box = None
            if box is not None and accept_evidence_re and not accept_evidence_re.search(evidence_text):
                box = None
            if box is not None and reject_evidence_re and reject_evidence_re.search(evidence_text):
                box = None
            gate = {
                "verdict": ((payload.get("parsed") or {}).get("verdict")),
                "confidence": ((payload.get("parsed") or {}).get("confidence")),
                "mapped_box": payload.get("mapped_box"),
                "accepted_box": box,
                "accepted": box is not None,
            }
            attempts.append({"candidate": cand, "gate": gate, "parsed": payload.get("parsed")})
            if box is None:
                continue
            if any(box_iou(box, old) >= 0.62 for old in occupied):
                continue
            entry = dict(cand)
            entry["box"] = box
            entry["source"] = f"qwen_exhaustive_crop_verifier:{cand.get('source')}"
            entry["confidence"] = gate.get("confidence")
            entry["text"] = ((payload.get("parsed") or {}).get("evidence") or cand.get("text") or "")[:260]
            accepted.append(entry)
            occupied.append(box)
            accepted_family_counts[str(cand.get("family") or "")] += 1
            if len(accepted) >= args.max_accepted_per_sample:
                break
        out = dict(row)
        if accepted:
            if args.insert_mode == "compact":
                out["raw_output"] = insert_verified_anomalies_compact(report, accepted)
            else:
                out["raw_output"] = insert_extra_anomalies(report, accepted)
            stage_outputs = dict(out.get("stage_outputs") or {})
            stage_outputs["qwen_exhaustive_crop_verifier"] = {
                "applied": True,
                "accepted_count": len(accepted),
                "attempted_count": len(attempts),
                "model": args.model,
                "insert_mode": args.insert_mode,
                "policy": "Exhaustive candidate crop verifier; prompts include current predicted report only, never GT labels/masks/reports/eval fields.",
                "accepted": accepted,
            }
            out["stage_outputs"] = stage_outputs
            stats["rows_changed"] += 1
            stats["accepted_boxes"] += len(accepted)
        out_rows.append(out)
        diag_rows.append({"sample_id": sid, "attempted": len(attempts), "accepted": accepted, "attempts": attempts})

    stats["candidate_family_counts"] = dict(family_counts)
    stats["accepted_family_counts"] = dict(accepted_family_counts)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    diag_path = resolve_pipe_path(args.diag_jsonl)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in diag_rows) + "\n", encoding="utf-8")
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
