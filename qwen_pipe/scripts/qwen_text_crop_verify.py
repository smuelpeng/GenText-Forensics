#!/usr/bin/env python3
"""Typed localization refinement with OCR/textline crops and optional Qwen verification.

This Qwen-pipe postprocess is GT-blind at inference time. It reads an existing
raw JSONL, generates localization candidates from the current final report,
OCR spans, textline-like boxes, and local visual patch detectors, then either
applies a local candidate policy or asks Qwen to verify cropped regions.

The verifier only sees the crop image, the current anomaly reason, and the
target type. It never receives GT labels, reports, masks, or eval scores.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v63_crop_patch_blockonly_w04_300.jsonl"
DEFAULT_EVAL = PIPE_ROOT / "outputs/eval/qwen_pipe_v63_crop_patch_blockonly_w04_300.json"
DEFAULT_CACHE = PIPE_ROOT / "outputs/cache/text_crop_verify"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

GROUNDING_RE = re.compile(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
QUOTED_RE = re.compile(r"[\"'“‘『「]([^\"'”’』」]{2,80})[\"'”’』」]")
NUMERIC_TOKEN_RE = re.compile(
    r"[-+]?\d+(?:[.,:]\d+)*(?:\s?%|\s?PM|\s?AM|\s?万元|\s?万|年|月|日)?",
    re.IGNORECASE,
)

TARGET_TERMS: dict[str, tuple[str, ...]] = {
    "redaction_block": (
        "redaction",
        "redacted",
        "censored",
        "obscur",
        "cover",
        "covered",
        "black block",
        "black rectangle",
        "solid black",
        "black box",
        "black-box",
        "rectangle",
        "rectangular block",
        "solid block",
        "kotak",
        "blok",
        "hitam",
        "บดบัง",
        "遮挡",
        "遮蔽",
        "覆盖",
        "黑块",
        "黑色块",
        "实心黑色块",
        "像素块",
    ),
    "render_pixel_blur": (
        "pixelated",
        "jagged",
        "blur",
        "blurry",
        "glitch",
        "rendering",
        "low-resolution",
        "low resolution",
        "raster",
        "aliasing",
        "overlap",
        "merged",
        "乱码",
        "渲染",
        "模糊",
        "锯齿",
        "重叠",
        "混叠",
        "ซ้อน",
    ),
    "style_color": (
        "color",
        "colour",
        "gray",
        "grey",
        "lighter",
        "darker",
        "font",
        "bold",
        "weight",
        "style",
        "highlight",
        "颜色",
        "灰色",
        "红色",
        "字体",
        "加粗",
        "高亮",
        "สี",
    ),
    "layout_table": (
        "layout",
        "spacing",
        "alignment",
        "table",
        "cell",
        "row",
        "column",
        "blank",
        "排版",
        "表格",
        "行距",
        "空白",
        "对齐",
    ),
    "logical_numeric": (
        "logical",
        "contradiction",
        "inconsistent",
        "date",
        "amount",
        "total",
        "sum",
        "chronolog",
        "negative",
        "percentage",
        "逻辑",
        "矛盾",
        "金额",
        "日期",
        "数字",
        "不一致",
    ),
}

PRIORITY = ("redaction_block", "render_pixel_blur", "style_color", "layout_table", "logical_numeric")


@dataclass(frozen=True)
class Span:
    span_id: str
    text: str
    box: list[int]
    role: str = ""


@dataclass
class Anomaly:
    index: int
    box: list[int]
    context: str
    reason: str
    target_type: str
    normalized: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    label: str
    box: list[int]
    source: str
    score: float
    crop_box: list[int] = field(default_factory=list)
    crop_path: str = ""


def resolve_pipe_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PIPE_ROOT / p


def setup_debug_import(debug_root: Path) -> None:
    docshield_dir = debug_root / "baselines" / "DocShield"
    if not docshield_dir.exists():
        raise SystemExit(f"DocShield helper directory not found: {docshield_dir}")
    sys.path.insert(0, str(docshield_dir))
    scripts_dir = PIPE_ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))


def load_api_key(path: str) -> str:
    key_path = Path(path).expanduser()
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        return api_key
    raise SystemExit("No API key found. Provide --api-key-file or set DASHSCOPE_API_KEY.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_eval_selection(eval_json: Path, *, threshold: float, limit: int) -> set[str]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    candidates = [
        s
        for s in data.get("samples") or []
        if s.get("gt_label") == "FORGED"
        and s.get("pred_label") == "FORGED"
        and float(s.get("loc_score") or 0.0) < threshold
    ]
    candidates.sort(key=lambda s: (float(s.get("loc_score") or 0.0), str(s.get("sample_id") or "")))
    if limit > 0:
        candidates = candidates[:limit]
    return {str(s.get("sample_id")) for s in candidates}


def resolve_image_path(row: dict[str, Any], debug_root: Path) -> Path | None:
    raw = row.get("image_path") or row.get("image_name")
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    if p.is_absolute() and p.exists():
        return p
    for base in (debug_root, debug_root / "data" / "images", PIPE_ROOT.parent):
        candidate = base / p
        if candidate.exists():
            return candidate
    return None


def clamp_box(box: list[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    out = [
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    ]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def project_box(value: Any, width: int, height: int, mode: str = "auto") -> list[int] | None:
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(nums) < 4:
            return None
        raw = [float(v) for v in nums[:4]]
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            raw = [float(value[i]) for i in range(4)]
        except (TypeError, ValueError):
            return None
    else:
        return None

    max_coord = max(abs(v) for v in raw)
    use_norm = mode == "normalized-1000" or (mode == "auto" and max_coord <= 1000 and (width > 1300 or height > 1300))
    if use_norm:
        return clamp_box([raw[0] * width / 1000.0, raw[1] * height / 1000.0, raw[2] * width / 1000.0, raw[3] * height / 1000.0], width, height)
    return clamp_box(raw, width, height)


def parse_report_anomalies(report: str) -> list[dict[str, Any]]:
    matches = list(GROUNDING_RE.finditer(report or ""))
    out: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        start = report.rfind("###", 0, match.start())
        if start < 0:
            start = max(0, report.rfind("\n", 0, match.start()))
        context = report[start:next_start]
        box = project_box(match.group(2), 10**9, 10**9, mode="pixel")
        if not box:
            continue
        reason_match = re.search(r"\[REASON\]\s*:\s*(.*)", context, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else context.strip()
        out.append({"index": idx, "box": box, "context": context, "reason": reason})
    return out


def normalized_anomalies(row: dict[str, Any]) -> list[dict[str, Any]]:
    stage_outputs = row.get("stage_outputs") or {}
    for stage_name in ("validation_grounding", "grounding"):
        normalized = ((stage_outputs.get(stage_name) or {}).get("normalized") or {})
        anomalies = normalized.get("validated_anomalies") or []
        if anomalies:
            return [a for a in anomalies if isinstance(a, dict)]
    return []


def classify_target(context: str) -> str:
    text = context.lower()
    hits: dict[str, int] = {}
    for target, terms in TARGET_TERMS.items():
        hits[target] = sum(1 for term in terms if term.lower() in text)
    for target in PRIORITY:
        if hits.get(target, 0) > 0:
            return target
    return "render_pixel_blur"


def extract_text_queries(text: str) -> list[str]:
    queries: list[str] = []
    for match in QUOTED_RE.findall(text or ""):
        q = " ".join(match.split())
        if 2 <= len(q) <= 80:
            queries.append(q)
    for match in NUMERIC_TOKEN_RE.findall(text or ""):
        q = " ".join(match.split())
        if len(q) >= 2:
            queries.append(q)
    # Add compact alphanumeric terms that often name fields/URLs.
    for match in re.findall(r"[A-Za-z][A-Za-z0-9_./:-]{3,40}", text or ""):
        if match.lower() not in {"visual", "clumsy", "logical", "fraud", "grounding", "reason"}:
            queries.append(match)
    deduped: list[str] = []
    seen = set()
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped[:12]


def overlap(a: list[int], b: list[int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def iou(a: list[int], b: list[int]) -> float:
    inter = overlap(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_min_ratio(a: list[int], b: list[int]) -> float:
    inter = overlap(a, b)
    denom = min(area(a), area(b))
    return inter / denom if denom > 0 else 0.0


def is_duplicate_box(box: list[int], existing: list[list[int]]) -> bool:
    for old in existing:
        if iou(box, old) > 0.45 or overlap_min_ratio(box, old) > 0.70:
            return True
    return False


def center_distance(a: list[int], b: list[int], width: int, height: int) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return ((ax - bx) / max(width, 1)) ** 2 + ((ay - by) / max(height, 1)) ** 2


def expand_box(box: list[int], width: int, height: int, pad_x: float, pad_y: float, min_pad: int = 16) -> list[int]:
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    dx = max(min_pad, int(round(bw * pad_x)))
    dy = max(min_pad, int(round(bh * pad_y)))
    return [max(0, box[0] - dx), max(0, box[1] - dy), min(width, box[2] + dx), min(height, box[3] + dy)]


def union_boxes(boxes: list[list[int]]) -> list[int] | None:
    valid = [b for b in boxes if b and b[2] > b[0] and b[3] > b[1]]
    if not valid:
        return None
    return [min(b[0] for b in valid), min(b[1] for b in valid), max(b[2] for b in valid), max(b[3] for b in valid)]


def collect_spans(row: dict[str, Any], width: int, height: int, coord_mode: str) -> list[Span]:
    parsed = (((row.get("stage_outputs") or {}).get("ocr_layout") or {}).get("parsed") or {})
    spans: list[Span] = []
    for span in parsed.get("text_spans") or []:
        if not isinstance(span, dict):
            continue
        box = project_box(span.get("bbox"), width, height, coord_mode)
        text = str(span.get("text") or "")
        span_id = str(span.get("id") or "")
        if box and text:
            spans.append(Span(span_id, text, box, str(span.get("role") or "")))
    for span in parsed.get("auxiliary_ocr_spans") or []:
        if not isinstance(span, dict):
            continue
        box = project_box(span.get("bbox"), width, height, "pixel")
        text = str(span.get("text") or "")
        span_id = str(span.get("id") or "")
        if box and text:
            spans.append(Span(span_id, text, box, str(span.get("role") or "auxiliary_ocr")))
    return spans


def span_match_score(span: Span, queries: list[str]) -> float:
    text = span.text.lower()
    compact = re.sub(r"\s+", "", text)
    score = 0.0
    for query in queries:
        q = query.lower()
        qc = re.sub(r"\s+", "", q)
        if not q or len(q) < 2:
            continue
        if q in text or qc in compact:
            score += 4.0 + min(len(q) / 20.0, 2.0)
        else:
            parts = [p for p in re.split(r"\W+", q) if len(p) >= 2]
            if parts:
                score += sum(0.8 for p in parts if p in text)
    return score


def nearby_spans(spans: list[Span], box: list[int], width: int, height: int, limit: int = 8) -> list[Span]:
    ranked = sorted(
        spans,
        key=lambda s: (
            0 if overlap(s.box, expand_box(box, width, height, 1.0, 1.0)) else 1,
            center_distance(s.box, box, width, height),
        ),
    )
    return ranked[:limit]


def textline_candidates(spans: list[Span], anomaly: Anomaly, width: int, height: int) -> list[Candidate]:
    queries = extract_text_queries(anomaly.reason)
    candidates: list[Candidate] = []
    scored = [(span_match_score(span, queries), span) for span in spans]
    for score, span in sorted(scored, key=lambda item: item[0], reverse=True)[:8]:
        if score <= 0:
            continue
        candidates.append(Candidate(f"text_match:{span.span_id}", span.box, "ocr_text_match", score + 2.0))

    span_ids = [str(v) for v in anomaly.normalized.get("span_ids") or []]
    if span_ids:
        span_by_id = {s.span_id: s for s in spans}
        boxes = [span_by_id[sid].box for sid in span_ids if sid in span_by_id]
        union = union_boxes(boxes)
        if union:
            candidates.append(Candidate("stage_span_union", union, "stage_span_ids", 7.5))
        for sid in span_ids[:6]:
            if sid in span_by_id:
                candidates.append(Candidate(f"stage_span:{sid}", span_by_id[sid].box, "stage_span_id", 6.0))

    normalized_box = project_box(anomaly.normalized.get("bbox"), width, height, "normalized-1000")
    if normalized_box:
        candidates.append(Candidate("normalized_bbox", normalized_box, "stage4_bbox", 5.0))

    for idx, span in enumerate(nearby_spans(spans, anomaly.box, width, height, limit=6), start=1):
        candidates.append(Candidate(f"nearby_span:{idx}", span.box, "nearby_ocr_span", 2.5 - idx * 0.1))
    return candidates


def local_patch_candidates(image: Image.Image, anomaly: Anomaly, width: int, height: int) -> list[Candidate]:
    try:
        from qwen_crop_patch_refine import detect_patches  # type: ignore
    except Exception:
        return []

    kinds: list[str]
    if anomaly.target_type == "redaction_block":
        kinds = ["dark"]
    elif anomaly.target_type == "style_color":
        kinds = ["red", "yellow", "dark"]
    else:
        kinds = ["dark"]

    local = expand_box(anomaly.box, width, height, 2.0, 2.0, min_pad=64)
    candidates: list[Candidate] = []
    for kind in kinds:
        for patch in detect_patches(image, kind)[:80]:
            if patch.box[2] - patch.box[0] < width * 0.02:
                continue
            if overlap(patch.box, local) or anomaly.target_type == "redaction_block":
                score = 5.0 + min(patch.score / 10000.0, 4.0) - center_distance(patch.box, anomaly.box, width, height)
                candidates.append(Candidate(f"patch:{kind}", patch.box, f"local_{kind}_patch", score))
    return candidates


def dedupe_candidates(candidates: list[Candidate], width: int, height: int, limit: int) -> list[Candidate]:
    current: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if candidate.box[2] <= candidate.box[0] or candidate.box[3] <= candidate.box[1]:
            continue
        if area(candidate.box) < 20:
            continue
        if area(candidate.box) > width * height * 0.35:
            continue
        if any(iou(candidate.box, kept.box) > 0.82 for kept in current):
            continue
        current.append(candidate)
        if len(current) >= limit:
            break
    return current


def make_anomalies(row: dict[str, Any], width: int, height: int) -> list[Anomaly]:
    report_items = parse_report_anomalies(str(row.get("raw_output") or ""))
    normalized = normalized_anomalies(row)
    anomalies: list[Anomaly] = []
    for idx, item in enumerate(report_items):
        box = clamp_box([float(v) for v in item["box"]], width, height)
        if not box:
            continue
        norm = normalized[idx] if idx < len(normalized) else {}
        context = str(item.get("context") or "")
        reason = str(item.get("reason") or context)
        target = classify_target(context)
        anomalies.append(Anomaly(idx, box, context, reason, target, norm))
    return anomalies


def generate_candidates(
    row: dict[str, Any],
    image: Image.Image,
    anomaly: Anomaly,
    spans: list[Span],
    max_candidates: int,
) -> list[Candidate]:
    width, height = image.size
    candidates = [
        Candidate("current_box", anomaly.box, "current_report_box", 1.0),
        Candidate("current_box_expanded", expand_box(anomaly.box, width, height, 0.35, 0.6), "current_report_box", 1.2),
    ]
    candidates.extend(textline_candidates(spans, anomaly, width, height))
    candidates.extend(local_patch_candidates(image, anomaly, width, height))
    return dedupe_candidates(candidates, width, height, max_candidates)


def crop_candidate(
    image: Image.Image,
    candidate: Candidate,
    cache_dir: Path,
    sample_id: str,
    anomaly_index: int,
    crop_pad: float,
) -> Candidate:
    width, height = image.size
    crop_box = expand_box(candidate.box, width, height, crop_pad, crop_pad, min_pad=32)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate.label)[:60]
    crop_path = cache_dir / "crops" / f"{sample_id}_a{anomaly_index:03d}_{safe_label}.jpg"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not crop_path.exists():
        crop = image.crop(tuple(crop_box))
        crop.save(crop_path, quality=92)
    candidate.crop_box = crop_box
    candidate.crop_path = str(crop_path)
    return candidate


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def verifier_prompt(anomaly: Anomaly, candidate: Candidate, crop_size: tuple[int, int]) -> str:
    return f"""You are verifying a local crop from a document-forensics pipeline.

Task: decide whether this crop contains the anomaly described below, and if yes return a tight bbox in CROP PIXEL COORDINATES.

Target type: {anomaly.target_type}
Candidate source: {candidate.source}
Crop size: width={crop_size[0]}, height={crop_size[1]}

Anomaly reason:
{anomaly.reason[:1400]}

Rules:
- Do not decide whether the whole document is forged.
- Only inspect this crop for the described local anomaly.
- If the crop does not clearly contain the described anomaly, answer NO.
- For text rendering/style issues, localize the smallest word/line/region that visibly shows the issue.
- For logical numeric/date/amount issues, localize the mentioned value or its table cell if visible.
- For redaction/block issues, localize the visible block(s), not the whole paragraph.
- Return only JSON, no markdown.

Schema:
{{"verdict":"YES|NO","target_type":"{anomaly.target_type}","local_bbox":[x1,y1,x2,y2],"confidence":0.0,"evidence":"short reason"}}
"""


def parse_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    match = JSON_FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def call_qwen_crop(
    *,
    crop_path: Path,
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: int,
    enable_thinking: bool = False,
) -> tuple[str, dict[str, Any]]:
    messages = [
        {"role": "system", "content": "You are a precise visual grounding verifier. Return strict JSON only."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(crop_path)}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.01,
        "enable_thinking": enable_thinking,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "\n".join(str(part.get("text", part)) for part in content)
            return str(content), result.get("usage", {})
        except Exception as exc:
            last_exc = exc
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"API call failed: {last_exc!r}")


def cache_key(sample_id: str, anomaly: Anomaly, candidate: Candidate, model: str) -> str:
    payload = {
        "sample_id": sample_id,
        "anomaly": anomaly.index,
        "target": anomaly.target_type,
        "candidate": candidate.label,
        "box": candidate.box,
        "crop_box": candidate.crop_box,
        "model": model,
        "reason_hash": hashlib.sha256(anomaly.reason.encode("utf-8")).hexdigest()[:16],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def verify_candidate(
    *,
    anomaly: Anomaly,
    candidate: Candidate,
    sample_id: str,
    model: str,
    api_key: str,
    cache_dir: Path,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    crop_path = Path(candidate.crop_path)
    crop_size = Image.open(crop_path).size
    key = cache_key(sample_id, anomaly, candidate, model)
    response_path = cache_dir / "responses" / f"{key}.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    if response_path.exists():
        cached = json.loads(response_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict):
            cached["cache_hit"] = True
        return cached

    prompt = verifier_prompt(anomaly, candidate, crop_size)
    raw, usage = call_qwen_crop(
        crop_path=crop_path,
        prompt=prompt,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    parsed = parse_json_object(raw)
    payload = {
        "sample_id": sample_id,
        "anomaly_index": anomaly.index,
        "candidate": candidate.__dict__,
        "model": model,
        "raw": raw,
        "parsed": parsed,
        "usage": usage,
        "cache_hit": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def map_local_bbox(local_bbox: Any, crop_box: list[int], width: int, height: int) -> list[int] | None:
    box = project_box(local_bbox, crop_box[2] - crop_box[0], crop_box[3] - crop_box[1], "pixel")
    if not box:
        return None
    return clamp_box(
        [box[0] + crop_box[0], box[1] + crop_box[1], box[2] + crop_box[0], box[3] + crop_box[1]],
        width,
        height,
    )


def verifier_accepts(
    payload: dict[str, Any],
    anomaly: Anomaly,
    candidate: Candidate,
    width: int,
    height: int,
    min_confidence: float,
) -> tuple[list[int] | None, dict[str, Any]]:
    parsed = payload.get("parsed") if isinstance(payload, dict) else None
    if not isinstance(parsed, dict):
        return None, {"accepted": False, "reason": "parse_error"}
    verdict = str(parsed.get("verdict") or "").upper()
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if verdict != "YES" or confidence < min_confidence:
        return None, {"accepted": False, "reason": "verdict_or_confidence", "verdict": verdict, "confidence": confidence}
    mapped = map_local_bbox(parsed.get("local_bbox"), candidate.crop_box, width, height)
    if not mapped:
        return None, {"accepted": False, "reason": "invalid_bbox", "verdict": verdict, "confidence": confidence}
    mapped_area = area(mapped)
    page_area = width * height
    if mapped_area < 20 or mapped_area > page_area * 0.30:
        return None, {"accepted": False, "reason": "area_gate", "area": mapped_area, "confidence": confidence}
    # The verifier should refine a candidate crop, not jump to an unrelated page area.
    candidate_neighborhood = expand_box(candidate.crop_box, width, height, 0.1, 0.1, min_pad=8)
    if not overlap(mapped, candidate_neighborhood):
        return None, {"accepted": False, "reason": "distance_gate", "confidence": confidence}
    # Stage span unions are inherited from the previous broad grounding stage.
    # For redaction/block anomalies they often describe an entire row; allowing
    # the verifier to pick any small dark-looking glyph inside that row caused
    # replacements that were less stable than the original row box.
    if anomaly.target_type == "redaction_block" and candidate.source == "stage_span_ids":
        return None, {
            "accepted": False,
            "reason": "stage_span_redaction_gate",
            "confidence": confidence,
            "source": candidate.source,
        }
    # Local visual detectors should be used as visual anchors. If the verifier
    # returns a box outside the detected patch, it is reacting to a neighboring
    # artifact visible in the padded crop rather than refining the candidate.
    if candidate.source.startswith("local_"):
        visual_neighborhood = expand_box(candidate.box, width, height, 0.25, 0.45, min_pad=16)
        if not overlap(mapped, visual_neighborhood):
            return None, {
                "accepted": False,
                "reason": "local_visual_anchor_gate",
                "confidence": confidence,
                "source": candidate.source,
            }
    return mapped, {
        "accepted": True,
        "verdict": verdict,
        "confidence": confidence,
        "evidence": parsed.get("evidence"),
        "target_type": parsed.get("target_type"),
    }


def local_choose_candidate(anomaly: Anomaly, candidates: list[Candidate], width: int, height: int) -> tuple[list[int] | None, dict[str, Any]]:
    if not candidates:
        return None, {"accepted": False, "reason": "no_candidates"}
    # Do not locally shrink broad layout anomalies; crop verifier is needed for those.
    if anomaly.target_type in {"layout_table"}:
        return None, {"accepted": False, "reason": "layout_requires_verifier"}
    old_area = area(anomaly.box) / max(width * height, 1)
    old_w = (anomaly.box[2] - anomaly.box[0]) / max(width, 1)
    old_h = (anomaly.box[3] - anomaly.box[1]) / max(height, 1)
    for candidate in candidates:
        if candidate.label == "current_box":
            continue
        candidate_area = area(candidate.box) / max(width * height, 1)
        candidate_w = (candidate.box[2] - candidate.box[0]) / max(width, 1)
        if candidate_area > 0.25:
            continue
        if anomaly.target_type == "redaction_block":
            if not candidate.source.startswith("local_"):
                continue
            # v63 already handles many redaction bars. Only accept extra local
            # patches that look like a real bar, not a tiny glyph stroke.
            if old_area > 0.03 or candidate_area < 0.0003 or candidate_w < 0.03:
                continue
            return candidate.box, {"accepted": True, "reason": "local_policy", "candidate": candidate.__dict__}

        if anomaly.target_type in {"render_pixel_blur", "style_color"}:
            if candidate.source != "ocr_text_match":
                continue
            # Only replace when the current box is clearly a broad text block.
            # Small existing boxes often already point to the right glyph/word.
            if old_area < 0.04 and old_h < 0.10:
                continue
            if candidate_area > old_area * 0.95:
                continue
            return candidate.box, {"accepted": True, "reason": "local_policy", "candidate": candidate.__dict__}

        if anomaly.target_type == "logical_numeric":
            if candidate.source != "ocr_text_match":
                continue
            if old_area < 0.06 and old_h < 0.12:
                continue
            if candidate_area > old_area * 0.95:
                continue
            return candidate.box, {"accepted": True, "reason": "local_policy", "candidate": candidate.__dict__}
    return None, {"accepted": False, "reason": "no_policy_candidate"}


def replace_groundings(report: str, boxes: list[list[int]]) -> tuple[str, int]:
    idx = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        if idx >= len(boxes):
            return match.group(0)
        box = boxes[idx]
        idx += 1
        return f"{match.group(1)}{box}"

    return GROUNDING_RE.sub(repl, report), idx


def append_grounding_anomalies(report: str, entries: list[dict[str, Any]]) -> str:
    if not entries:
        return report
    lines: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        box = entry["box"]
        target = str(entry.get("target_type") or "visual_candidate")
        source_index = int(entry.get("source_anomaly_index") or 0) + 1
        evidence = re.sub(r"\s+", " ", str(entry.get("evidence") or "")).strip()
        if len(evidence) > 220:
            evidence = evidence[:217].rstrip() + "..."
        if not evidence:
            evidence = "The crop verifier confirmed this localized region as consistent with the nearby anomaly description."
        lines.extend(
            [
                f"### ANOMALY_CROP_EXTRA_{idx:03d}: Verified Crop Candidate ({target})",
                f"[GROUNDING]:{box}",
                f"[REASON]: Additional localized evidence for ANOMALY_{source_index:03d}. {evidence}",
                "",
            ]
        )
    block = "\n" + "\n".join(lines)
    marker = re.search(r"\n---\s*\n\s*## SUMMARY", report, re.IGNORECASE)
    if marker:
        return report[: marker.start()] + block + report[marker.start() :]
    end_marker = re.search(r"\n---\s*\n\s*\*\*END OF REPORT\*\*", report, re.IGNORECASE)
    if end_marker:
        return report[: end_marker.start()] + block + report[end_marker.start() :]
    return report.rstrip() + block


def append_candidate_ok(
    box: list[int],
    anomaly: Anomaly,
    candidate: Candidate,
    existing_boxes: list[list[int]],
    width: int,
    height: int,
) -> tuple[bool, str]:
    if candidate.label in {"current_box", "current_box_expanded", "stage_span_union"}:
        return False, "broad_candidate"
    if candidate.source == "stage_span_ids":
        return False, "stage_union_source"
    box_area = area(box)
    page_area = width * height
    if box_area < 20:
        return False, "tiny_box"
    max_ratio = 0.12 if anomaly.target_type == "layout_table" else 0.08
    if box_area > page_area * max_ratio:
        return False, "append_area_gate"
    if is_duplicate_box(box, existing_boxes):
        return False, "duplicate_box"
    return True, "accepted"


def process_row(
    row: dict[str, Any],
    *,
    debug_root: Path,
    mode: str,
    api_key: str | None,
    model: str,
    cache_dir: Path,
    max_candidates: int,
    max_tokens: int,
    timeout: int,
    crop_pad: float,
    min_confidence: float,
    coord_mode: str,
    append_accepted: bool,
    max_appends_per_row: int,
    max_appends_per_anomaly: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(row)
    parsed = row.get("parsed") or {}
    if str(parsed.get("conclusion") or "").upper() != "FORGED":
        return out, {"applied": False, "reason": "non_forged"}
    image_path = resolve_image_path(row, debug_root)
    if not image_path:
        return out, {"applied": False, "reason": "missing_image"}
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    anomalies = make_anomalies(row, width, height)
    if not anomalies:
        return out, {"applied": False, "reason": "no_anomalies"}
    spans = collect_spans(row, width, height, coord_mode)
    sample_id = str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)

    new_boxes = [a.box for a in anomalies]
    append_entries: list[dict[str, Any]] = []
    anomaly_meta: list[dict[str, Any]] = []
    replaced = 0
    api_calls = 0
    for anomaly in anomalies:
        candidates = [
            crop_candidate(image, candidate, cache_dir, sample_id, anomaly.index, crop_pad)
            for candidate in generate_candidates(row, image, anomaly, spans, max_candidates)
        ]
        chosen_box: list[int] | None = None
        chosen_meta: dict[str, Any] = {"accepted": False}
        if mode in {"local", "hybrid"}:
            chosen_box, chosen_meta = local_choose_candidate(anomaly, candidates, width, height)
        if (chosen_box is None) and mode in {"api", "hybrid"}:
            if api_key is None:
                raise SystemExit("--mode api/hybrid requires --api-key-file or DASHSCOPE_API_KEY")
            attempts = []
            # Broad current/stage boxes make the verifier prone to selecting a
            # plausible but unrelated defect inside a large crop. Ask the VLM
            # to verify fine-grained OCR/local candidates first, then fall back
            # only if no focused candidate exists.
            api_candidates = [c for c in candidates if c.label not in {"current_box", "current_box_expanded"}]
            if not api_candidates:
                api_candidates = candidates
            accepted_for_append = 0
            for candidate in api_candidates[:max_candidates]:
                payload = verify_candidate(
                    anomaly=anomaly,
                    candidate=candidate,
                    sample_id=sample_id,
                    model=model,
                    api_key=api_key,
                    cache_dir=cache_dir,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                api_calls += 0 if payload.get("cache_hit") else 1
                mapped, gate_meta = verifier_accepts(payload, anomaly, candidate, width, height, min_confidence)
                attempts.append({"candidate": candidate.__dict__, "gate": gate_meta})
                if mapped is not None:
                    if chosen_box is None:
                        chosen_box = mapped
                        chosen_meta = {"accepted": True, "reason": "api_verifier", "candidate": candidate.__dict__, "gate": gate_meta}
                    elif append_accepted and accepted_for_append < max_appends_per_anomaly and len(append_entries) < max_appends_per_row:
                        ok, append_reason = append_candidate_ok(
                            mapped,
                            anomaly,
                            candidate,
                            new_boxes + [entry["box"] for entry in append_entries],
                            width,
                            height,
                        )
                        attempts[-1]["append_gate"] = append_reason
                        if ok:
                            append_entries.append(
                                {
                                    "box": mapped,
                                    "target_type": anomaly.target_type,
                                    "source_anomaly_index": anomaly.index,
                                    "candidate": candidate.__dict__,
                                    "confidence": gate_meta.get("confidence"),
                                    "evidence": gate_meta.get("evidence"),
                                }
                            )
                            accepted_for_append += 1
                    if not append_accepted:
                        break
            if chosen_box is None:
                chosen_meta = {"accepted": False, "reason": "api_no_candidate", "attempts": attempts[:5]}

        if chosen_box is not None and chosen_box != anomaly.box:
            new_boxes[anomaly.index] = chosen_box
            replaced += 1
        anomaly_meta.append(
            {
                "index": anomaly.index,
                "target_type": anomaly.target_type,
                "old_box": anomaly.box,
                "new_box": new_boxes[anomaly.index],
                "candidate_count": len(candidates),
                "decision": chosen_meta,
            }
        )

    report = str(row.get("raw_output") or "")
    new_report, count = replace_groundings(report, new_boxes)
    if append_entries:
        new_report = append_grounding_anomalies(new_report, append_entries)
    if (replaced or append_entries) and count == len(new_boxes) and new_report != report:
        setup_debug_import(debug_root)
        from postprocess import parse_cct_report  # type: ignore

        out["raw_output"] = new_report
        out["parsed"] = parse_cct_report(new_report)
    stage_outputs = dict(out.get("stage_outputs") or {})
    stage_outputs["qwen_pipe_text_crop_verify"] = {
        "applied": bool(replaced or append_entries),
        "mode": mode,
        "boxes_replaced": replaced,
        "boxes_appended": len(append_entries),
        "api_calls_estimate": api_calls,
        "anomalies": anomaly_meta,
        "append_entries": append_entries,
        "policy": "Typed OCR/textline crop localization; Qwen crop verifier sees only crop, current reason, and target type. No GT fields are used.",
    }
    out["stage_outputs"] = stage_outputs
    return out, stage_outputs["qwen_pipe_text_crop_verify"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL), help="Used only for optional low-loc sample selection.")
    parser.add_argument("--low-loc-threshold", type=float, default=0.02)
    parser.add_argument("--select-low-loc-limit", type=int, default=0, help="0 disables eval-driven low-loc selection.")
    parser.add_argument("--sample-id", action="append", default=[], help="Explicit sample id to process; repeatable.")
    parser.add_argument("--write-subset-only", action="store_true")
    parser.add_argument("--mode", choices=["local", "api", "hybrid"], default="local")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--api-key-file", default="/Users/penpen/Desktop/api-key.txt")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--max-candidates-per-anomaly", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--crop-pad", type=float, default=0.45)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--coord-mode", choices=["auto", "normalized-1000", "pixel"], default="normalized-1000")
    parser.add_argument("--append-accepted", action="store_true", help="Append extra verifier-confirmed fine-grained boxes instead of only replacing existing boxes.")
    parser.add_argument("--max-appends-per-row", type=int, default=2)
    parser.add_argument("--max-appends-per-anomaly", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_pipe_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected: set[str] = set(args.sample_id or [])
    if args.select_low_loc_limit > 0:
        selected.update(
            load_eval_selection(
                resolve_pipe_path(args.eval_json),
                threshold=args.low_loc_threshold,
                limit=args.select_low_loc_limit,
            )
        )
    api_key = None
    if args.mode in {"api", "hybrid"}:
        api_key = load_api_key(args.api_key_file)

    stats: dict[str, Any] = {
        "rows": 0,
        "rows_written": 0,
        "rows_processed": 0,
        "rows_changed": 0,
        "boxes_replaced": 0,
        "boxes_appended": 0,
        "selected_count": len(selected),
        "write_subset_only": args.write_subset_only,
        "mode": args.mode,
        "target_type_counts": {},
    }

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            sample_id = str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)
            should_process = not selected or sample_id in selected
            if should_process:
                out, meta = process_row(
                    row,
                    debug_root=debug_root,
                    mode=args.mode,
                    api_key=api_key,
                    model=args.model,
                    cache_dir=cache_dir,
                    max_candidates=args.max_candidates_per_anomaly,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    crop_pad=args.crop_pad,
                    min_confidence=args.min_confidence,
                    coord_mode=args.coord_mode,
                    append_accepted=args.append_accepted,
                    max_appends_per_row=args.max_appends_per_row,
                    max_appends_per_anomaly=args.max_appends_per_anomaly,
                )
                stats["rows_processed"] += 1
                if meta.get("applied"):
                    stats["rows_changed"] += 1
                    stats["boxes_replaced"] += int(meta.get("boxes_replaced") or 0)
                    stats["boxes_appended"] += int(meta.get("boxes_appended") or 0)
                for anomaly in meta.get("anomalies") or []:
                    target = str(anomaly.get("target_type") or "unknown")
                    stats["target_type_counts"][target] = stats["target_type_counts"].get(target, 0) + 1
            else:
                out = row
            if (not args.write_subset_only) or should_process:
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
