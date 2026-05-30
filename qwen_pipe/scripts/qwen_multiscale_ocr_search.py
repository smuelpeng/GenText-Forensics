#!/usr/bin/env python3
"""Multi-grid / multi-scale OCR search for dispersed forgery localization.

This postprocess is GT-blind at inference time. It addresses the common VLM
failure where a full-page report names one or two representative anomalies but
misses other tampered regions. The script searches OCR rows, row pyramids, and
local visual windows, then appends compact grounding boxes to existing FORGED
reports without changing the verdict.

Optional API modes verify generated crops with Qwen. The verifier sees only the
crop image, OCR text inside the crop, and the local search task. GT labels,
reports, masks, and eval scores are never used in prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v77_evidence_multibox_300.jsonl"
DEFAULT_EVAL = PIPE_ROOT / "outputs/eval/qwen_pipe_v77_evidence_multibox_300.json"
DEFAULT_CACHE = PIPE_ROOT / "outputs/cache/multiscale_ocr_search"
DEFAULT_VERIFIER_KEY = "/Users/penpen/Desktop/api-key.txt"
DEFAULT_SEMANTIC_KEY = "/Users/penpen/Desktop/api-key.txt"
EMBEDDINGS_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[([^\[\]]+)\]", re.IGNORECASE)

sys.path.insert(0, str(PIPE_ROOT / "scripts"))
from qwen_evidence_multibox_refine import iou, report_boxes  # noqa: E402
from qwen_issue_refine import (  # noqa: E402
    BENIGN_RESCUE_EXCLUDE_TERMS,
    HARD_VISUAL_TERMS,
    evidence_candidates,
    has_any,
    parse_box,
    parsed_conclusion,
    project_box,
)
from qwen_ocr_cluster_refine import build_rows  # noqa: E402
from qwen_text_crop_verify import (  # noqa: E402
    Span,
    area,
    call_qwen_crop,
    clamp_box,
    collect_spans,
    expand_box,
    extract_text_queries,
    image_to_data_url,
    map_local_bbox,
    overlap,
    parse_json_object,
    resolve_image_path,
    resolve_pipe_path,
    setup_debug_import,
    union_boxes,
)


LOCAL_VISUAL_CATEGORIES = {
    "rendering_artifact",
    "color_mismatch",
    "layout_inconsistency",
    "font_mismatch",
    "copy_paste_boundary",
}

LOGICAL_CUES = (
    "logical",
    "contradiction",
    "inconsistent",
    "date",
    "amount",
    "total",
    "sum",
    "number",
    "reference",
    "编号",
    "金额",
    "日期",
    "数字",
    "合计",
    "总计",
    "矛盾",
    "不一致",
    "อ้างอิง",
    "วันที่",
    "จำนวน",
    "مجموع",
    "تاريخ",
    "رقم",
)

NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,:/-]\d+)*(?:\s?%|\s?万元|\s?万|年|月|日)?")
WEIRD_RE = re.compile(r"[�□■�]{1,}|[^\w\s\u4e00-\u9fff\u0e00-\u0e7f\u0600-\u06ff.,:;()/\\-]{3,}")


@dataclass
class SearchCandidate:
    label: str
    box: list[int]
    target_type: str
    source: str
    score: float
    evidence: str
    ocr_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_low_loc_selection(eval_json: Path, threshold: float, limit: int) -> set[str]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    samples = [
        s
        for s in data.get("samples") or []
        if s.get("gt_label") == "FORGED"
        and s.get("pred_label") == "FORGED"
        and float(s.get("loc_score") or 0.0) < threshold
    ]
    samples.sort(key=lambda s: (float(s.get("loc_score") or 0.0), str(s.get("sample_id") or "")))
    if limit > 0:
        samples = samples[:limit]
    return {str(s.get("sample_id")) for s in samples}


def report_text(row: dict[str, Any]) -> str:
    return str(row.get("raw_output") or "")


def visual_context(row: dict[str, Any]) -> str:
    parts = [report_text(row)]
    parsed = (((row.get("stage_outputs") or {}).get("evidence_candidates") or {}).get("parsed") or {})
    for key in ("visual_candidates", "logical_candidates"):
        for cand in parsed.get(key) or []:
            if isinstance(cand, dict):
                parts.extend(str(cand.get(k) or "") for k in ("category", "evidence", "notes", "reason"))
    return "\n".join(parts)


def box_area_ratio(box: list[int], width: int, height: int) -> float:
    return area(box) / max(1, width * height)


def crop_stats(image: Image.Image, box: list[int]) -> dict[str, float]:
    crop = image.crop(tuple(box)).convert("RGB")
    gray = crop.convert("L")
    hist = gray.histogram()
    total = max(1, sum(hist))
    dark = sum(hist[:55]) / total
    mid_dark = sum(hist[:110]) / total
    light = sum(hist[210:]) / total
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edge)
    gray_stat = ImageStat.Stat(gray)
    rgb_stat = ImageStat.Stat(crop)
    channels = rgb_stat.mean
    color_spread = (max(channels) - min(channels)) / 255.0 if channels else 0.0
    return {
        "dark_ratio": dark,
        "mid_dark_ratio": mid_dark,
        "light_ratio": light,
        "edge_mean": (edge_stat.mean[0] if edge_stat.mean else 0.0) / 255.0,
        "gray_std": (gray_stat.stddev[0] if gray_stat.stddev else 0.0) / 255.0,
        "color_spread": color_spread,
    }


def row_text(row: Any) -> str:
    return " ".join(str(span.text or "") for span in row.spans)


def row_box(row: Any) -> list[int]:
    return list(row.box)


def row_confidence_score(text: str, context: str) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []
    nums = NUMERIC_RE.findall(text or "")
    if nums and has_any(context, LOGICAL_CUES):
        score += min(3.5, 1.2 + 0.35 * len(nums))
        reasons.append(f"numeric/date tokens={len(nums)}")
    if WEIRD_RE.search(text or ""):
        score += 2.5
        reasons.append("weird glyph sequence")
    return score, "; ".join(reasons)


def build_row_windows(rows: list[Any], radius_values: tuple[int, ...]) -> list[tuple[str, list[int], str]]:
    windows: list[tuple[str, list[int], str]] = []
    seen: set[tuple[int, int, int]] = set()
    n = len(rows)
    for idx in range(n):
        for radius in radius_values:
            lo = max(0, idx - radius)
            hi = min(n - 1, idx + radius)
            key = (lo, hi, radius)
            if key in seen:
                continue
            seen.add(key)
            box = union_boxes([row_box(rows[j]) for j in range(lo, hi + 1)])
            if box:
                text = " ".join(row_text(rows[j]) for j in range(lo, hi + 1))
                windows.append((f"rows:{lo}-{hi}:r{radius}", box, text))
    return windows


def existing_candidate_boxes(row: dict[str, Any], width: int, height: int, coord_mode: str) -> list[SearchCandidate]:
    out: list[SearchCandidate] = []
    for cand in evidence_candidates(row):
        category = str(cand.get("category") or "").lower()
        if category not in LOCAL_VISUAL_CATEGORIES and cand.get("_source_list") != "logical_candidates":
            continue
        raw_box = parse_box(cand.get("bbox"))
        if not raw_box:
            continue
        box = project_box(raw_box, width, height, coord_mode)
        if not box:
            continue
        text = " ".join(str(cand.get(k) or "") for k in ("category", "evidence", "notes", "reason"))
        try:
            conf = float(cand.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if has_any(text, BENIGN_RESCUE_EXCLUDE_TERMS):
            continue
        hard = has_any(text, HARD_VISUAL_TERMS)
        logical = cand.get("_source_list") == "logical_candidates" or has_any(text, LOGICAL_CUES)
        if not hard and not logical and category not in {"font_mismatch", "copy_paste_boundary"}:
            continue
        score = conf + (2.5 if hard else 0.0) + (1.4 if logical else 0.0)
        out.append(
            SearchCandidate(
                label=f"stage_candidate:{cand.get('id') or len(out)+1}",
                box=box,
                target_type="logical_numeric" if logical else category or "local_visual",
                source="existing_evidence_candidate",
                score=score,
                evidence=re.sub(r"\s+", " ", str(cand.get("evidence") or text)).strip()[:700],
                ocr_text="",
                meta={"candidate": cand},
            )
        )
    return out


def evidence_coverage(existing: list[list[int]], row: dict[str, Any], width: int, height: int, coord_mode: str) -> float:
    if not existing:
        return 0.0
    candidate_boxes: list[list[int]] = []
    for cand in evidence_candidates(row):
        raw_box = parse_box(cand.get("bbox"))
        if not raw_box:
            continue
        box = project_box(raw_box, width, height, coord_mode)
        if box:
            candidate_boxes.append(box)
    if not candidate_boxes:
        return 0.0
    covered = sum(1 for box in existing if any(iou(box, cand_box) > 0.10 for cand_box in candidate_boxes))
    return covered / max(1, len(existing))


def multiscale_candidates(
    row: dict[str, Any],
    image: Image.Image,
    spans: list[Span],
    *,
    coord_mode: str,
    row_radii: tuple[int, ...],
    max_window_area_ratio: float,
) -> list[SearchCandidate]:
    width, height = image.size
    context = visual_context(row)
    queries = extract_text_queries(context)
    rows = build_rows(spans)
    candidates = existing_candidate_boxes(row, width, height, coord_mode)

    all_windows = build_row_windows(rows, row_radii)
    for label, box, text in all_windows:
        if box_area_ratio(box, width, height) > max_window_area_ratio:
            continue
        stats = crop_stats(image, expand_box(box, width, height, 0.04, 0.18, min_pad=6))
        score, why = row_confidence_score(text, context)
        q_hits = []
        compact_text = re.sub(r"\s+", "", text.lower())
        for query in queries:
            q = query.lower()
            qc = re.sub(r"\s+", "", q)
            if q and (q in text.lower() or qc in compact_text):
                q_hits.append(query)
        if q_hits:
            score += min(6.0, 2.0 + len(q_hits))
        if stats["dark_ratio"] > 0.045 and stats["mid_dark_ratio"] > 0.08:
            score += min(5.0, 1.0 + stats["dark_ratio"] * 35.0)
        if stats["edge_mean"] > 0.09 and stats["gray_std"] > 0.18:
            score += 1.6
        if stats["color_spread"] > 0.18:
            score += 1.3
        if score < 3.0:
            continue
        target = "logical_numeric" if has_any(context, LOGICAL_CUES) and NUMERIC_RE.search(text or "") else "render_pixel_blur"
        evidence_bits = []
        if why:
            evidence_bits.append(why)
        if q_hits:
            evidence_bits.append("matches report/context query: " + ", ".join(q_hits[:4]))
        evidence_bits.append(
            "window stats dark={dark_ratio:.3f}, edge={edge_mean:.3f}, color={color_spread:.3f}".format(**stats)
        )
        candidates.append(
            SearchCandidate(
                label=f"grid_{label}",
                box=box,
                target_type=target,
                source="multiscale_ocr_row_window",
                score=score,
                evidence="; ".join(evidence_bits),
                ocr_text=text[:800],
                meta={"stats": stats, "query_hits": q_hits[:8]},
            )
        )

    # Page strips catch header/footer artifacts that the model often ignores.
    strip_specs = [
        ("page_header", [0, 0, width, int(height * 0.16)]),
        ("page_footer", [0, int(height * 0.84), width, height]),
        ("left_margin", [0, 0, int(width * 0.18), height]),
        ("right_margin", [int(width * 0.82), 0, width, height]),
    ]
    lower_context = context.lower()
    for label, box in strip_specs:
        if label.endswith("header") and not any(k in lower_context for k in ("header", "logo", "页眉", "抬头")):
            continue
        if label.endswith("footer") and not any(k in lower_context for k in ("footer", "page", "页脚", "页码")):
            continue
        stats = crop_stats(image, box)
        score = 0.0
        if stats["dark_ratio"] > 0.025:
            score += 2.5
        if stats["edge_mean"] > 0.08:
            score += 1.2
        if score >= 2.8:
            candidates.append(
                SearchCandidate(
                    label=f"strip_{label}",
                    box=box,
                    target_type="render_pixel_blur",
                    source="page_region_strip",
                    score=score,
                    evidence="header/footer/side strip with local visual outlier stats",
                    meta={"stats": stats},
                )
            )
    return candidates


def dedupe_select(
    candidates: list[SearchCandidate],
    existing: list[list[int]],
    *,
    width: int,
    height: int,
    min_score: float,
    max_area_ratio: float,
    duplicate_iou: float,
    max_extra_boxes: int,
) -> list[SearchCandidate]:
    selected: list[SearchCandidate] = []
    occupied = list(existing)
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        if cand.score < min_score:
            continue
        if cand.box[2] <= cand.box[0] or cand.box[3] <= cand.box[1]:
            continue
        if area(cand.box) < 20 or box_area_ratio(cand.box, width, height) > max_area_ratio:
            continue
        if any(iou(cand.box, box) >= duplicate_iou for box in occupied):
            continue
        selected.append(cand)
        occupied.append(cand.box)
        if len(selected) >= max_extra_boxes:
            break
    return selected


def api_candidate_rank(cand: SearchCandidate) -> tuple[int, float]:
    source_priority = {
        "existing_evidence_candidate": 0,
        "page_region_strip": 1,
        "multiscale_ocr_row_window": 2,
    }.get(cand.source, 3)
    # Within the same source, keep the strongest local evidence first.
    return (source_priority, -cand.score)


def rank_api_candidates(candidates: list[SearchCandidate], mode: str) -> list[SearchCandidate]:
    if mode == "stage-first":
        return sorted(candidates, key=api_candidate_rank)
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def semantic_query(row: dict[str, Any]) -> str:
    context = re.sub(r"\s+", " ", visual_context(row)).strip()
    queries = extract_text_queries(context)
    query_bits = [
        "Document forgery localization. Rank OCR windows that likely contain the visible tampered region.",
        "Prefer windows with altered dates, amounts, IDs, totals, references, garbled text, blur, redaction, font/color mismatch, table row/cell mismatch, or copy-paste artifacts.",
    ]
    if queries:
        query_bits.append("Mentioned tokens: " + " ; ".join(queries[:16]))
    if context:
        query_bits.append("Current report/evidence context: " + context[:1800])
    return "\n".join(query_bits)


def semantic_doc(cand: SearchCandidate) -> str:
    parts = [
        f"candidate={cand.label}",
        f"type={cand.target_type}",
        f"source={cand.source}",
        f"local_score={cand.score:.3f}",
    ]
    if cand.ocr_text:
        parts.append("OCR window text: " + re.sub(r"\s+", " ", cand.ocr_text).strip()[:1500])
    if cand.evidence:
        parts.append("Local evidence: " + re.sub(r"\s+", " ", cand.evidence).strip()[:1000])
    return "\n".join(parts)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def api_key_cache_tag(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def ranking_cache_key(sample_id: str, mode: str, model: str, query: str, docs: list[str], api_key_tag: str) -> str:
    payload = {
        "sample_id": sample_id,
        "mode": mode,
        "model": model,
        "api_key_tag": api_key_tag,
        "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "doc_hashes": [hashlib.sha256(d.encode("utf-8")).hexdigest() for d in docs],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def call_embedding_rank(
    *,
    sample_id: str,
    row: dict[str, Any],
    candidates: list[SearchCandidate],
    model: str,
    api_key: str,
    cache_dir: Path,
    timeout: int,
) -> tuple[list[SearchCandidate], dict[str, Any]]:
    if not candidates:
        return [], {"mode": "embedding", "candidate_count": 0}
    query = semantic_query(row)
    docs = [semantic_doc(c) for c in candidates]
    cache_path = cache_dir / "rankings" / f"{ranking_cache_key(sample_id, 'embedding', model, query, docs, api_key_cache_tag(api_key))}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["cache_hit"] = True
    else:
        req_payload = {"model": model, "input": [query] + docs, "encoding_format": "float"}
        req = urllib.request.Request(
            EMBEDDINGS_URL,
            data=json.dumps(req_payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        vectors = [item["embedding"] for item in sorted(result.get("data") or [], key=lambda x: x.get("index", 0))]
        query_vec = vectors[0]
        scores = [cosine(query_vec, vec) for vec in vectors[1:]]
        payload = {
            "mode": "embedding",
            "model": model,
            "api_key_tag": api_key_cache_tag(api_key),
            "scores": scores,
            "usage": result.get("usage", {}),
            "cache_hit": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    scores = payload.get("scores") or []
    ranked: list[SearchCandidate] = []
    for cand, score in zip(candidates, scores):
        cand.meta["semantic_score"] = float(score)
        cand.meta["semantic_ranker"] = model
        ranked.append(cand)
    ranked.sort(key=lambda c: (float(c.meta.get("semantic_score") or 0.0), c.score), reverse=True)
    payload["candidate_count"] = len(candidates)
    return ranked, payload


def call_rerank_rank(
    *,
    sample_id: str,
    row: dict[str, Any],
    candidates: list[SearchCandidate],
    model: str,
    api_key: str,
    cache_dir: Path,
    timeout: int,
) -> tuple[list[SearchCandidate], dict[str, Any]]:
    if not candidates:
        return [], {"mode": "rerank", "candidate_count": 0}
    query = semantic_query(row)
    docs = [semantic_doc(c) for c in candidates]
    cache_path = cache_dir / "rankings" / f"{ranking_cache_key(sample_id, 'rerank', model, query, docs, api_key_cache_tag(api_key))}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["cache_hit"] = True
    else:
        req_payload = {
            "model": model,
            "query": query,
            "documents": docs,
            "top_n": len(docs),
            "instruct": "Rank OCR/text windows by whether they contain the local forged or manipulated region described by the document-forensics evidence.",
        }
        req = urllib.request.Request(
            RERANK_URL,
            data=json.dumps(req_payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        scores = [0.0 for _ in docs]
        for item in result.get("results") or []:
            idx = int(item.get("index", -1))
            if 0 <= idx < len(scores):
                scores[idx] = float(item.get("relevance_score") or 0.0)
        payload = {
            "mode": "rerank",
            "model": model,
            "api_key_tag": api_key_cache_tag(api_key),
            "scores": scores,
            "usage": result.get("usage", {}),
            "cache_hit": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    scores = payload.get("scores") or []
    ranked: list[SearchCandidate] = []
    for cand, score in zip(candidates, scores):
        cand.meta["semantic_score"] = float(score)
        cand.meta["semantic_ranker"] = model
        ranked.append(cand)
    ranked.sort(key=lambda c: (float(c.meta.get("semantic_score") or 0.0), c.score), reverse=True)
    payload["candidate_count"] = len(candidates)
    return ranked, payload


def crop_candidate(image: Image.Image, cand: SearchCandidate, cache_dir: Path, sample_id: str) -> tuple[Path, list[int]]:
    width, height = image.size
    crop_box = expand_box(cand.box, width, height, 0.35, 0.35, min_pad=32)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", cand.label)[:90]
    crop_path = cache_dir / "crops" / f"{sample_id}_{safe}.jpg"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not crop_path.exists():
        image.crop(tuple(crop_box)).save(crop_path, quality=92)
    return crop_path, crop_box


def api_prompt(cand: SearchCandidate, crop_size: tuple[int, int]) -> str:
    return f"""You are a local document-forensics verifier.

Inspect this crop only. It comes from a multi-scale OCR search window.

Target type: {cand.target_type}
Candidate source: {cand.source}
Crop size: width={crop_size[0]}, height={crop_size[1]}
OCR text in/near this window:
{cand.ocr_text[:1200]}

Local evidence hypothesis:
{cand.evidence[:900]}

Return JSON only:
{{"verdict":"YES|NO","target_type":"{cand.target_type}","local_boxes":[[x1,y1,x2,y2]],"confidence":0.0,"evidence":"short reason"}}

Rules:
- Do not decide whether the whole page is forged.
- Return YES only for visible local tampering signs: redaction, pasted/blurred/garbled text, abnormal font/color/style, table/row/cell mismatch, or directly suspicious numeric/date/reference value.
- Ignore normal template design, decoration, icons, logos, photo cutouts, watermarks, and ordinary handwriting-style fonts.
- For YES, localize the smallest visible word, line, block, or cell that shows the issue.
"""


def api_cache_key(sample_id: str, cand: SearchCandidate, model: str) -> str:
    payload = {
        "sample_id": sample_id,
        "label": cand.label,
        "box": cand.box,
        "target_type": cand.target_type,
        "source": cand.source,
        "model": model,
        "enable_thinking": cand.meta.get("_enable_thinking", False),
        "evidence_hash": hashlib.sha256(cand.evidence.encode("utf-8")).hexdigest()[:16],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def verify_api_candidate(
    *,
    image: Image.Image,
    cand: SearchCandidate,
    sample_id: str,
    model: str,
    api_key: str,
    cache_dir: Path,
    max_tokens: int,
    timeout: int,
    min_confidence: float,
    enable_thinking: bool,
) -> tuple[list[list[int]], dict[str, Any]]:
    cand.meta["_enable_thinking"] = bool(enable_thinking)
    crop_path, crop_box = crop_candidate(image, cand, cache_dir, sample_id)
    response_path = cache_dir / "responses" / f"{api_cache_key(sample_id, cand, model)}.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    if response_path.exists():
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        payload["cache_hit"] = True
    else:
        prompt = api_prompt(cand, Image.open(crop_path).size)
        raw, usage = call_qwen_crop(
            crop_path=crop_path,
            prompt=prompt,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            enable_thinking=enable_thinking,
        )
        payload = {
            "raw": raw,
            "parsed": parse_json_object(raw),
            "usage": usage,
            "candidate": cand.__dict__,
            "crop_box": crop_box,
            "enable_thinking": enable_thinking,
            "cache_hit": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    parsed = payload.get("parsed") if isinstance(payload, dict) else None
    accepted: list[list[int]] = []
    if isinstance(parsed, dict):
        verdict = str(parsed.get("verdict") or "").upper()
        try:
            conf = float(parsed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        raw_boxes = parsed.get("local_boxes")
        if raw_boxes is None and parsed.get("local_bbox") is not None:
            raw_boxes = [parsed.get("local_bbox")]
        if verdict == "YES" and conf >= min_confidence and isinstance(raw_boxes, list):
            width, height = image.size
            for raw_box in raw_boxes[:3]:
                mapped = map_local_bbox(raw_box, crop_box, width, height)
                if mapped and area(mapped) >= 20 and box_area_ratio(mapped, width, height) <= 0.12:
                    accepted.append(mapped)
    return accepted, payload


def insert_extra_anomalies(report: str, extras: list[SearchCandidate]) -> str:
    if not extras:
        return report
    block: list[str] = []
    for idx, cand in enumerate(extras, start=1):
        evidence = re.sub(r"\s+", " ", cand.evidence).strip()
        block.extend(
            [
                f"### ANOMALY_SEARCH_{idx:03d}: Multi-scale OCR Search ({cand.target_type})",
                f"[GROUNDING]:{cand.box}",
                f"[REASON]: {evidence} This extra grounding was added by GT-blind multi-grid OCR search to improve recall over dispersed tamper regions.",
                "",
            ]
        )
    extra_text = "\n".join(block)
    marker = re.search(r"\n\s*-{3,}\s*\n\s*##\s*SUMMARY|\n\s*##\s*SUMMARY", report, re.IGNORECASE)
    if marker:
        return report[: marker.start()] + "\n\n" + extra_text + report[marker.start():]
    return report.rstrip() + "\n\n" + extra_text


def process_row(
    row: dict[str, Any],
    *,
    debug_root: Path,
    mode: str,
    api_key: str | None,
    cache_dir: Path,
    model: str,
    max_tokens: int,
    timeout: int,
    min_confidence: float,
    coord_mode: str,
    row_radii: tuple[int, ...],
    min_score: float,
    max_area_ratio: float,
    max_window_area_ratio: float,
    max_total_boxes: int,
    max_existing_boxes_to_process: int,
    max_existing_evidence_coverage: float,
    max_extra_boxes: int,
    duplicate_iou: float,
    max_api_candidates_per_sample: int,
    api_ranking: str,
    enable_thinking: bool,
    semantic_api_key: str | None,
    embedding_model: str,
    rerank_model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(row)
    if parsed_conclusion(row) != "FORGED":
        return out, {"applied": False, "reason": "non_forged"}
    image_path = resolve_image_path(row, debug_root)
    if not image_path:
        return out, {"applied": False, "reason": "missing_image"}
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    existing = report_boxes(report_text(row))
    if len(existing) >= max_total_boxes:
        return out, {"applied": False, "reason": "enough_existing_boxes", "existing_boxes": len(existing)}
    if len(existing) > max_existing_boxes_to_process:
        return out, {"applied": False, "reason": "too_many_existing_boxes_for_search", "existing_boxes": len(existing)}
    ev_cov = evidence_coverage(existing, row, width, height, coord_mode)
    if ev_cov > max_existing_evidence_coverage:
        return out, {
            "applied": False,
            "reason": "existing_boxes_already_cover_evidence",
            "existing_boxes": len(existing),
            "evidence_coverage": ev_cov,
        }
    spans = collect_spans(row, width, height, coord_mode)
    if not spans:
        return out, {"applied": False, "reason": "no_ocr_spans", "existing_boxes": len(existing)}

    sample_id = str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)
    candidates = multiscale_candidates(
        row,
        image,
        spans,
        coord_mode=coord_mode,
        row_radii=row_radii,
        max_window_area_ratio=max_window_area_ratio,
    )
    budget = max(0, min(max_extra_boxes, max_total_boxes - len(existing)))
    local_selected = dedupe_select(
        candidates,
        existing,
        width=width,
        height=height,
        min_score=min_score,
        max_area_ratio=max_area_ratio,
        duplicate_iou=duplicate_iou,
        max_extra_boxes=budget,
    )

    selected = list(local_selected)
    api_calls = 0
    api_attempts: list[dict[str, Any]] = []
    if mode in {"api", "hybrid"} and budget > 0:
        if api_key is None:
            raise SystemExit("--mode api/hybrid requires --api-key-file or DASHSCOPE_API_KEY")
        verified: list[SearchCandidate] = []
        occupied = list(existing)
        api_limit = max(1, max_api_candidates_per_sample)
        ranking_meta: dict[str, Any] = {"mode": api_ranking}
        if api_ranking == "embedding":
            if semantic_api_key is None:
                raise SystemExit("--api-ranking embedding requires --semantic-api-key-file or an API key.")
            try:
                ranked_candidates, ranking_meta = call_embedding_rank(
                    sample_id=sample_id,
                    row=row,
                    candidates=candidates,
                    model=embedding_model,
                    api_key=semantic_api_key,
                    cache_dir=cache_dir,
                    timeout=timeout,
                )
            except Exception as exc:
                ranking_meta = {"mode": "embedding", "error": repr(exc), "fallback": "score"}
                ranked_candidates = rank_api_candidates(candidates, "score")
        elif api_ranking == "rerank":
            if semantic_api_key is None:
                raise SystemExit("--api-ranking rerank requires --semantic-api-key-file or an API key.")
            try:
                ranked_candidates, ranking_meta = call_rerank_rank(
                    sample_id=sample_id,
                    row=row,
                    candidates=candidates,
                    model=rerank_model,
                    api_key=semantic_api_key,
                    cache_dir=cache_dir,
                    timeout=timeout,
                )
            except Exception as exc:
                ranking_meta = {"mode": "rerank", "error": repr(exc), "fallback": "score"}
                ranked_candidates = rank_api_candidates(candidates, "score")
        else:
            ranked_candidates = rank_api_candidates(candidates, api_ranking)
        for cand in ranked_candidates[:api_limit]:
            if any(iou(cand.box, box) >= duplicate_iou for box in occupied):
                continue
            boxes, payload = verify_api_candidate(
                image=image,
                cand=cand,
                sample_id=sample_id,
                model=model,
                api_key=api_key,
                cache_dir=cache_dir,
                max_tokens=max_tokens,
                timeout=timeout,
                min_confidence=min_confidence,
                enable_thinking=enable_thinking,
            )
            api_calls += 0 if payload.get("cache_hit") else 1
            api_attempts.append({"candidate": cand.__dict__, "accepted_boxes": boxes, "cache_hit": payload.get("cache_hit", False)})
            for box in boxes:
                if any(iou(box, prev) >= duplicate_iou for prev in occupied):
                    continue
                verified.append(
                    SearchCandidate(
                        label=f"api_verified:{cand.label}",
                        box=box,
                        target_type=cand.target_type,
                        source="api_crop_verifier",
                        score=cand.score + 3.0,
                        evidence=f"Qwen crop verifier accepted: {cand.evidence}",
                        ocr_text=cand.ocr_text,
                        meta={"base_candidate": cand.__dict__},
                    )
                )
                occupied.append(box)
                if len(verified) >= budget:
                    break
            if len(verified) >= budget:
                break
        if mode == "api":
            selected = verified
        else:
            selected = dedupe_select(
                verified + local_selected,
                existing,
                width=width,
                height=height,
                min_score=0.0,
                max_area_ratio=max_area_ratio,
                duplicate_iou=duplicate_iou,
                max_extra_boxes=budget,
            )

    if selected:
        setup_debug_import(debug_root)
        from postprocess import parse_cct_report  # type: ignore

        new_report = insert_extra_anomalies(report_text(row), selected)
        out["raw_output"] = new_report
        out["parsed"] = parse_cct_report(new_report)

    stage_outputs = dict(out.get("stage_outputs") or {})
    meta = {
        "applied": bool(selected),
        "mode": mode,
        "existing_boxes": len(existing),
        "evidence_coverage": ev_cov,
        "candidate_count": len(candidates),
        "boxes_added": len(selected),
        "api_calls_estimate": api_calls,
        "selected": [c.__dict__ for c in selected],
        "api_attempts": api_attempts[:10],
        "ranking": ranking_meta if mode in {"api", "hybrid"} else {"mode": api_ranking},
        "enable_thinking": enable_thinking,
        "policy": "GT-blind multi-grid/multi-scale OCR row/window search. Appends boxes to FORGED reports; optional crop verifier sees only crop and local OCR context.",
    }
    stage_outputs["qwen_pipe_multiscale_ocr_search"] = meta
    out["stage_outputs"] = stage_outputs
    return out, meta


def load_api_key(path: str) -> str:
    key_path = Path(path).expanduser()
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    import os

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        return api_key
    raise SystemExit("No API key found. Provide --api-key-file or set DASHSCOPE_API_KEY.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL), help="Used only for optional diagnostic subset selection.")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--select-low-loc-limit", type=int, default=0)
    parser.add_argument("--low-loc-threshold", type=float, default=0.03)
    parser.add_argument("--write-subset-only", action="store_true")
    parser.add_argument("--mode", choices=["local", "api", "hybrid"], default="local")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--api-key-file", default=DEFAULT_VERIFIER_KEY, help="Verifier/VLM API key. Keep this as the stable base key.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel"], default="normalized-1000")
    parser.add_argument("--row-radii", default="0,1,2,4", help="Comma-separated OCR row pyramid radii.")
    parser.add_argument("--min-score", type=float, default=5.0)
    parser.add_argument("--max-area-ratio", type=float, default=0.12)
    parser.add_argument("--max-window-area-ratio", type=float, default=0.18)
    parser.add_argument("--max-total-boxes", type=int, default=12)
    parser.add_argument("--max-existing-boxes-to-process", type=int, default=6)
    parser.add_argument("--max-existing-evidence-coverage", type=float, default=1.0)
    parser.add_argument("--max-extra-boxes", type=int, default=6)
    parser.add_argument("--duplicate-iou", type=float, default=0.16)
    parser.add_argument("--max-api-candidates-per-sample", type=int, default=8)
    parser.add_argument("--api-ranking", choices=["score", "stage-first", "embedding", "rerank"], default="score")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable model reasoning mode for crop verifier calls.")
    parser.add_argument("--semantic-api-key-file", default=DEFAULT_SEMANTIC_KEY, help="Embedding/rerank API key. Defaults to the stable base key; override only for isolated rerank experiments.")
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    parser.add_argument("--rerank-model", default="qwen3-rerank")
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
    row_radii = tuple(int(v) for v in re.findall(r"\d+", args.row_radii)) or (0, 1, 2, 4)

    selected: set[str] = set(args.sample_id or [])
    if args.select_low_loc_limit > 0:
        selected.update(
            load_low_loc_selection(
                resolve_pipe_path(args.eval_json),
                threshold=args.low_loc_threshold,
                limit=args.select_low_loc_limit,
            )
        )
    api_key = load_api_key(args.api_key_file) if args.mode in {"api", "hybrid"} else None
    semantic_api_key = None
    if args.mode in {"api", "hybrid"} and args.api_ranking in {"embedding", "rerank"}:
        semantic_api_key = load_api_key(args.semantic_api_key_file)

    stats: dict[str, Any] = {
        "rows": 0,
        "rows_written": 0,
        "rows_processed": 0,
        "rows_changed": 0,
        "boxes_added": 0,
        "api_calls_estimate": 0,
        "selected_count": len(selected),
        "write_subset_only": args.write_subset_only,
        "reasons": {},
        "added_by_source": {},
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
                    cache_dir=cache_dir,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    min_confidence=args.min_confidence,
                    coord_mode=args.coord_mode,
                    row_radii=row_radii,
                    min_score=args.min_score,
                    max_area_ratio=args.max_area_ratio,
                    max_window_area_ratio=args.max_window_area_ratio,
                    max_total_boxes=args.max_total_boxes,
                    max_existing_boxes_to_process=args.max_existing_boxes_to_process,
                    max_existing_evidence_coverage=args.max_existing_evidence_coverage,
                    max_extra_boxes=args.max_extra_boxes,
                    duplicate_iou=args.duplicate_iou,
                    max_api_candidates_per_sample=args.max_api_candidates_per_sample,
                    api_ranking=args.api_ranking,
                    enable_thinking=args.enable_thinking,
                    semantic_api_key=semantic_api_key,
                    embedding_model=args.embedding_model,
                    rerank_model=args.rerank_model,
                )
                stats["rows_processed"] += 1
                reason = str(meta.get("reason") or ("applied" if meta.get("applied") else "unchanged"))
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                if meta.get("applied"):
                    stats["rows_changed"] += 1
                    stats["boxes_added"] += int(meta.get("boxes_added") or 0)
                    stats["api_calls_estimate"] += int(meta.get("api_calls_estimate") or 0)
                    for cand in meta.get("selected") or []:
                        source = str(cand.get("source") or "unknown")
                        stats["added_by_source"][source] = stats["added_by_source"].get(source, 0) + 1
            else:
                out = row
            if (not args.write_subset_only) or should_process:
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
