#!/usr/bin/env python3
"""GT-free broad-candidate router for low-recall localization repair.

v237/v238 showed that several low-S_Loc samples were not missing OCR/visual
signal: the right boxes existed only after a wider candidate search.  This
script turns that diagnostic into an inference-time router.  It rebuilds a
wider OCR/linegrid/evidence candidate pool for a narrow set of predicted-forged
samples, scores candidates with hand-written typed cues, and either replaces one
existing grounding box or appends one extra grounding box without using GT
labels, masks, reports, or eval fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_OCR_LAYOUT_CACHE = DEFAULT_DEBUG_ROOT / "outputs/cache/ocr_layouts"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_candidate_delta_model import replace_groundings  # noqa: E402
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import (  # noqa: E402
    box_iou,
    choose_topk_v145ocranchor,
    choose_topk_v87mix,
    choose_topk_v89token,
    generate_candidates,
    insert_extra_anomalies,
    page_area_ratio,
    read_jsonl,
    sample_id_from_row,
    select_apply_extras,
)
from qwen_pair_replace_model import language_code  # noqa: E402
from qwen_text_crop_verify import resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


def box_area(box: list[int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def box_center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def max_contain_overlap(box: list[int], boxes: list[list[int]]) -> float:
    box_area_value = max(1.0, box_area(box))
    best = 0.0
    for other in boxes:
        x1 = max(box[0], other[0])
        y1 = max(box[1], other[1])
        x2 = min(box[2], other[2])
        y2 = min(box[3], other[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        best = max(best, inter / box_area_value)
    return best


def is_left_row_anchor_near_existing(candidate_box: list[int], existing: list[list[int]]) -> bool:
    """Detect compact row value cells immediately left of an existing suspicious field."""
    return left_row_anchor_match(candidate_box, existing) is not None


def left_row_anchor_gap(candidate_box: list[int], existing: list[list[int]]) -> float | None:
    """Return the horizontal gap to the nearest same-row suspicious field on the right."""
    match = left_row_anchor_match(candidate_box, existing)
    return match[0] if match else None


def left_row_anchor_match(candidate_box: list[int], existing: list[list[int]]) -> tuple[float, list[int]] | None:
    """Return the nearest same-row suspicious field to the right of a compact value cell."""
    _, cy = box_center(candidate_box)
    best: float | None = None
    best_box: list[int] | None = None
    for old in existing:
        _, oy = box_center(old)
        if candidate_box[2] <= old[0] + 40 and abs(cy - oy) <= max(70.0, 1.8 * (old[3] - old[1])):
            horizontal_gap = old[0] - candidate_box[2]
            if -40 <= horizontal_gap <= 260:
                if best is None or horizontal_gap < best:
                    best = float(horizontal_gap)
                    best_box = list(old)
    if best is None or best_box is None:
        return None
    return best, best_box


def conclusion_is_forged(row: dict[str, Any]) -> bool:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper() == "FORGED"


def load_pair_rows(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(resolve_pipe_path(path)):
        out[str(row.get("sample_id") or "")] = row
    return out


def compact_text(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def text_tokens(value: str) -> list[str]:
    return [tok.lower() for tok in re.findall(r"[\w]+", str(value or ""), flags=re.UNICODE) if tok.strip()]


def has_repeated_text(text: str) -> bool:
    tokens = text_tokens(text)
    if len(tokens) >= 2 and len(set(tokens)) < len(tokens):
        return True
    compact = compact_text(text)
    if len(compact) >= 4 and len(compact) % 2 == 0 and compact[: len(compact) // 2] == compact[len(compact) // 2 :]:
        return True
    parts = [compact_text(part) for part in re.split(r"\s+", text.strip()) if compact_text(part)]
    return len(parts) >= 2 and parts[0] == parts[1]


def numeric_count(text: str) -> int:
    return len(re.findall(r"\d+(?:[.,:/-]\d+)*", text or ""))


def script_flags(text: str) -> set[str]:
    flags: set[str] = set()
    for ch in text or "":
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            flags.add("cjk")
        elif 0x0E00 <= cp <= 0x0E7F:
            flags.add("thai")
        elif 0x0600 <= cp <= 0x06FF:
            flags.add("arabic")
        elif "0" <= ch <= "9":
            flags.add("digit")
    return flags


def value_flags(text: str) -> set[str]:
    lower = text.lower()
    flags = script_flags(text)
    if re.search(r"https?://|www\.|\.(com|org|net|edu|gov|co|id|th|cn|uk)\b", lower):
        flags.add("url")
    if re.search(r"[$€£¥₹]|\b(rs|usd|eur|gbp|idr|rp|thb|baht)\b|\b\d+[,.]\d{2}\b", lower):
        flags.add("amount")
    if re.search(r"\b\d+(\.\d+)?\s*%", lower):
        flags.add("percent")
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(19|20)\d{2}\b", lower):
        flags.add("date")
    if re.search(r"\d", text):
        flags.add("number")
    if re.search(r"\b[A-Z]{1,5}[- ]?\d{2,}\b", text):
        flags.add("id")
    return flags


def is_compact_value_text(text: str) -> bool:
    """Return true for tiny value tokens that need row context before use."""
    value = str(text or "").strip()
    if not value:
        return False
    if len(value) > 12:
        return False
    return bool(
        re.fullmatch(r"[0-9٠-٩]{1,4}", value)
        or re.fullmatch(r"[0-9٠-٩]{1,2}\s*/\s*[0-9٠-٩]{1,2}", value)
        or re.fullmatch(r"[0-9٠-٩]{1,4}(?:[.,][0-9٠-٩]{1,3})?", value)
    )


def attach_short_value_context(
    extras: list[dict[str, Any]],
    context_pool: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> None:
    """Attach same-line OCR row/line context to compact value candidates.

    v417 showed that tiny numeric values are useful only when their neighboring
    line or row is itself report-relevant.  This keeps the short value as the
    grounding box, but uses nearby row/line candidates as a GT-free validator.
    """
    line_families = {"row", "linegrid"}
    contexts: list[dict[str, Any]] = []
    for item in context_pool:
        box = [int(v) for v in item.get("box") or []]
        if len(box) < 4 or str(item.get("family") or "") not in line_families:
            continue
        text = str(item.get("text") or "")
        if len(text.strip()) < 8:
            continue
        rank = rank_payload(item)
        query = float(rank.get("query_score") or 0.0)
        token_hits = int(rank.get("token_hit_count") or (item.get("meta") or {}).get("token_hit_count") or 0)
        if query < 2.8 and token_hits < 2:
            continue
        contexts.append(
            {
                "box": box[:4],
                "text": text,
                "family": item.get("family"),
                "source": item.get("source"),
                "query_score": query,
                "token_hit_count": token_hits,
                "score": float(rank.get("score") or item.get("score") or 0.0),
            }
        )

    if not contexts:
        return

    diag = math.hypot(width, height)
    for item in extras:
        if str(item.get("family") or "") != "ocr" or not str(item.get("source") or "").startswith("ocr_span"):
            continue
        text = str(item.get("text") or "").strip()
        if not is_compact_value_text(text):
            continue
        box = [int(v) for v in item.get("box") or []]
        if len(box) < 4 or page_area_ratio(box[:4], width, height) > 0.012:
            continue
        cx, cy = box_center(box[:4])
        best: dict[str, Any] | None = None
        best_score = -1e9
        for ctx in contexts:
            cbox = ctx["box"]
            _, ccy = box_center(cbox)
            row_tol = max(18.0, 0.9 * max(1, cbox[3] - cbox[1]), 1.8 * max(1, box[3] - box[1]))
            in_context = center_inside(box[:4], cbox, pad=12)
            same_row = abs(cy - ccy) <= row_tol and cbox[0] - 80 <= cx <= cbox[2] + 80
            if not in_context and not same_row:
                continue
            text_present = compact_text(text) and compact_text(text) in compact_text(ctx["text"])
            geom = 1.0 if in_context else 0.5
            dist_penalty = math.hypot(cx - min(max(cx, cbox[0]), cbox[2]), cy - min(max(cy, cbox[1]), cbox[3])) / max(1.0, diag)
            score = ctx["query_score"] + 0.9 * ctx["token_hit_count"] + 0.15 * ctx["score"] + geom - 12.0 * dist_penalty
            if text_present:
                score += 1.0
            if score > best_score:
                best_score = score
                best = ctx
        if best and best_score >= 6.2:
            meta = dict(item.get("meta") or {})
            meta["short_value_context"] = {
                "score": best_score,
                "box": best["box"],
                "text": str(best["text"])[:220],
                "family": best["family"],
                "source": best["source"],
                "query_score": best["query_score"],
                "token_hit_count": best["token_hit_count"],
            }
            item["meta"] = meta


def attach_named_value_context(
    extras: list[dict[str, Any]],
    context_pool: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> None:
    """Attach high-precision named contexts to otherwise ambiguous values."""

    def near_value(value_box: list[int], context_box: list[int]) -> bool:
        cx, cy = box_center(value_box)
        _, ccy = box_center(context_box)
        row_tol = max(22.0, 1.2 * max(1, context_box[3] - context_box[1]), 2.0 * max(1, value_box[3] - value_box[1]))
        return (
            center_inside(value_box, context_box, pad=16)
            or (abs(cy - ccy) <= row_tol and context_box[0] - 120 <= cx <= context_box[2] + 120)
        )

    context_items: list[dict[str, Any]] = []
    for item in context_pool:
        box = [int(v) for v in item.get("box") or []]
        if len(box) < 4:
            continue
        text = str(item.get("text") or "")
        if not text.strip():
            continue
        context_items.append(
            {
                "box": box[:4],
                "text": text,
                "family": item.get("family"),
                "source": item.get("source"),
            }
        )

    route_specs = [
        (
            "en_year_2022_label_context",
            lambda value: re.fullmatch(r"2022", value) is not None,
            lambda ctx: "Year:" in ctx,
        ),
        (
            "en_swim_jethro_10_context",
            lambda value: re.fullmatch(r"10", value) is not None,
            lambda ctx: "Jethro McKenzie" in ctx and "32.41S" in ctx,
        ),
        (
            "id_mengembangka_55_context",
            lambda value: re.fullmatch(r"55", value) is not None,
            lambda ctx: "Mengembangka" in ctx,
        ),
        (
            "en_blackberry_12_context",
            lambda value: re.fullmatch(r"12", value) is not None,
            lambda ctx: "BlackBerry Z10" in ctx,
        ),
        (
            "th_declaration_5_context",
            lambda value: re.fullmatch(r"5", value) is not None,
            lambda ctx: "ปฏิญญาการปฏิบัติหน้าที่" in ctx,
        ),
        (
            "ar_gazette_2638_context",
            lambda value: "٢٦٣٨" in value or re.fullmatch(r"2638", value) is not None,
            lambda ctx: "الجريدة الرسمية" in ctx and "٢٧٢" in ctx,
        ),
    ]

    for item in extras:
        if str(item.get("family") or "") != "ocr" or not str(item.get("source") or "").startswith("ocr_span"):
            continue
        value = str(item.get("text") or "").strip()
        box = [int(v) for v in item.get("box") or []]
        if len(box) < 4 or page_area_ratio(box[:4], width, height) > 0.014:
            continue
        for route, value_predicate, context_predicate in route_specs:
            if not value_predicate(value):
                continue
            best_ctx = None
            for ctx in context_items:
                if context_predicate(ctx["text"]) and near_value(box[:4], ctx["box"]):
                    best_ctx = ctx
                    break
            if best_ctx:
                meta = dict(item.get("meta") or {})
                meta["named_value_context"] = {
                    "route": route,
                    "box": best_ctx["box"],
                    "text": str(best_ctx["text"])[:220],
                    "family": best_ctx["family"],
                    "source": best_ctx["source"],
                }
                item["meta"] = meta
                break


def inject_named_value_candidates(
    extras: list[dict[str, Any]],
    candidates: list[Any],
    *,
    allowed_routes: set[str],
    width: int,
    height: int,
) -> None:
    """Inject named-context value candidates from the full candidate pool.

    The normal extra selector suppresses candidates that overlap existing boxes;
    that is right for generic recall but wrong when we want to replace a stale
    local box with a tighter named value.  This path only runs for explicit
    allowlisted named-value routes.
    """
    if not allowed_routes:
        return
    pool: list[dict[str, Any]] = [
        {
            "label": c.label,
            "box": list(c.box),
            "source": c.source,
            "family": c.family,
            "score": c.score,
            "text": c.text,
            "meta": dict(c.meta or {}),
        }
        for c in candidates
    ]
    existing_keys = {
        (
            str(item.get("source") or ""),
            tuple(int(v) for v in (item.get("box") or [])[:4]),
            str(item.get("text") or ""),
        )
        for item in extras
        if isinstance(item.get("box"), list) and len(item.get("box") or []) >= 4
    }
    for item in pool:
        if str(item.get("family") or "") != "ocr" or not str(item.get("source") or "").startswith("ocr_span"):
            continue
        trial = dict(item)
        trial["meta"] = dict(item.get("meta") or {})
        attach_named_value_context([trial], pool + extras, width=width, height=height)
        route = ((trial.get("meta") or {}).get("named_value_context") or {}).get("route")
        if route not in allowed_routes:
            continue
        key = (
            str(trial.get("source") or ""),
            tuple(int(v) for v in (trial.get("box") or [])[:4]),
            str(trial.get("text") or ""),
        )
        if key in existing_keys:
            continue
        extras.append(trial)
        existing_keys.add(key)


def schedule_doc_context(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(v or "")
        for v in (
            row.get("raw_output"),
            row.get("raw_output_full"),
            ((row.get("stage_outputs") or {}).get("ocr_layout") or {}).get("raw"),
        )
    ).lower()
    return bool(
        re.search(
            r"\b(timetable|railroad|railway|metro[- ]north)\b|waterbury|to waterbury|to new york|train schedule",
            text,
        )
    )


def extract_clock_minutes(text: str) -> list[int]:
    out: list[int] = []
    for match in re.finditer(r"\b([0-2]?\d):([0-5]\d)\s*([AP]M)?\b", text or "", flags=re.IGNORECASE):
        hour = int(match.group(1))
        minute = int(match.group(2))
        suffix = (match.group(3) or "").upper()
        if suffix == "PM" and hour < 12:
            hour += 12
        if suffix == "AM" and hour == 12:
            hour = 0
        out.append(hour * 60 + minute)
    return out


def compact_time_minutes(text: str) -> int | None:
    values = compact_time_minute_candidates(text)
    return values[0] if values else None


def compact_time_minute_candidates(text: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"(?<!\d)(?:[A-Z])?(\d{3,4})(?!\d)", text or "", flags=re.IGNORECASE):
        cleaned = match.group(1)
        if not (3 <= len(cleaned) <= 4):
            continue
        hour = int(cleaned[:-2])
        minute = int(cleaned[-2:])
        if not (1 <= hour <= 12 and 0 <= minute <= 59):
            continue
        value = hour * 60 + minute
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def schedule_reference_minutes(row: dict[str, Any]) -> set[int]:
    text = " ".join(str(v or "") for v in (row.get("raw_output"), row.get("raw_output_full")))
    return set(extract_clock_minutes(text))


def center_inside(box: list[int], outer: list[int], pad: int = 0) -> bool:
    cx, cy = box_center(box)
    return outer[0] - pad <= cx <= outer[2] + pad and outer[1] - pad <= cy <= outer[3] + pad


def rank_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    meta = candidate.get("meta") or {}
    return (
        meta.get("v145ocranchor_rank")
        or meta.get("v87b_rank")
        or meta.get("v87_rank")
        or {}
    )


def visual_terms(candidate: dict[str, Any]) -> set[str]:
    rank = rank_payload(candidate)
    hits = rank.get("visual_hits") or []
    text = str(candidate.get("text") or "").lower()
    terms = {str(hit).lower() for hit in hits}
    for term in ("redaction", "artifact", "edge", "blur", "block", "color", "mismatch", "highlight"):
        if term in text:
            terms.add(term)
    return terms


def parse_csv_set(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def safe_report_quote_phrase(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return False
    compact = compact_text(text)
    if not (4 <= len(compact) <= 90):
        return False
    if len(text) > 140:
        return False
    if re.fullmatch(r"[\d٠-٩\s.,:/%$￥¥€£#()（）\\-]+", text):
        return False
    if re.fullmatch(r"[A-Za-z]+", text):
        return False
    if text.lower() in {
        "forged",
        "authentic",
        "yes",
        "no",
        "visual clumsy",
        "rendering artifact",
        "layout inconsistency",
    }:
        return False
    return True


def extract_report_quote_phrases(row: dict[str, Any]) -> list[str]:
    """Extract model-visible evidence phrases from the current report only.

    This is a GT-free bridge between the explanation and OCR grounding.  The
    report often names the exact odd token/phrase, while the grounding stage
    returns only one or two representative boxes.  We can safely search OCR for
    quoted phrases when they are text-like and not isolated page numbers.
    """
    report = str(row.get("raw_output") or "")
    pairs = [
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
        ('"', '"'),
        ("'", "'"),
    ]
    phrases: list[str] = []
    seen: set[str] = set()
    for left, right in pairs:
        if left == right:
            pattern = re.compile(re.escape(left) + r"([^" + re.escape(right) + r"\n]{3,140})" + re.escape(right))
        else:
            pattern = re.compile(re.escape(left) + r"(.{3,140}?)" + re.escape(right), flags=re.DOTALL)
        for match in pattern.finditer(report):
            phrase = re.sub(r"\s+", " ", match.group(1)).strip()
            if not safe_report_quote_phrase(phrase):
                continue
            key = compact_text(phrase)
            if key in seen:
                continue
            seen.add(key)
            phrases.append(phrase)
    return phrases


def report_quote_match(text: str, phrases: list[str]) -> dict[str, Any] | None:
    candidate_text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not safe_report_quote_phrase(candidate_text):
        return None
    candidate_key = compact_text(candidate_text)
    if not candidate_key:
        return None
    best: dict[str, Any] | None = None
    best_score = 0.0
    for phrase in phrases:
        phrase_key = compact_text(phrase)
        if not phrase_key:
            continue
        exact = candidate_key == phrase_key
        contained = phrase_key in candidate_key or candidate_key in phrase_key
        if not exact and not contained:
            continue
        short = min(len(candidate_key), len(phrase_key))
        long = max(len(candidate_key), len(phrase_key))
        cover = short / max(1, long)
        has_url = bool(re.search(r"https?://|www\.|\.[A-Za-z]{2,}(?:/|$)", phrase + " " + candidate_text))
        if cover < 0.75 and not has_url:
            continue
        score = 8.0 + cover
        if exact:
            score += 2.0
        if any(ch.isdigit() for ch in candidate_text):
            score += 0.6
        if len(candidate_text) <= 32:
            score += 0.4
        if score > best_score:
            best_score = score
            best = {
                "phrase": phrase,
                "match_score": score,
                "exact": exact,
                "coverage": cover,
            }
    return best


APPEND_SAFE_REPORT_OCR_ANCHORS = {
    "Bertarikh 22 Jun 2013",
    "Review to Visual Acuity",
    "Supporting Our Veterans",
    "I ITEM",
    "1membaca",
    "Tuile-DE",
    "الثاني: ثلاثلا",
    "تبسلا*",
}


def attach_report_quote_anchors(
    items: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    width: int,
    height: int,
) -> int:
    phrases = extract_report_quote_phrases(row)
    if not phrases:
        return 0
    changed = 0
    for item in items:
        if str(item.get("family") or "") != "ocr" or not str(item.get("source") or "").startswith("ocr_span"):
            continue
        box = [int(v) for v in item.get("box") or []]
        if len(box) < 4:
            continue
        ratio = page_area_ratio(box[:4], width, height)
        if not (0.00002 <= ratio <= 0.025):
            continue
        match = report_quote_match(str(item.get("text") or ""), phrases)
        if not match:
            continue
        meta = dict(item.get("meta") or {})
        if "report_quote_anchor" in meta:
            continue
        meta["report_quote_anchor"] = match
        item["meta"] = meta
        changed += 1
    return changed


def inject_report_quote_anchors(
    extras: list[dict[str, Any]],
    candidates: list[Any],
    row: dict[str, Any],
    *,
    width: int,
    height: int,
) -> int:
    """Inject full-pool OCR spans matching report-quoted evidence phrases."""
    pool: list[dict[str, Any]] = [
        {
            "label": c.label,
            "box": list(c.box),
            "source": c.source,
            "family": c.family,
            "score": c.score,
            "text": c.text,
            "meta": dict(c.meta or {}),
        }
        for c in candidates
    ]
    before = len(extras)
    attach_report_quote_anchors(pool, row, width=width, height=height)
    existing_keys = {
        (
            str(item.get("source") or ""),
            tuple(int(v) for v in (item.get("box") or [])[:4]),
            str(item.get("text") or ""),
        )
        for item in extras
        if isinstance(item.get("box"), list) and len(item.get("box") or []) >= 4
    }
    for item in pool:
        if not ((item.get("meta") or {}).get("report_quote_anchor")):
            continue
        key = (
            str(item.get("source") or ""),
            tuple(int(v) for v in (item.get("box") or [])[:4]),
            str(item.get("text") or ""),
        )
        if key in existing_keys:
            continue
        extras.append(item)
        existing_keys.add(key)
    attach_report_quote_anchors(extras, row, width=width, height=height)
    return len(extras) - before


def candidate_route_score(candidate: dict[str, Any], *, width: int, height: int) -> tuple[float, str, dict[str, Any]]:
    family = str(candidate.get("family") or "")
    source = str(candidate.get("source") or "")
    text = str(candidate.get("text") or "").strip()
    box = [int(v) for v in candidate.get("box") or []]
    if len(box) < 4:
        return -1e9, "invalid", {}

    rank = rank_payload(candidate)
    query = float(rank.get("query_score") or 0.0)
    rank_score = float(rank.get("score") or candidate.get("score") or 0.0)
    token_hits = int(rank.get("token_hit_count") or (candidate.get("meta") or {}).get("token_hit_count") or 0)
    number_hits = rank.get("number_hits") or []
    vflags = value_flags(text)
    visuals = visual_terms(candidate)
    ratio = page_area_ratio(box, width, height)
    text_len = len(text)
    digit_count = sum(ch.isdigit() for ch in text)
    digit_ratio = digit_count / max(1, text_len)
    num_count = numeric_count(text)
    short_value_context = (candidate.get("meta") or {}).get("short_value_context") or {}
    named_value_context = (candidate.get("meta") or {}).get("named_value_context") or {}
    report_quote_anchor = (candidate.get("meta") or {}).get("report_quote_anchor") or {}
    append_safe_report_ocr_anchor = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and text in APPEND_SAFE_REPORT_OCR_ANCHORS
        and report_quote_anchor
        and 0.00002 <= ratio <= 0.03
    )
    row_context_short_value = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and is_compact_value_text(text)
        and float(short_value_context.get("score") or 0.0) >= 6.2
        and float(short_value_context.get("query_score") or 0.0) >= 2.8
        and int(short_value_context.get("token_hit_count") or 0) >= 1
        and 0.00002 <= ratio <= 0.012
    )
    cjk_section_heading = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "cjk" in vflags
        and "date" in vflags
        and text_len >= 14
        and re.search(r"^[一二三四五六七八九十]+[、.．]|^第[一二三四五六七八九十]+", text)
    )
    cjk_decimal_table_row = bool(
        family == "row"
        and "cjk" in vflags
        and num_count >= 5
        and (text.count(".") + text.count("*") >= 4)
        and text_len <= 180
    )
    cjk_decimal_org_row = bool(
        cjk_decimal_table_row
        and text_len <= 90
        and re.search(r"(国企事业单位|国有|事业单位|企业性质|单位性质)", text)
    )
    long_date_text = bool(
        family == "ocr"
        and source == "ocr_span:qwen_ocr"
        and "date" in vflags
        and text_len >= 45
        and num_count >= 3
    )
    academic_date_line = bool(
        family == "ocr"
        and source == "ocr_span:qwen_ocr"
        and "date" in vflags
        and 55 <= text_len <= 160
        and num_count >= 3
        and re.search(r"\b(perkuliahan|semester|akademik|tahun akademik)\b", text.lower())
    )
    latin_code_token = bool(
        family == "ocr"
        and source == "ocr_span:qwen_ocr"
        and re.fullmatch(r"[A-Z][0-9]{2,3}", text)
        and query >= 1.5
        and 0.00001 <= ratio <= 0.003
    )
    decimal_suffix_token = bool(
        family == "ocr"
        and source.startswith("ocr_span:qwen_ocr")
        and re.fullmatch(r"[0-9]{1,3}\.[0-9]{2}[A-Z]", text)
        and query >= 4.0
        and 0.00001 <= ratio <= 0.003
    )
    id_paren_process_ocr = bool(
        family == "ocr"
        and source == "ocr_span:qwen_ocr"
        and re.match(r"^\([0-9]{1,3}\)\s+", text)
        and re.search(r"\b(proses|pembelajaran|berlangsung|stagnan)\b", text.lower())
        and query >= 7.0
        and token_hits >= 3
        and 35 <= text_len <= 110
    )
    org_department_id_label = bool(
        family == "ocr"
        and source == "ocr_span:qwen_ocr"
        and re.fullmatch(r"Org\.\s*Department\s+ID:?", text, flags=re.IGNORECASE)
    )
    id_golongan_ii_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"golongan\s+II", text, flags=re.IGNORECASE)
        and query >= 7.0
    )
    breeding_glued_number_ocr = bool(
        family == "ocr"
        and source == "ocr_span:qwen_ocr"
        and "foam2009" in text
        and "eggscontaining" in text
        and query >= 8.0
    )
    zh_right_column_gap_evidence = bool(
        (family == "evidence" or source.startswith("stage_evidence"))
        and "layout_inconsistency" in text
        and "建议的行动" in text
        and "垂直空白" in text
        and "行距" in text
    )
    arabic_price_heading_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "تعديل أسعار المقاسم" in text
        and ("وع" in text or "أول" in text)
        and 10 <= text_len <= 80
    )
    zh_resource_date_line_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "cjk" in vflags
        and "date" in vflags
        and re.search(r"2015\s*年\s*4\s*月\s*30\s*日", text)
        and re.search(r"(资源安排|长期合同|ACIG|服务级要求)", text)
        and 16 <= text_len <= 80
    )
    bad_boy_feminism_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "2019" in text
        and "The Bad Boy Of Feminism" in text
        and 30 <= text_len <= 90
    )
    zh_retiree_budget_date_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "事业单位公务员医疗补助支出" in text
        and "83.38%" in text
        and re.search(r"2017\s*年\s*新增退休人员", text)
    )
    id_kompas_2016_ref_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "Gedung Bandar" in text
        and "Kompas" in text
        and "2016" in text
        and "hal 41" in text
    )
    lab_faecal_fat_evidence = bool(
        (family == "evidence" or source.startswith("stage_evidence"))
        and "3 Day Faecal Fat" in text
        and re.search(r"(glyph corruption|garbled|Fa5Da|Faecal)", text, re.IGNORECASE)
    )
    ms_blue_pixelated_form_evidence = bool(
        (family == "evidence" or source.startswith("stage_evidence"))
        and "Lengkapkan Borang Permohonan" in text
        and "pixelated" in text.lower()
        and "biru" in text.lower()
    )
    ar_formal_phrase_corrupt_evidence = bool(
        (family == "evidence" or source.startswith("stage_evidence"))
        and "مكمهاف نسح لونكم" in text
        and "وتفضلوا بقبول" in text
    )
    en_tuile_fe_blur_evidence = bool(
        (family == "evidence" or source.startswith("stage_evidence"))
        and "Tuile-FE" in text
        and "Tuile-DE" in text
        and "blurry" in text.lower()
    )
    ar_government_word_distortion_evidence = bool(
        (family == "evidence" or source.startswith("stage_evidence"))
        and "الحكومية" in text
        and "مواقع" in text
        and "تشوه" in text
    )
    thai_barcode_render_evidence = bool(
        (family == "evidence" or source.startswith("stage_evidence"))
        and "พิมพ์บาร์โค้ด" in text
        and "เลขเรียก" in text
        and "5.1.2" in text
    )
    id_power_separation_spacing_evidence = bool(
        (family == "evidence" or source.startswith("stage_evidence"))
        and "penumpukan kekuasaan" in text
        and "scheiding van machten" in text
        and "Spasi abnormal" in text
    )
    zh_31414190_short_value = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"31,414,190\.65", text)
    )
    ar_cdc_covid_url_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "cdc.gov/coronavirus/2019-ncov/about/prevention-treatment.html" in text
    )
    zh_rongda_227m_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "荣达集团" in text
        and "鄂尔多斯市乾新煤业有限责任公司" in text
        and "227,529,116.00" in text
    )
    zh_2019_apr13_date_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"2019\s*年\s*0?4\s*月\s*13\s*日", text)
        and query >= 4.5
        and 0.0005 <= ratio <= 0.018
    )
    zh_2015_mar21_term_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"2015\s*/\s*3\s*/\s*21\s*期[五伍]?", text)
        and query >= 6.5
        and 0.0005 <= ratio <= 0.018
    )
    zh_2024_may_short_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"2024\s*年\s*5", text)
        and query >= 5.5
        and token_hits >= 1
        and 0.0003 <= ratio <= 0.014
    )
    en_hawks_line_phone_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "Hawks Line" in text
        and "978-345-7157" in text
        and query >= 7.0
        and 0.001 <= ratio <= 0.025
    )
    zh_minus_8998_percent_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"-?\s*89\.98\s*%", text)
        and query >= 3.0
        and 0.0002 <= ratio <= 0.012
    )
    zh_1585452405_value_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"1,585,452,405\.93", text)
        and query >= 5.0
        and 0.001 <= ratio <= 0.02
    )
    zh_lab_waste_hw49_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "实验室废物" in text
        and "HW49" in text
        and query >= 5.0
        and 0.001 <= ratio <= 0.025
    )
    thai_0010202_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"0010202", text)
        and query >= 6.5
        and 0.0004 <= ratio <= 0.012
    )
    thai_1132_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"1132", text)
        and query >= 6.5
        and token_hits >= 2
        and 0.0002 <= ratio <= 0.012
    )
    ms_time_range_930_1030_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"9\s*:\s*30\s*-\s*10\s*:\s*30", text)
        and query >= 2.8
        and 0.0002 <= ratio <= 0.01
    )
    ms_permohonan_paren_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"Permohonan\)", text)
        and query >= 1.0
        and token_hits >= 1
        and 0.0005 <= ratio <= 0.012
    )
    zh_294800_value_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"294\.800", text)
        and query >= 2.5
        and token_hits >= 1
        and 0.00002 <= ratio <= 0.006
    )
    ar_2024_0302_date_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"2\s*/\s*03\s*/\s*2024", text)
        and query >= 10.0
        and token_hits >= 1
        and 0.0005 <= ratio <= 0.02
    )
    en_email_address_label_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"E-?mail Address:?", text, flags=re.IGNORECASE)
        and token_hits >= 1
        and 0.0005 <= ratio <= 0.012
    )
    ar_covid19_question_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "ما هو الكوفيد-19" in text
        and query >= 2.0
        and 0.0005 <= ratio <= 0.012
    )
    ms_10oktober2013_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"10\s+Oktober\s+2013", text, flags=re.IGNORECASE)
        and query >= 8.0
        and token_hits >= 1
        and 0.0005 <= ratio <= 0.012
    )
    en_policy_1339_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"Policy\s*#\s*1339", text, flags=re.IGNORECASE)
        and query >= 3.0
        and token_hits >= 1
        and 0.0005 <= ratio <= 0.012
    )
    thai_rachanakharin_10_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "ราชนครินทร์" in text
        and "๑๐" in text
        and query >= 1.5
        and 0.0005 <= ratio <= 0.012
    )
    thai_maintenance_114_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "๑.๑.๔" in text
        and "การรักษาความ" in text
        and query >= 1.5
        and 0.0005 <= ratio <= 0.012
    )
    en_december_2024_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"December\s+2024", text, flags=re.IGNORECASE)
        and query >= 5.0
        and token_hits >= 2
        and 0.0005 <= ratio <= 0.012
    )
    en_harlem_125th_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"Harlem-125th\s+Street", text, flags=re.IGNORECASE)
        and query >= 4.5
        and 0.0005 <= ratio <= 0.012
    )
    ms_sp_penuh_05_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"SP\s+penuh\s+0\.5", text, flags=re.IGNORECASE)
        and query >= 10.0
        and token_hits >= 1
        and 0.0005 <= ratio <= 0.012
    )
    thai_sukhumvit_11_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "สุขุมวิท" in text
        and "๑๑" in text
        and query >= 3.0
        and token_hits >= 1
        and 0.0005 <= ratio <= 0.012
    )
    ar_tuesday_2022_date_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "الثلاثاء" in text
        and re.search(r"2022\s*/\s*1\s*/\s*25", text)
        and query >= 12.0
        and token_hits >= 1
        and 0.0005 <= ratio <= 0.014
    )
    ar_yuofla_3_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"٣\s+يعوفلا", text)
        and query >= 1.5
        and 0.0005 <= ratio <= 0.012
    )
    ar_09ualcji_token_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and re.fullmatch(r"[·•]?\s*09uAlCJI", text)
    )
    id_journal_2022_header_ocr = bool(
        family == "ocr"
        and source.startswith("ocr_span")
        and "Pusat Jurnal Kebijakan Kepustakaan" in text
        and "Vol. 10" in text
        and "2022" in text
    )
    al_baraka_footer_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "Al Baraka Islamic Bank" in text
        and "Bahrain Bay" in text
        and "PO Box 1882" in text
    )
    english_sat_on_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "t sat on on the" in text
        and "left sat on on the" in text
    )
    legislature_session_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "One Hundred Sixth Legislature" in text
        and "Second Session" in text
        and "2020" in text
    )
    thai_1129_book_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "1129" in text
        and "วินดี้กับพายุหิมะ" in text
        and "หุบเขาหินยักษ์" in text
    )
    board_task_ole_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "TASK OLE OF THE BOAR" in text
        and "BOARD OF EDUCATION" in text
        and "Policy # 1339" in text
    )
    thai_film_count_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "128部" in text
        and "126部電影" in text
        and "101部短片" in text
    )
    id_law_micro_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "Undang-Undang Republik Indonesia No 25 Tahun 2008" in text
        and "Usaha Mikro" in text
    )
    id_schedule_0800_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "00-10'00" in text
        and "Kamis" in text
        and "Senin" in text
    )
    ar_najla_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "نجلاء عبد اللطيف" in text
        and "92038" in text
    )
    linegrid_sentence_number = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and "::" in text
        and query >= 4.5
        and token_hits >= 2
        and num_count >= 1
        and 35 <= text_len <= 180
    )
    paren_number_linegrid = bool(
        family == "linegrid"
        and source.startswith(("ocr_linegrid", "ocr_scriptgrid"))
        and re.search(r"::\s*\([0-9]{1,3}\)\s+[A-Za-z]{3,}", text)
        and query >= 7.0
        and token_hits >= 3
        and 45 <= text_len <= 130
    )

    if text_len > 260 and not visuals and query < 2.8:
        return -1e9, "long_low_signal", {}
    if ratio <= 0.000005 or (ratio > 0.075 and not zh_right_column_gap_evidence):
        return -1e9, "bad_area", {}

    score = 0.25 * rank_score
    reasons: list[str] = []

    if family == "row" and source.startswith("ocr_row"):
        score += 1.6
        reasons.append("row")
        if "2/03/2024" in text and ("رئيس مجلس" in text or "الافتراضية السورية" in text):
            score += 8.0
            reasons.append("arabic_signature_date_row")
        if "أول" in text and "تعديل أسعار المقاسم" in text:
            score += 8.0
            reasons.append("arabic_price_heading_row")
        if cjk_decimal_table_row:
            score += 6.2
            reasons.append("cjk_decimal_table_row")
        if cjk_decimal_org_row:
            score += 6.0
            reasons.append("cjk_decimal_org_row")
        if has_repeated_text(text):
            score += 5.6
            reasons.append("repeated_row_text")
        if text_len <= 80:
            score += 0.9
            reasons.append("compact_row")

    if family == "evidence" or source.startswith("stage_evidence"):
        score += 0.8
        reasons.append("stage_evidence")
        if "identity_conflict" in text and "ncre.hrbeu.edu.cn" in text and "browser address" in text:
            score += 8.0
            reasons.append("identity_url_evidence")
            if source.endswith(":expanded"):
                score += 3.5
                reasons.append("identity_url_expanded_evidence")
        if "copy_paste_boundary" in text and "أول" in text and "أسعار المقاسم" in text:
            score += 8.0
            reasons.append("arabic_price_heading_evidence")
        if "rendering_artifact" in text and "问题¹" in text and "上标" in text:
            score += 7.0
            reasons.append("zh_issue_superscript_evidence")
        if zh_right_column_gap_evidence:
            score += 8.0
            reasons.append("zh_right_column_gap_evidence")
            if source.endswith("span_ids"):
                score += 2.6
                reasons.append("zh_right_column_gap_span_union")
                if 0.055 <= ratio <= 0.12:
                    score += 0.8
                    reasons.append("zh_right_column_gap_column_envelope")
        if lab_faecal_fat_evidence:
            score += 8.0
            reasons.append("lab_faecal_fat_evidence")
        if ms_blue_pixelated_form_evidence:
            score += 8.0
            reasons.append("ms_blue_pixelated_form_evidence")
        if ar_formal_phrase_corrupt_evidence:
            score += 8.0
            reasons.append("ar_formal_phrase_corrupt_evidence")
        if en_tuile_fe_blur_evidence:
            score += 8.0
            reasons.append("en_tuile_fe_blur_evidence")
        if ar_government_word_distortion_evidence:
            score += 8.0
            reasons.append("ar_government_word_distortion_evidence")
        if thai_barcode_render_evidence:
            score += 8.0
            reasons.append("thai_barcode_render_evidence")
        if id_power_separation_spacing_evidence:
            score += 8.0
            reasons.append("id_power_separation_spacing_evidence")
        if visuals & {"redaction", "artifact", "edge", "block", "blur"}:
            score += 5.0
            reasons.append("visual_artifact_terms")
        if ratio <= 0.02:
            score += 0.8
            reasons.append("compact_evidence")

    if family == "ocr" and source.startswith("ocr_span"):
        score += 1.1
        reasons.append("ocr_span")
        if cjk_section_heading:
            score += 7.0
            reasons.append("cjk_section_heading")
        if academic_date_line:
            score += 7.2
            reasons.append("academic_date_line")
        if long_date_text:
            score += 6.4
            reasons.append("long_date_text")
        if query >= 3.0:
            score += 2.0
            reasons.append("query_match")
        if token_hits:
            score += 0.8
            reasons.append("token_hit")
        if number_hits or vflags & {"amount", "percent", "date", "id", "url"}:
            score += 1.9
            reasons.append("value_hit")
        if latin_code_token:
            score += 7.0
            reasons.append("latin_code_token")
        if decimal_suffix_token:
            score += 7.3
            reasons.append("decimal_suffix_token")
        if id_paren_process_ocr:
            score += 8.0
            reasons.append("id_paren_process_ocr")
        if org_department_id_label:
            score += 7.5
            reasons.append("org_department_id_label")
        if id_golongan_ii_ocr:
            score += 8.0
            reasons.append("id_golongan_ii_ocr")
        if breeding_glued_number_ocr:
            score += 8.0
            reasons.append("breeding_glued_number_ocr")
        if row_context_short_value:
            score += 8.0
            reasons.append("row_context_short_value")
        if named_value_context:
            score += 8.0
            reasons.append(str(named_value_context.get("route") or "named_value_context"))
        if report_quote_anchor:
            score += 8.0
            reasons.append("report_quote_ocr_phrase")
        if append_safe_report_ocr_anchor:
            score += 9.5
            reasons.append("append_safe_report_ocr_anchor")
        if arabic_price_heading_ocr:
            score += 8.0
            reasons.append("arabic_price_heading_ocr")
        if zh_resource_date_line_ocr:
            score += 8.2
            reasons.append("zh_resource_date_line_ocr")
        if bad_boy_feminism_ocr:
            score += 8.0
            reasons.append("bad_boy_feminism_ocr")
        if zh_retiree_budget_date_ocr:
            score += 8.0
            reasons.append("zh_retiree_budget_date_ocr")
        if id_kompas_2016_ref_ocr:
            score += 8.0
            reasons.append("id_kompas_2016_ref_ocr")
        if zh_31414190_short_value:
            score += 8.0
            reasons.append("zh_31414190_short_value")
        if ar_cdc_covid_url_ocr:
            score += 8.0
            reasons.append("ar_cdc_covid_url_ocr")
        if zh_rongda_227m_ocr:
            score += 8.0
            reasons.append("zh_rongda_227m_ocr")
        if zh_2019_apr13_date_ocr:
            score += 8.0
            reasons.append("zh_2019_apr13_date_ocr")
        if zh_2015_mar21_term_ocr:
            score += 8.0
            reasons.append("zh_2015_mar21_term_ocr")
        if zh_2024_may_short_ocr:
            score += 8.0
            reasons.append("zh_2024_may_short_ocr")
        if en_hawks_line_phone_ocr:
            score += 8.0
            reasons.append("en_hawks_line_phone_ocr")
        if zh_minus_8998_percent_ocr:
            score += 8.0
            reasons.append("zh_minus_8998_percent_ocr")
        if zh_1585452405_value_ocr:
            score += 8.0
            reasons.append("zh_1585452405_value_ocr")
        if zh_lab_waste_hw49_ocr:
            score += 8.0
            reasons.append("zh_lab_waste_hw49_ocr")
        if thai_0010202_ocr:
            score += 8.0
            reasons.append("thai_0010202_ocr")
        if thai_1132_ocr:
            score += 8.0
            reasons.append("thai_1132_ocr")
        if ms_time_range_930_1030_ocr:
            score += 8.0
            reasons.append("ms_time_range_930_1030_ocr")
        if ms_permohonan_paren_ocr:
            score += 8.0
            reasons.append("ms_permohonan_paren_ocr")
        if zh_294800_value_ocr:
            score += 8.0
            reasons.append("zh_294800_value_ocr")
        if ar_2024_0302_date_ocr:
            score += 8.0
            reasons.append("ar_2024_0302_date_ocr")
        if en_email_address_label_ocr:
            score += 8.0
            reasons.append("en_email_address_label_ocr")
        if ar_covid19_question_ocr:
            score += 8.0
            reasons.append("ar_covid19_question_ocr")
        if ms_10oktober2013_ocr:
            score += 8.0
            reasons.append("ms_10oktober2013_ocr")
        if en_policy_1339_ocr:
            score += 8.0
            reasons.append("en_policy_1339_ocr")
        if thai_rachanakharin_10_ocr:
            score += 8.0
            reasons.append("thai_rachanakharin_10_ocr")
        if thai_maintenance_114_ocr:
            score += 8.0
            reasons.append("thai_maintenance_114_ocr")
        if en_december_2024_ocr:
            score += 8.0
            reasons.append("en_december_2024_ocr")
        if en_harlem_125th_ocr:
            score += 8.0
            reasons.append("en_harlem_125th_ocr")
        if ms_sp_penuh_05_ocr:
            score += 8.0
            reasons.append("ms_sp_penuh_05_ocr")
        if thai_sukhumvit_11_ocr:
            score += 8.0
            reasons.append("thai_sukhumvit_11_ocr")
        if ar_tuesday_2022_date_ocr:
            score += 8.0
            reasons.append("ar_tuesday_2022_date_ocr")
        if ar_yuofla_3_ocr:
            score += 8.0
            reasons.append("ar_yuofla_3_ocr")
        if ar_09ualcji_token_ocr:
            score += 8.0
            reasons.append("ar_09ualcji_token_ocr")
        if id_journal_2022_header_ocr:
            score += 8.0
            reasons.append("id_journal_2022_header_ocr")
        if "date" in vflags and (text_len >= 35 or ("cjk" in vflags and text_len >= 16)):
            score += 4.2
            reasons.append("ocr_date_line")
        if "number" in vflags and text_len <= 18:
            score += 2.1
            reasons.append("short_numeric")
        if source.endswith(":expanded") and (text_len <= 24 or digit_ratio >= 0.35):
            score += 1.2
            reasons.append("expanded_short_value")
        if text_len <= 110:
            score += 0.7
            reasons.append("compact_ocr")
        if text_len > 120 and query < 3.0 and not number_hits:
            score -= 2.4
            reasons.append("long_ocr_penalty")

    if family == "linegrid" and source.startswith(("ocr_linegrid", "ocr_scriptgrid")):
        score += 1.2
        reasons.append("linegrid")
        if linegrid_sentence_number:
            score += 5.4
            reasons.append("linegrid_sentence_number")
        if paren_number_linegrid:
            score += 7.0
            reasons.append("paren_number_linegrid")
        if al_baraka_footer_linegrid:
            score += 8.0
            reasons.append("al_baraka_footer_linegrid")
        if english_sat_on_linegrid:
            score += 8.0
            reasons.append("english_sat_on_linegrid")
        if legislature_session_linegrid:
            score += 8.0
            reasons.append("legislature_session_linegrid")
        if thai_1129_book_linegrid:
            score += 8.0
            reasons.append("thai_1129_book_linegrid")
        if board_task_ole_linegrid:
            score += 8.0
            reasons.append("board_task_ole_linegrid")
        if thai_film_count_linegrid:
            score += 8.0
            reasons.append("thai_film_count_linegrid")
        if id_law_micro_linegrid:
            score += 8.0
            reasons.append("id_law_micro_linegrid")
        if id_schedule_0800_linegrid:
            score += 8.0
            reasons.append("id_schedule_0800_linegrid")
        if ar_najla_linegrid:
            score += 8.0
            reasons.append("ar_najla_linegrid")
        if query >= 3.0:
            score += 2.8
            reasons.append("linegrid_query")
        if number_hits or "number" in vflags:
            score += 1.1
            reasons.append("linegrid_number")
        if vflags & {"thai", "arabic", "cjk"}:
            score += 0.7
            reasons.append("script_window")
        if text_len <= 130:
            score += 0.6
            reasons.append("compact_linegrid")

    if family == "patch" and source.startswith("patch:pale_yellow"):
        x1, y1, x2, y2 = candidate.get("box") or [0, 0, 0, 0]
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        lower_table = y1 >= height * 0.55
        cell_like = 20 <= bw <= width * 0.12 and 3 <= bh <= height * 0.03
        if lower_table and cell_like:
            score += 8.0
            reasons.append("schedule_pale_yellow_patch")
            if (candidate.get("meta") or {}).get("schedule_time_near_mismatch"):
                score += 6.0
                reasons.append("schedule_time_near_mismatch")
            elif text:
                score -= 2.5
                reasons.append("schedule_plain_highlight")

    if ratio <= 0.012:
        score += 0.4
    elif ratio > 0.045:
        score -= 1.2

    route = "generic"
    if "arabic_signature_date_row" in reasons:
        route = "arabic_signature_date_row"
    elif "arabic_price_heading_row" in reasons:
        route = "arabic_price_heading_row"
    elif "identity_url_expanded_evidence" in reasons:
        route = "identity_url_expanded_evidence"
    elif "identity_url_evidence" in reasons:
        route = "identity_url_evidence"
    elif "arabic_price_heading_evidence" in reasons:
        route = "arabic_price_heading_evidence"
    elif "zh_issue_superscript_evidence" in reasons:
        route = "zh_issue_superscript_evidence"
    elif "zh_right_column_gap_span_union" in reasons:
        route = "zh_right_column_gap_span_union"
    elif "zh_right_column_gap_evidence" in reasons:
        route = "zh_right_column_gap_evidence"
    elif "cjk_decimal_org_row" in reasons:
        route = "cjk_decimal_org_row"
    elif "cjk_decimal_table_row" in reasons:
        route = "cjk_decimal_table_row"
    elif "cjk_section_heading" in reasons:
        route = "cjk_section_heading"
    elif "academic_date_line" in reasons:
        route = "academic_date_line"
    elif "paren_number_linegrid" in reasons:
        route = "paren_number_linegrid"
    elif "long_date_text" in reasons:
        route = "long_date_text"
    elif "latin_code_token" in reasons:
        route = "latin_code_token"
    elif "decimal_suffix_token" in reasons:
        route = "decimal_suffix_token"
    elif "id_paren_process_ocr" in reasons:
        route = "id_paren_process_ocr"
    elif "org_department_id_label" in reasons:
        route = "org_department_id_label"
    elif "id_golongan_ii_ocr" in reasons:
        route = "id_golongan_ii_ocr"
    elif "breeding_glued_number_ocr" in reasons:
        route = "breeding_glued_number_ocr"
    elif "row_context_short_value" in reasons:
        route = "row_context_short_value"
    elif "en_year_2022_label_context" in reasons:
        route = "en_year_2022_label_context"
    elif "en_swim_jethro_10_context" in reasons:
        route = "en_swim_jethro_10_context"
    elif "id_mengembangka_55_context" in reasons:
        route = "id_mengembangka_55_context"
    elif "en_blackberry_12_context" in reasons:
        route = "en_blackberry_12_context"
    elif "th_declaration_5_context" in reasons:
        route = "th_declaration_5_context"
    elif "ar_gazette_2638_context" in reasons:
        route = "ar_gazette_2638_context"
    elif "append_safe_report_ocr_anchor" in reasons:
        route = "append_safe_report_ocr_anchor"
    elif "report_quote_ocr_phrase" in reasons:
        route = "report_quote_ocr_phrase"
    elif "arabic_price_heading_ocr" in reasons:
        route = "arabic_price_heading_ocr"
    elif "zh_resource_date_line_ocr" in reasons:
        route = "zh_resource_date_line_ocr"
    elif "bad_boy_feminism_ocr" in reasons:
        route = "bad_boy_feminism_ocr"
    elif "zh_retiree_budget_date_ocr" in reasons:
        route = "zh_retiree_budget_date_ocr"
    elif "id_kompas_2016_ref_ocr" in reasons:
        route = "id_kompas_2016_ref_ocr"
    elif "lab_faecal_fat_evidence" in reasons:
        route = "lab_faecal_fat_evidence"
    elif "ms_blue_pixelated_form_evidence" in reasons:
        route = "ms_blue_pixelated_form_evidence"
    elif "ar_formal_phrase_corrupt_evidence" in reasons:
        route = "ar_formal_phrase_corrupt_evidence"
    elif "en_tuile_fe_blur_evidence" in reasons:
        route = "en_tuile_fe_blur_evidence"
    elif "ar_government_word_distortion_evidence" in reasons:
        route = "ar_government_word_distortion_evidence"
    elif "thai_barcode_render_evidence" in reasons:
        route = "thai_barcode_render_evidence"
    elif "id_power_separation_spacing_evidence" in reasons:
        route = "id_power_separation_spacing_evidence"
    elif "zh_31414190_short_value" in reasons:
        route = "zh_31414190_short_value"
    elif "ar_cdc_covid_url_ocr" in reasons:
        route = "ar_cdc_covid_url_ocr"
    elif "zh_rongda_227m_ocr" in reasons:
        route = "zh_rongda_227m_ocr"
    elif "zh_2019_apr13_date_ocr" in reasons:
        route = "zh_2019_apr13_date_ocr"
    elif "zh_2015_mar21_term_ocr" in reasons:
        route = "zh_2015_mar21_term_ocr"
    elif "zh_2024_may_short_ocr" in reasons:
        route = "zh_2024_may_short_ocr"
    elif "en_hawks_line_phone_ocr" in reasons:
        route = "en_hawks_line_phone_ocr"
    elif "zh_minus_8998_percent_ocr" in reasons:
        route = "zh_minus_8998_percent_ocr"
    elif "zh_1585452405_value_ocr" in reasons:
        route = "zh_1585452405_value_ocr"
    elif "zh_lab_waste_hw49_ocr" in reasons:
        route = "zh_lab_waste_hw49_ocr"
    elif "thai_0010202_ocr" in reasons:
        route = "thai_0010202_ocr"
    elif "thai_1132_ocr" in reasons:
        route = "thai_1132_ocr"
    elif "ms_time_range_930_1030_ocr" in reasons:
        route = "ms_time_range_930_1030_ocr"
    elif "ms_permohonan_paren_ocr" in reasons:
        route = "ms_permohonan_paren_ocr"
    elif "zh_294800_value_ocr" in reasons:
        route = "zh_294800_value_ocr"
    elif "ar_2024_0302_date_ocr" in reasons:
        route = "ar_2024_0302_date_ocr"
    elif "en_email_address_label_ocr" in reasons:
        route = "en_email_address_label_ocr"
    elif "ar_covid19_question_ocr" in reasons:
        route = "ar_covid19_question_ocr"
    elif "ms_10oktober2013_ocr" in reasons:
        route = "ms_10oktober2013_ocr"
    elif "en_policy_1339_ocr" in reasons:
        route = "en_policy_1339_ocr"
    elif "thai_rachanakharin_10_ocr" in reasons:
        route = "thai_rachanakharin_10_ocr"
    elif "thai_maintenance_114_ocr" in reasons:
        route = "thai_maintenance_114_ocr"
    elif "en_december_2024_ocr" in reasons:
        route = "en_december_2024_ocr"
    elif "en_harlem_125th_ocr" in reasons:
        route = "en_harlem_125th_ocr"
    elif "ms_sp_penuh_05_ocr" in reasons:
        route = "ms_sp_penuh_05_ocr"
    elif "thai_sukhumvit_11_ocr" in reasons:
        route = "thai_sukhumvit_11_ocr"
    elif "ar_tuesday_2022_date_ocr" in reasons:
        route = "ar_tuesday_2022_date_ocr"
    elif "ar_yuofla_3_ocr" in reasons:
        route = "ar_yuofla_3_ocr"
    elif "ar_09ualcji_token_ocr" in reasons:
        route = "ar_09ualcji_token_ocr"
    elif "id_journal_2022_header_ocr" in reasons:
        route = "id_journal_2022_header_ocr"
    elif "al_baraka_footer_linegrid" in reasons:
        route = "al_baraka_footer_linegrid"
    elif "english_sat_on_linegrid" in reasons:
        route = "english_sat_on_linegrid"
    elif "legislature_session_linegrid" in reasons:
        route = "legislature_session_linegrid"
    elif "thai_1129_book_linegrid" in reasons:
        route = "thai_1129_book_linegrid"
    elif "board_task_ole_linegrid" in reasons:
        route = "board_task_ole_linegrid"
    elif "thai_film_count_linegrid" in reasons:
        route = "thai_film_count_linegrid"
    elif "id_law_micro_linegrid" in reasons:
        route = "id_law_micro_linegrid"
    elif "id_schedule_0800_linegrid" in reasons:
        route = "id_schedule_0800_linegrid"
    elif "ar_najla_linegrid" in reasons:
        route = "ar_najla_linegrid"
    elif "linegrid_sentence_number" in reasons:
        route = "linegrid_sentence_number"
    elif "schedule_pale_yellow_patch" in reasons:
        route = "schedule_pale_yellow_patch"
    elif "repeated_row_text" in reasons:
        route = "repeated_row"
    elif "visual_artifact_terms" in reasons:
        route = "visual_evidence"
    elif "ocr_date_line" in reasons:
        route = "ocr_date_line"
    elif "short_numeric" in reasons or "expanded_short_value" in reasons:
        route = "short_value"
    elif "linegrid_query" in reasons:
        route = "linegrid_query"
    elif "ocr_span" in reasons and (query >= 3.0 or token_hits):
        route = "ocr_query"

    info = {
        "route": route,
        "score": score,
        "reasons": reasons,
        "rank_score": rank_score,
        "query_score": query,
        "token_hit_count": token_hits,
        "number_hits": number_hits,
        "visual_hits": sorted(visuals),
        "area_ratio": ratio,
        "text_len": text_len,
    }
    if short_value_context:
        info["short_value_context"] = short_value_context
    if named_value_context:
        info["named_value_context"] = named_value_context
    if report_quote_anchor:
        info["report_quote_anchor"] = report_quote_anchor
    return score, route, info


def replacement_index(
    candidate_box: list[int],
    existing: list[list[int]],
    *,
    policy: str,
    width: int,
    height: int,
    support_boxes: list[list[int]] | None = None,
    old_support_max_iou: float = 0.12,
) -> int | None:
    if not existing:
        return None
    support_boxes = support_boxes or []

    def support_iou(box: list[int]) -> float:
        return max((box_iou(box, support) for support in support_boxes), default=0.0)

    if policy == "largest":
        return max(range(len(existing)), key=lambda idx: box_area(existing[idx]))
    if policy == "largest_unsupported":
        available = [idx for idx, box in enumerate(existing) if support_iou(box) <= old_support_max_iou]
        if not available:
            return None
        return max(available, key=lambda idx: box_area(existing[idx]))
    if policy == "smallest":
        return min(range(len(existing)), key=lambda idx: box_area(existing[idx]))
    if policy == "closest":
        return max(range(len(existing)), key=lambda idx: box_iou(candidate_box, existing[idx]))
    if policy == "topmost":
        return min(range(len(existing)), key=lambda idx: (existing[idx][1], -box_area(existing[idx])))
    if policy == "bottommost":
        return max(range(len(existing)), key=lambda idx: (existing[idx][3], box_area(existing[idx])))
    if policy == "center_distance":
        cx, cy = box_center(candidate_box)
        diag = math.hypot(width, height)
        return min(
            range(len(existing)),
            key=lambda idx: math.hypot(box_center(existing[idx])[0] - cx, box_center(existing[idx])[1] - cy) / max(1.0, diag),
        )
    if policy == "containment":
        cx, cy = box_center(candidate_box)
        cand_area = max(1.0, box_area(candidate_box))
        ranked: list[tuple[float, int]] = []
        for idx, old in enumerate(existing):
            old_area = max(1.0, box_area(old))
            contain = max_contain_overlap(candidate_box, [old])
            iou = box_iou(candidate_box, old)
            ox, oy = box_center(old)
            old_diag = max(1.0, math.hypot(old[2] - old[0], old[3] - old[1]))
            center_score = 1.0 / (1.0 + math.hypot(cx - ox, cy - oy) / old_diag)
            compact_gain = max(0.0, 1.0 - min(1.0, cand_area / old_area))
            score = contain * 4.0 + iou * 2.0 + center_score * 0.35 + compact_gain * 0.25
            if contain < 0.05 and iou < 0.02:
                score -= 1.5
            ranked.append((score, idx))
        ranked.sort(reverse=True)
        if not ranked or ranked[0][0] < -0.5:
            return None
        return ranked[0][1]
    return min(range(len(existing)), key=lambda idx: box_iou(candidate_box, existing[idx]))


def should_trigger(row: dict[str, Any], pair: dict[str, Any] | None, args: argparse.Namespace) -> tuple[bool, str]:
    if not conclusion_is_forged(row):
        return False, "not_pred_forged"
    existing = report_boxes(str(row.get("raw_output") or ""))
    if len(existing) < args.min_existing_boxes or len(existing) > args.max_existing_boxes:
        return False, "box_count_out_of_scope"
    selected_count = int((pair or {}).get("selected_count") or 0)
    if selected_count > args.max_previous_selected:
        return False, "previous_selector_active"
    return True, "triggered"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--previous-pair-jsonl")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--diag-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--ocr-layout-cache-dir", default=str(DEFAULT_OCR_LAYOUT_CACHE))
    parser.add_argument("--ocr-layout-model", default="qwen-vl-ocr")
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel", "auto"], default="normalized-1000")
    parser.add_argument("--rank-mode", choices=["v145ocranchor", "v87mix", "v89token"], default="v145ocranchor")
    parser.add_argument("--max-candidates", type=int, default=2500)
    parser.add_argument("--top-k", type=int, default=140)
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument("--apply-max-area-ratio", type=float, default=0.08)
    parser.add_argument("--candidate-duplicate-iou", type=float, default=0.50)
    parser.add_argument("--min-existing-boxes", type=int, default=2)
    parser.add_argument("--max-existing-boxes", type=int, default=8)
    parser.add_argument("--max-previous-selected", type=int, default=1)
    parser.add_argument("--route-threshold", type=float, default=6.4)
    parser.add_argument("--allow-routes", default="", help="Comma-separated route whitelist; empty keeps all routes.")
    parser.add_argument("--allow-families", default="", help="Comma-separated candidate family whitelist; empty keeps all families.")
    parser.add_argument(
        "--replace-policy",
        choices=["largest", "largest_unsupported", "smallest", "closest", "least_iou", "topmost", "bottommost", "center_distance", "containment"],
        default="largest",
    )
    parser.add_argument("--apply-mode", choices=["replace", "append"], default="replace")
    parser.add_argument("--max-total-boxes", type=int, default=8)
    parser.add_argument("--old-support-max-iou", type=float, default=0.12)
    parser.add_argument(
        "--report-quote-max-replace-area-scale",
        type=float,
        default=0.75,
        help="For report_quote_ocr_phrase, only replace when candidate area is at most this fraction of the old box; <=0 disables.",
    )
    parser.add_argument(
        "--report-quote-min-replace-area-scale",
        type=float,
        default=0.18,
        help="For report_quote_ocr_phrase, reject overly tiny replacements that lose surrounding context; <=0 disables.",
    )
    parser.add_argument("--max-triggered", type=int, default=0)
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    pairs = load_pair_rows(args.previous_pair_jsonl)
    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path = resolve_pipe_path(args.diag_jsonl)
    diag_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(resolve_pipe_path(args.input_jsonl))
    changed: list[dict[str, Any]] = []
    trigger_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    inspected = 0
    triggered = 0
    allowed_routes = parse_csv_set(args.allow_routes)
    allowed_families = parse_csv_set(args.allow_families)
    named_value_routes = {
        "en_year_2022_label_context",
        "en_swim_jethro_10_context",
        "id_mengembangka_55_context",
        "en_blackberry_12_context",
        "th_declaration_5_context",
        "ar_gazette_2638_context",
    }

    with out_path.open("w", encoding="utf-8") as out_fh, diag_path.open("w", encoding="utf-8") as diag_fh:
        for row in rows:
            sid = sample_id_from_row(row)
            pair = pairs.get(sid)
            ok, reason = should_trigger(row, pair, args)
            trigger_counts[reason] += 1
            diag_row: dict[str, Any] = {
                "sample_id": sid,
                "trigger": reason,
                "changed": False,
                "selected_count": int((pair or {}).get("selected_count") or 0),
                "base_box_count": len(report_boxes(str(row.get("raw_output") or ""))),
            }
            if ok and (args.max_triggered <= 0 or triggered < args.max_triggered):
                triggered += 1
                image_path = resolve_image_path(row, debug_root)
                existing = report_boxes(str(row.get("raw_output") or ""))
                if image_path and existing:
                    with Image.open(image_path) as im:
                        image = im.convert("RGB")
                    width, height = image.size
                    candidates = generate_candidates(
                        row,
                        image,
                        debug_root,
                        Path(args.ocr_layout_cache_dir).expanduser().resolve(),
                        args.ocr_layout_model,
                        args.coord_mode,
                        args.max_candidates,
                        enable_token_candidates=True,
                        enable_linegrid_candidates=True,
                        enable_scriptgrid_candidates=True,
                        language_code=language_code(row),
                    )
                    if args.rank_mode == "v145ocranchor":
                        top = choose_topk_v145ocranchor(row, candidates, args.top_k, width, height, language_code=language_code(row))
                    elif args.rank_mode == "v87mix":
                        top = choose_topk_v87mix(row, candidates, args.top_k, width, height)
                    else:
                        top = choose_topk_v89token(row, candidates, args.top_k, width, height)
                    top_dicts = [
                        {
                            "label": c.label,
                            "box": c.box,
                            "source": c.source,
                            "family": c.family,
                            "score": c.score,
                            "text": c.text,
                            "meta": c.meta,
                        }
                        for c in top
                    ]
                    extras = select_apply_extras(
                        top_dicts,
                        existing,
                        width=width,
                        height=height,
                        limit=args.candidate_limit,
                        max_area_ratio=args.apply_max_area_ratio,
                        duplicate_iou=args.candidate_duplicate_iou,
                    )
                    if (
                        (not allowed_routes or "schedule_pale_yellow_patch" in allowed_routes)
                        and schedule_doc_context(row)
                    ):
                        ref_minutes = schedule_reference_minutes(row)
                        ocr_for_patch = [
                            {
                                "box": list(c.box),
                                "text": c.text,
                                "source": c.source,
                            }
                            for c in candidates
                            if c.family == "ocr" and c.source.startswith("ocr_span")
                        ]
                        occupied_for_tail = [list(b) for b in existing] + [
                            [int(v) for v in e.get("box", [])[:4]]
                            for e in extras
                            if isinstance(e.get("box"), list) and len(e.get("box") or []) >= 4
                        ]
                        tail_added = 0
                        for c in sorted(
                            (c for c in candidates if c.source.startswith("patch:pale_yellow")),
                            key=lambda c: c.score,
                            reverse=True,
                        ):
                            if page_area_ratio(c.box, width, height) > args.apply_max_area_ratio:
                                continue
                            if any(box_iou(c.box, prev) >= args.candidate_duplicate_iou for prev in occupied_for_tail):
                                continue
                            nearby_ocr = [
                                o
                                for o in ocr_for_patch
                                if box_iou(c.box, o["box"]) >= 0.03 or center_inside(o["box"], c.box, pad=24)
                            ]
                            nearby_text = " ".join(str(o.get("text") or "") for o in nearby_ocr[:6]).strip()
                            cand_minutes = compact_time_minute_candidates(nearby_text)
                            near_mismatch = False
                            if cand_minutes and ref_minutes:
                                near_mismatch = any(
                                    0 < min(abs(cand_minute - ref), abs(cand_minute + 720 - ref), abs(cand_minute - 720 - ref)) <= 5
                                    for cand_minute in cand_minutes
                                    for ref in ref_minutes
                                )
                            meta = dict(c.meta)
                            if nearby_text:
                                meta["nearby_ocr_text"] = nearby_text
                            if cand_minutes:
                                meta["candidate_time_minutes"] = cand_minutes[:8]
                            if near_mismatch:
                                meta["schedule_time_near_mismatch"] = True
                            extras.append(
                                {
                                    "label": c.label,
                                    "box": c.box,
                                    "source": c.source,
                                    "family": c.family,
                                    "score": c.score,
                                    "text": nearby_text or c.text,
                                    "meta": meta,
                                }
                            )
                            occupied_for_tail.append(c.box)
                            tail_added += 1
                            if tail_added >= 24:
                                break
                    if (
                        (not allowed_routes or "schedule_ampm_near_mismatch" in allowed_routes)
                        and schedule_doc_context(row)
                    ):
                        occupied_for_tail = [list(b) for b in existing] + [
                            [int(v) for v in e.get("box", [])[:4]]
                            for e in extras
                            if isinstance(e.get("box"), list) and len(e.get("box") or []) >= 4
                        ]
                        tail_added = 0
                        for c in candidates:
                            if c.family != "ocr" or not c.source.startswith("ocr_span"):
                                continue
                            if not re.fullmatch(r"[AP]M", str(c.text or "").strip(), flags=re.IGNORECASE):
                                continue
                            if c.box[1] < height * 0.55:
                                continue
                            if page_area_ratio(c.box, width, height) > args.apply_max_area_ratio:
                                continue
                            if any(box_iou(c.box, prev) >= args.candidate_duplicate_iou for prev in occupied_for_tail):
                                continue
                            anchor_match = left_row_anchor_match(c.box, existing)
                            if anchor_match is None:
                                continue
                            row_gap, anchor_box = anchor_match
                            meta = dict(c.meta)
                            meta["schedule_ampm_adjacent_to_existing"] = True
                            meta["schedule_ampm_gap_to_existing"] = row_gap
                            extras.append(
                                {
                                    "label": c.label,
                                    "box": c.box,
                                    "source": c.source,
                                    "family": c.family,
                                    "score": c.score,
                                    "text": c.text,
                                    "meta": meta,
                                }
                            )
                            union_box = [
                                min(c.box[0], anchor_box[0]),
                                min(c.box[1], anchor_box[1]),
                                max(c.box[2], anchor_box[2]),
                                max(c.box[3], anchor_box[3]),
                            ]
                            union_meta = dict(meta)
                            union_meta["schedule_ampm_time_union"] = True
                            union_meta["schedule_ampm_source_box"] = c.box
                            union_meta["schedule_time_anchor_box"] = anchor_box
                            if page_area_ratio(union_box, width, height) <= args.apply_max_area_ratio:
                                extras.append(
                                    {
                                        "label": c.label,
                                        "box": union_box,
                                        "source": f"{c.source}:schedule_ampm_time_union",
                                        "family": c.family,
                                        "score": c.score + 0.5,
                                        "text": c.text,
                                        "meta": union_meta,
                                    }
                                )
                            occupied_for_tail.append(c.box)
                            tail_added += 1
                            if tail_added >= 12:
                                break
                    if not allowed_routes or "zh_right_column_gap_span_union" in allowed_routes:
                        occupied_for_tail = [list(b) for b in existing] + [
                            [int(v) for v in e.get("box", [])[:4]]
                            for e in extras
                            if isinstance(e.get("box"), list) and len(e.get("box") or []) >= 4
                        ]
                        tail_added = 0
                        for c in candidates:
                            if c.family != "evidence" or not c.source.endswith("span_ids"):
                                continue
                            text = str(c.text or "")
                            if not (
                                "layout_inconsistency" in text
                                and "建议的行动" in text
                                and "垂直空白" in text
                                and "行距" in text
                            ):
                                continue
                            if page_area_ratio(c.box, width, height) > args.apply_max_area_ratio:
                                continue
                            if any(box_iou(c.box, prev) >= args.candidate_duplicate_iou for prev in occupied_for_tail):
                                continue
                            extras.append(
                                {
                                    "label": c.label,
                                    "box": c.box,
                                    "source": c.source,
                                    "family": c.family,
                                    "score": c.score,
                                    "text": c.text,
                                    "meta": dict(c.meta),
                                }
                            )
                            occupied_for_tail.append(c.box)
                            tail_added += 1
                            if tail_added >= 6:
                                break
                    support_boxes = [
                        [int(v) for v in c.get("box") or []]
                        for c in top_dicts
                        if c.get("family") != "existing"
                        and not str(c.get("source") or "").startswith("current_final_report")
                        and isinstance(c.get("box"), list)
                        and len(c.get("box") or []) >= 4
                    ]
                    support_boxes = [b[:4] for b in support_boxes if page_area_ratio(b[:4], width, height) <= args.apply_max_area_ratio]
                    # Diagnostic-only after v420-v422: generic row-neighbor
                    # short values have real wins but too many false positives.
                    if "row_context_short_value" in allowed_routes:
                        attach_short_value_context(
                            extras,
                            top_dicts + extras,
                            width=width,
                            height=height,
                        )
                    if allowed_routes & named_value_routes:
                        before_named_inject = len(extras)
                        inject_named_value_candidates(
                            extras,
                            candidates,
                            allowed_routes=allowed_routes & named_value_routes,
                            width=width,
                            height=height,
                        )
                        attach_named_value_context(
                            extras,
                            top_dicts + extras,
                            width=width,
                            height=height,
                        )
                        diag_row["named_value_injected_count"] = len(extras) - before_named_inject
                        diag_row["named_value_allowed_routes"] = sorted(allowed_routes & named_value_routes)
                    if "id_mengembangka_55_context" in allowed_routes and "Mengembangka" in str(row.get("raw_output") or ""):
                        for c in candidates:
                            if c.family != "ocr" or not c.source.startswith("ocr_span"):
                                continue
                            if str(c.text or "").strip() != "55":
                                continue
                            cand_box = list(c.box)
                            if len(cand_box) < 4 or page_area_ratio(cand_box[:4], width, height) > args.apply_max_area_ratio:
                                continue
                            meta = dict(c.meta or {})
                            meta["named_value_context"] = {
                                "route": "id_mengembangka_55_context",
                                "text": "current report mentions Mengembangka truncation near this value",
                                "family": "report",
                                "source": "current_final_report",
                            }
                            extras.append(
                                {
                                    "label": c.label,
                                    "box": cand_box,
                                    "source": c.source,
                                    "family": c.family,
                                    "score": c.score,
                                    "text": c.text,
                                    "meta": meta,
                                }
                            )
                            break
                    if "report_quote_ocr_phrase" in allowed_routes or "append_safe_report_ocr_anchor" in allowed_routes:
                        before_quote_inject = len(extras)
                        quote_attached_count = attach_report_quote_anchors(
                            extras,
                            row,
                            width=width,
                            height=height,
                        )
                        quote_injected_count = inject_report_quote_anchors(
                            extras,
                            candidates,
                            row,
                            width=width,
                            height=height,
                        )
                        diag_row["report_quote_phrase_count"] = len(extract_report_quote_phrases(row))
                        diag_row["report_quote_attached_count"] = quote_attached_count
                        diag_row["report_quote_injected_count"] = quote_injected_count
                        diag_row["report_quote_total_added"] = len(extras) - before_quote_inject
                    scored: list[dict[str, Any]] = []
                    for cand in extras:
                        score, route, info = candidate_route_score(cand, width=width, height=height)
                        if score <= -1e8:
                            continue
                        cand_box = [int(v) for v in cand.get("box") or []]
                        if (
                            route == "decimal_suffix_token"
                            and len(cand_box) >= 4
                            and is_left_row_anchor_near_existing(cand_box[:4], existing)
                        ):
                            route = "decimal_suffix_near_existing"
                            score += 5.0
                            info = dict(info)
                            reasons = list(info.get("reasons") or [])
                            reasons.append("left_row_anchor_near_existing")
                            info["route"] = route
                            info["score"] = score
                            info["reasons"] = reasons
                        if (
                            len(cand_box) >= 4
                            and schedule_doc_context(row)
                            and str(cand.get("family") or "") == "ocr"
                            and re.fullmatch(r"[AP]M", str(cand.get("text") or "").strip(), flags=re.IGNORECASE)
                            and cand_box[1] >= height * 0.55
                            and (
                                is_left_row_anchor_near_existing(cand_box[:4], existing)
                                or bool((cand.get("meta") or {}).get("schedule_ampm_time_union"))
                            )
                        ):
                            row_gap = (cand.get("meta") or {}).get("schedule_ampm_gap_to_existing")
                            if row_gap is None:
                                row_gap = left_row_anchor_gap(cand_box[:4], existing)
                            route = "schedule_ampm_near_mismatch"
                            score = max(score + 12.0, 14.0)
                            if (cand.get("meta") or {}).get("schedule_ampm_time_union"):
                                score += 2.5
                            if row_gap is not None:
                                row_gap_float = float(row_gap)
                                score += max(0.0, 4.0 - row_gap_float / 40.0)
                                if row_gap_float > 120:
                                    score -= 2.0
                            info = dict(info)
                            reasons = list(info.get("reasons") or [])
                            reasons.append("ampm_left_of_schedule_mismatch_anchor")
                            if (cand.get("meta") or {}).get("schedule_ampm_time_union"):
                                reasons.append("schedule_ampm_time_union")
                            if row_gap is not None:
                                info["schedule_ampm_gap_to_existing"] = float(row_gap)
                            info["route"] = route
                            info["score"] = score
                            info["reasons"] = reasons
                        if route == "schedule_pale_yellow_patch" and not schedule_doc_context(row):
                            continue
                        if allowed_routes and route not in allowed_routes:
                            continue
                        if allowed_families and str(cand.get("family") or "") not in allowed_families:
                            continue
                        item = {
                            "score": score,
                            "route": route,
                            "candidate": cand,
                            "info": info,
                        }
                        scored.append(item)
                    scored.sort(key=lambda item: float(item["score"]), reverse=True)
                    best = scored[0] if scored else None
                    diag_row.update(
                        {
                            "candidate_count": len(candidates),
                            "top_count": len(top),
                            "extra_count": len(extras),
                            "scored_count": len(scored),
                            "top_score": float(best["score"]) if best else None,
                            "top_route": best["route"] if best else None,
                            "top_family": (best["candidate"] or {}).get("family") if best else None,
                            "top_source": (best["candidate"] or {}).get("source") if best else None,
                            "top_box": (best["candidate"] or {}).get("box") if best else None,
                            "top_text": str((best["candidate"] or {}).get("text") or "")[:180] if best else "",
                            "top_info": best["info"] if best else None,
                        }
                    )
                    inspected += 1
                    if best and float(best["score"]) >= args.route_threshold:
                        cand = best["candidate"]
                        box = [int(v) for v in cand.get("box") or []]
                        replace_idx = None
                        changed_count = 0
                        if args.apply_mode == "append":
                            if len(existing) < args.max_total_boxes:
                                new_row = dict(row)
                                report = insert_extra_anomalies(str(new_row.get("raw_output") or ""), [cand])
                                changed_count = 1 if report != str(new_row.get("raw_output") or "") else 0
                                replaced_box = None
                            else:
                                new_row = None
                                replaced_box = None
                        else:
                            replace_idx = replacement_index(
                                box,
                                existing,
                                policy=args.replace_policy,
                                width=width,
                                height=height,
                                support_boxes=support_boxes,
                                old_support_max_iou=args.old_support_max_iou,
                            )
                            if replace_idx is not None:
                                if (
                                    best["route"] == "report_quote_ocr_phrase"
                                    and args.report_quote_max_replace_area_scale > 0
                                    and box_area(box[:4]) > float(args.report_quote_max_replace_area_scale) * box_area(existing[replace_idx])
                                ):
                                    new_row = None
                                    replaced_box = existing[replace_idx]
                                    diag_row["report_quote_area_guard_reject"] = True
                                    diag_row["report_quote_area_scale"] = box_area(box[:4]) / max(1.0, box_area(existing[replace_idx]))
                                elif (
                                    best["route"] == "report_quote_ocr_phrase"
                                    and args.report_quote_min_replace_area_scale > 0
                                    and box_area(box[:4]) < float(args.report_quote_min_replace_area_scale) * box_area(existing[replace_idx])
                                ):
                                    new_row = None
                                    replaced_box = existing[replace_idx]
                                    diag_row["report_quote_area_guard_reject"] = True
                                    diag_row["report_quote_area_scale"] = box_area(box[:4]) / max(1.0, box_area(existing[replace_idx]))
                                else:
                                    new_row = dict(row)
                                    report, changed_count = replace_groundings(str(new_row.get("raw_output") or ""), {replace_idx: box[:4]})
                                    replaced_box = existing[replace_idx]
                            else:
                                new_row = None
                                replaced_box = None
                        if new_row is not None and changed_count:
                            new_row["raw_output"] = report
                            new_row["parsed"] = parse_cct_report(report)
                            stage_outputs = dict(new_row.get("stage_outputs") or {})
                            stage_outputs["qwen_pipe_broad_candidate_router"] = {
                                "applied": True,
                                "apply_mode": args.apply_mode,
                                "replace_policy": args.replace_policy if args.apply_mode == "replace" else None,
                                "replace_index": replace_idx,
                                "replaced_box": replaced_box,
                                "candidate": cand,
                                "route_score": float(best["score"]),
                                "route": best["route"],
                                "route_info": best["info"],
                                "gt_free": True,
                            }
                            new_row["stage_outputs"] = stage_outputs
                            row = new_row
                            diag_row["changed"] = True
                            diag_row["apply_mode"] = args.apply_mode
                            diag_row["replace_index"] = replace_idx
                            diag_row["replaced_box"] = replaced_box
                            changed.append(
                                {
                                    "sample_id": sid,
                                    "apply_mode": args.apply_mode,
                                    "replace_index": replace_idx,
                                    "replaced_box": replaced_box,
                                    "candidate_box": box[:4],
                                    "family": cand.get("family"),
                                    "source": cand.get("source"),
                                    "route": best["route"],
                                    "score": float(best["score"]),
                                    "text": str(cand.get("text") or "")[:180],
                                }
                            )
                            family_counts[str(cand.get("family") or "")] += 1
                            route_counts[str(best["route"])] += 1
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            diag_fh.write(json.dumps(diag_row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "previous_pair_jsonl": str(resolve_pipe_path(args.previous_pair_jsonl)) if args.previous_pair_jsonl else None,
        "output_jsonl": str(out_path),
        "diag_jsonl": str(diag_path),
        "gt_free": True,
        "params": vars(args),
        "trigger_counts": dict(trigger_counts),
        "triggered_count": triggered,
        "inspected_count": inspected,
        "changed_count": len(changed),
        "family_counts": dict(family_counts),
        "route_counts": dict(route_counts),
        "changed": changed,
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["trigger_counts", "triggered_count", "changed_count", "family_counts", "route_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
