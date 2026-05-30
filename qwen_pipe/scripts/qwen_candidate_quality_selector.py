#!/usr/bin/env python3
"""GT-free candidate quality selector for low-localization repair.

The v471 oracle diagnostic showed that OCR/line/token candidates often contain
useful boxes, while v472 showed that naive top-1 replacement still regresses.
This script tests a different mechanism: rerank an existing exhaustive candidate
pool with quality guards that prefer compact, report-supported regions and
penalize footer/page noise, generic numeric hits, and whole-row/table windows.

GT artifacts are never read for selection.  An optional oracle diagnostic path is
used only after selection for local analysis of what the selector chose.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402


GROUNDING_RE = re.compile(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", re.IGNORECASE)


GENERIC_NUMBERS = {
    "0",
    "00",
    "000",
    "1",
    "01",
    "001",
    "2",
    "3",
    "4",
    "5",
    "9",
    "10",
    "11",
    "12",
    "20",
    "21",
    "50",
    "90",
    "99",
    "100",
}

FOOTER_NOISE_RE = re.compile(
    r"^(?:page|p\.?|页|第|copyright|confidential|draft|www\.|https?://)?\s*"
    r"(?:\d{1,3}|[ivxlcdm]{1,6})\s*(?:/|of|共|-)?\s*(?:\d{0,3})\s*$",
    re.IGNORECASE,
)
CONTACT_NOISE_RE = re.compile(r"\b(?:qq|tel|fax|email|e-mail|phone|copyright|all rights reserved)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff\u0e00-\u0e7f\u0600-\u06ff]{2,}", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*(?:%|万元|万|年|月|日)?", re.UNICODE)


def resolve_pipe_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PIPE_ROOT / path


def setup_debug_import(debug_root: Path) -> None:
    docshield_dir = debug_root / "baselines" / "DocShield"
    if not docshield_dir.exists():
        raise SystemExit(f"DocShield helper directory not found: {docshield_dir}")
    for path in (docshield_dir, PIPE_ROOT / "scripts"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def clamp_box(values: list[float], width: int, height: int) -> list[int] | None:
    if len(values) < 4:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in values[:4]]
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def replace_groundings(report: str, replacements: dict[int, list[int]]) -> tuple[str, int]:
    if not replacements:
        return report, 0
    idx = 0
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx, count
        cur_idx = idx
        idx += 1
        box = replacements.get(cur_idx)
        if not box:
            return match.group(0)
        count += 1
        return f"{match.group(1)}{box}"

    return GROUNDING_RE.sub(repl, report), count


def extra_reason_text(cand: dict[str, Any]) -> str:
    family = str(cand.get("family") or "candidate")
    text = re.sub(r"\s+", " ", str(cand.get("text") or "")).strip()
    if len(text) > 220:
        text = text[:217].rstrip() + "..."
    reason = "Additional localization candidate selected by GT-free OCR/candidate quality scoring."
    if text:
        reason += f" Related local text: {text}"
    reason += f" Candidate family: {family}."
    return reason


def insert_extra_anomalies(report: str, extras: list[dict[str, Any]]) -> str:
    if not extras:
        return report
    block: list[str] = []
    for idx, cand in enumerate(extras, start=1):
        box = cand["box"]
        family = str(cand.get("family") or "candidate")
        block.extend(
            [
                f"### ANOMALY_QUALITY_EXTRA_{idx:03d}: Quality-Selected Candidate ({family})",
                f"[GROUNDING]:{box}",
                f"[REASON]: {extra_reason_text(cand)}",
                "",
            ]
        )
    extra_text = "\n".join(block)
    marker = re.search(r"\n\s*-{3,}\s*\n\s*##\s*SUMMARY|\n\s*##\s*SUMMARY", report, re.IGNORECASE)
    if marker:
        return report[: marker.start()] + "\n\n" + extra_text + report[marker.start():]
    return report.rstrip() + "\n\n" + extra_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_id_from_row(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)


def box_area(box: list[int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def page_area_ratio(box: list[int], width: int, height: int) -> float:
    return box_area(box) / max(1.0, float(width * height))


def box_iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    return float(inter) / max(1.0, box_area(a) + box_area(b) - inter)


def center_distance(a: list[int], b: list[int], width: int, height: int) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return math.hypot(ax - bx, ay - by) / max(1.0, math.hypot(width, height))


def rank_payload(cand: dict[str, Any]) -> dict[str, Any]:
    meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
    for key in ("v145ocranchor_rank", "v446physdiverse_rank", "v87b_rank", "v87_rank"):
        payload = meta.get(key)
        if isinstance(payload, dict):
            return payload
    return {}


def normalize_token(value: str) -> str:
    return re.sub(r"\W+", "", str(value or "").lower(), flags=re.UNICODE)


def useful_query_hits(hits: list[Any]) -> list[str]:
    useful: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        text = re.sub(r"\s+", " ", str(hit or "")).strip()
        norm = normalize_token(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if norm in GENERIC_NUMBERS:
            continue
        if len(norm) <= 2 and not re.search(r"[\u4e00-\u9fff\u0e00-\u0e7f\u0600-\u06ff]", norm):
            continue
        useful.append(text)
    return useful


def useful_number_hits(hits: list[Any]) -> list[str]:
    useful: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        text = str(hit or "").strip()
        norm = re.sub(r"\D+", "", text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if norm in GENERIC_NUMBERS:
            continue
        if len(norm) < 3 and not re.search(r"[%万年月日:/.-]", text):
            continue
        useful.append(text)
    return useful


def report_terms(row: dict[str, Any]) -> set[str]:
    text = str(row.get("raw_output") or "")
    terms: set[str] = set()
    for token in TOKEN_RE.findall(text):
        norm = normalize_token(token)
        if len(norm) >= 4 or re.search(r"[\u4e00-\u9fff\u0e00-\u0e7f\u0600-\u06ff]", norm):
            terms.add(norm)
    for num in NUMBER_RE.findall(text):
        norm = re.sub(r"\D+", "", num)
        if norm and norm not in GENERIC_NUMBERS:
            terms.add(norm)
    return terms


def text_report_overlap(text: str, terms: set[str]) -> int:
    if not text or not terms:
        return 0
    count = 0
    for token in TOKEN_RE.findall(text):
        if normalize_token(token) in terms:
            count += 1
    for num in NUMBER_RE.findall(text):
        norm = re.sub(r"\D+", "", num)
        if norm in terms:
            count += 1
    return count


def candidate_local_text(cand: dict[str, Any]) -> str:
    """Return the visible anchor text, not the parent OCR context after '::'."""
    text = re.sub(r"\s+", " ", str(cand.get("text") or "")).strip()
    family = str(cand.get("family") or "")
    source = str(cand.get("source") or "")
    if " :: " in text and (family in {"token", "linegrid"} or "subspan" in source):
        return text.split(" :: ", 1)[0].strip()
    return text


def is_footer_noise(text: str, box: list[int], width: int, height: int) -> bool:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return False
    near_edge = box[1] < 0.06 * height or box[3] > 0.94 * height
    if FOOTER_NOISE_RE.match(value):
        return True
    return near_edge and bool(CONTACT_NOISE_RE.search(value)) and len(value) < 32


def has_many_generic_numbers(text: str) -> bool:
    nums = [re.sub(r"\D+", "", n) for n in NUMBER_RE.findall(text or "")]
    nums = [n for n in nums if n]
    if len(nums) < 5:
        return False
    generic = sum(1 for n in nums if n in GENERIC_NUMBERS)
    return generic / max(1, len(nums)) >= 0.55


def score_candidate(
    cand: dict[str, Any],
    row: dict[str, Any],
    *,
    width: int,
    height: int,
    existing_boxes: list[list[int]],
    terms: set[str],
    max_area_ratio: float,
) -> tuple[float, dict[str, Any]]:
    raw_box = cand.get("box")
    if not isinstance(raw_box, list) or len(raw_box) < 4:
        return -1e9, {"reject": "invalid_box"}
    box = clamp_box([float(v) for v in raw_box[:4]], width, height)
    if not box:
        return -1e9, {"reject": "empty_box"}

    family = str(cand.get("family") or "")
    source = str(cand.get("source") or "")
    text = str(cand.get("text") or "")
    local_text = candidate_local_text(cand)
    rank = rank_payload(cand)
    area_ratio = page_area_ratio(box, width, height)
    w_ratio = max(0, box[2] - box[0]) / max(1.0, float(width))
    h_ratio = max(0, box[3] - box[1]) / max(1.0, float(height))
    query_hits = useful_query_hits(list(rank.get("query_hits") or (cand.get("meta") or {}).get("query_hits") or []))
    number_hits = useful_number_hits(list(rank.get("number_hits") or (cand.get("meta") or {}).get("number_hits") or []))
    visual_hits = list(rank.get("visual_hits") or (cand.get("meta") or {}).get("visual_hits") or [])
    token_hits = int(rank.get("token_hit_count") or (cand.get("meta") or {}).get("token_hit_count") or 0)
    report_overlap = text_report_overlap(local_text, terms)
    duplicate_iou = max((box_iou(box, old) for old in existing_boxes), default=0.0)
    min_dist = min((center_distance(box, old, width, height) for old in existing_boxes), default=0.5)

    info: dict[str, Any] = {
        "box": box,
        "family": family,
        "source": source,
        "area_ratio": area_ratio,
        "width_ratio": w_ratio,
        "height_ratio": h_ratio,
        "useful_query_hits": query_hits[:8],
        "useful_number_hits": number_hits[:8],
        "visual_hits": visual_hits[:8],
        "token_hits": token_hits,
        "report_overlap": report_overlap,
        "local_text": local_text[:120],
        "duplicate_iou": duplicate_iou,
        "min_center_distance": min_dist,
    }

    if family == "existing" or source.startswith("current_final_report"):
        return -1e9, {**info, "reject": "existing"}
    if area_ratio <= 0.0 or area_ratio > max_area_ratio:
        return -1e9, {**info, "reject": "area_gate"}
    if area_ratio < 0.000002 and len(query_hits) + len(number_hits) + report_overlap < 2:
        return -1e9, {**info, "reject": "tiny_unsupported"}
    aspect = (max(1, box[3] - box[1]) / max(1, box[2] - box[0]))
    if family == "token" and aspect >= 3.2 and h_ratio >= 0.045:
        return -1e9, {**info, "reject": "vertical_token_strip"}
    if family == "linegrid" and aspect >= 4.5 and h_ratio >= 0.070:
        return -1e9, {**info, "reject": "vertical_line_strip"}
    if family == "token" and normalize_token(local_text) in GENERIC_NUMBERS and len(query_hits) + len(number_hits) < 3:
        return -1e9, {**info, "reject": "generic_token_anchor"}
    if is_footer_noise(local_text, box, width, height) and len(query_hits) + report_overlap < 2:
        return -1e9, {**info, "reject": "footer_noise"}
    if family in {"grid", "paragraph"} and len(query_hits) + len(visual_hits) + report_overlap < 4:
        return -1e9, {**info, "reject": "broad_family_unsupported"}
    if duplicate_iou >= 0.82:
        return -1e9, {**info, "reject": "duplicate_existing"}

    family_weight = {
        "token": 4.0,
        "ocr": 3.0,
        "linegrid": 2.2,
        "evidence": 1.1,
        "patch": 2.4,
        "row": -1.4,
        "grid": -3.0,
        "paragraph": -3.8,
    }.get(family, 0.0)
    score = family_weight
    score += min(float(rank.get("score") or cand.get("score") or 0.0), 24.0) * 0.12
    score += min(float(rank.get("query_score") or 0.0), 14.0) * 0.20
    score += min(len(query_hits), 6) * 1.25
    score += min(len(number_hits), 5) * 0.85
    score += min(token_hits, 6) * 0.45
    score += min(len(visual_hits), 4) * 0.75
    score += min(report_overlap, 6) * 0.60

    if 0.000015 <= area_ratio <= 0.0045:
        score += 1.8
    elif area_ratio <= 0.010:
        score += 0.7
    if family in {"ocr", "token"} and source.startswith(("ocr_span:qwen_ocr", "ocr_token_subspan")):
        score += 0.55
    if source.startswith("ocr_span:table") and w_ratio > 0.55:
        score -= 2.2
    if w_ratio > 0.68 and len(query_hits) + report_overlap < 5:
        score -= 3.0
    if family == "row" and w_ratio > 0.50:
        score -= 2.0
    if family == "evidence" and area_ratio > 0.008:
        score -= 2.0
    if has_many_generic_numbers(local_text) and len(number_hits) <= 2:
        score -= 2.4
    if len(local_text.strip()) <= 3 and len(number_hits) == 0 and len(query_hits) == 0:
        score -= 2.5
    if min_dist > 0.42 and len(query_hits) + report_overlap < 3:
        score -= 1.8

    info["score"] = score
    return score, info


def choose_replace_index(
    candidate_box: list[int],
    existing_boxes: list[list[int]],
    *,
    width: int,
    height: int,
    policy: str,
) -> int | None:
    if not existing_boxes:
        return None
    if policy == "closest":
        return max(range(len(existing_boxes)), key=lambda idx: box_iou(candidate_box, existing_boxes[idx]))
    if policy == "farthest":
        return min(range(len(existing_boxes)), key=lambda idx: box_iou(candidate_box, existing_boxes[idx]))
    if policy == "largest":
        return max(range(len(existing_boxes)), key=lambda idx: box_area(existing_boxes[idx]))
    # Prefer removing broad boxes that are spatially unrelated to the new support.
    return max(
        range(len(existing_boxes)),
        key=lambda idx: page_area_ratio(existing_boxes[idx], width, height)
        + 0.35 * center_distance(candidate_box, existing_boxes[idx], width, height)
        - 0.25 * box_iou(candidate_box, existing_boxes[idx]),
    )


def choose_action(
    candidate: dict[str, Any],
    quality: dict[str, Any],
    existing_boxes: list[list[int]],
    *,
    width: int,
    height: int,
    apply_mode: str,
    replace_policy: str,
    max_total_boxes: int,
) -> tuple[str, int | None]:
    box = quality["box"]
    duplicate_iou = float(quality.get("duplicate_iou") or 0.0)
    if apply_mode == "append":
        if duplicate_iou >= 0.50 or len(existing_boxes) >= max_total_boxes:
            return "skip", None
        return "append", None
    if apply_mode == "replace":
        return "replace", choose_replace_index(box, existing_boxes, width=width, height=height, policy=replace_policy)

    # Auto mode: compact, report-supported candidates add missing recall; broad
    # candidates retarget a likely bad old box instead of increasing box count.
    family = str(candidate.get("family") or "")
    area_ratio = float(quality.get("area_ratio") or 0.0)
    support = len(quality.get("useful_query_hits") or []) + len(quality.get("useful_number_hits") or []) + int(
        quality.get("report_overlap") or 0
    )
    if (
        family in {"token", "ocr", "linegrid"}
        and area_ratio <= 0.0045
        and duplicate_iou < 0.35
        and support >= 2
        and len(existing_boxes) < max_total_boxes
    ):
        return "append", None
    return "replace", choose_replace_index(box, existing_boxes, width=width, height=height, policy=replace_policy)


def load_oracle(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = resolve_pipe_path(path)
    if not p.exists():
        return {}
    return {str(r.get("sample_id")): r for r in read_jsonl(p)}


def oracle_overlap(selected_box: list[int], oracle_row: dict[str, Any] | None) -> dict[str, Any]:
    if not oracle_row:
        return {}
    best_box = oracle_row.get("best_oracle_box")
    deploy_box = oracle_row.get("best_deployable_box")
    out: dict[str, Any] = {}
    if isinstance(best_box, list) and len(best_box) >= 4:
        out["selected_vs_best_oracle_iou"] = box_iou(selected_box, [int(v) for v in best_box[:4]])
        out["best_oracle_family"] = oracle_row.get("best_oracle_family")
        out["best_oracle_delta"] = oracle_row.get("best_oracle_delta")
    if isinstance(deploy_box, list) and len(deploy_box) >= 4:
        out["selected_vs_best_deployable_iou"] = box_iou(selected_box, [int(v) for v in deploy_box[:4]])
        out["best_deployable_family"] = oracle_row.get("best_deployable_family")
        out["best_deployable_delta"] = oracle_row.get("best_deployable_delta")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--oracle-jsonl", default="")
    parser.add_argument("--candidate-source", choices=["all", "top"], default="all")
    parser.add_argument("--apply-mode", choices=["auto", "append", "replace"], default="auto")
    parser.add_argument("--replace-policy", choices=["largest", "large_unmatched", "closest", "farthest"], default="large_unmatched")
    parser.add_argument("--score-threshold", type=float, default=9.0)
    parser.add_argument("--max-area-ratio", type=float, default=0.012)
    parser.add_argument("--max-total-boxes", type=int, default=9)
    parser.add_argument("--limit-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    candidate_path = resolve_pipe_path(args.candidate_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = read_jsonl(input_path)
    candidate_rows = {str(r.get("sample_id")): r for r in read_jsonl(candidate_path)}
    oracle_rows = load_oracle(args.oracle_jsonl)

    selected_records: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    changed = 0
    rows_seen = 0
    rows_with_candidates = 0
    rows_below_threshold = 0
    family_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    oracle_iou_values: list[float] = []

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_seen += 1
            sid = sample_id_from_row(row)
            diag = candidate_rows.get(sid)
            if not diag:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            if args.limit_samples and rows_with_candidates >= args.limit_samples:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            rows_with_candidates += 1
            width = int(diag.get("width") or row.get("width") or 0)
            height = int(diag.get("height") or row.get("height") or 0)
            existing = report_boxes(str(row.get("raw_output") or ""))
            terms = report_terms(row)
            source_key = "candidates" if args.candidate_source == "all" else "top_candidates"
            candidates = list(diag.get(source_key) or [])
            scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for cand in candidates:
                score, quality = score_candidate(
                    cand,
                    row,
                    width=width,
                    height=height,
                    existing_boxes=existing,
                    terms=terms,
                    max_area_ratio=args.max_area_ratio,
                )
                if score <= -1e8:
                    reject_counts[str(quality.get("reject") or "unknown")] += 1
                    continue
                scored.append((score, cand, quality))
            scored.sort(key=lambda item: item[0], reverse=True)
            if not scored or scored[0][0] < args.score_threshold:
                rows_below_threshold += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            score, cand, quality = scored[0]
            action, replace_idx = choose_action(
                cand,
                quality,
                existing,
                width=width,
                height=height,
                apply_mode=args.apply_mode,
                replace_policy=args.replace_policy,
                max_total_boxes=args.max_total_boxes,
            )
            if action == "skip" or (action == "replace" and replace_idx is None):
                reject_counts["action_skip"] += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            selected_box = [int(v) for v in quality["box"]]
            selected = dict(cand)
            selected["box"] = selected_box
            selected["quality_selector_score"] = score
            selected["quality_selector"] = quality
            stage_outputs = dict(row.get("stage_outputs") or {})
            if action == "replace":
                new_report, replaced_count = replace_groundings(str(row.get("raw_output") or ""), {int(replace_idx): selected_box})
                if replaced_count <= 0:
                    reject_counts["replace_failed"] += 1
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                row = dict(row)
                row["raw_output"] = new_report
                row["parsed"] = parse_cct_report(new_report)
                selected["replace_index"] = int(replace_idx)
                selected["replaced_box"] = existing[int(replace_idx)]
            else:
                row = dict(row)
                new_report = insert_extra_anomalies(str(row.get("raw_output") or ""), [selected])
                row["raw_output"] = new_report
                row["parsed"] = parse_cct_report(new_report)

            oracle_info = oracle_overlap(selected_box, oracle_rows.get(sid))
            if "selected_vs_best_oracle_iou" in oracle_info:
                oracle_iou_values.append(float(oracle_info["selected_vs_best_oracle_iou"]))
            selected_record = {
                "sample_id": sid,
                "action": action,
                "score": score,
                "candidate": selected,
                "oracle_diagnostic": oracle_info,
                "top_scores": [
                    {
                        "score": s,
                        "family": str(c.get("family") or ""),
                        "source": str(c.get("source") or ""),
                        "box": q.get("box"),
                        "text": str(c.get("text") or "")[:120],
                        "query_hits": q.get("useful_query_hits"),
                        "number_hits": q.get("useful_number_hits"),
                        "report_overlap": q.get("report_overlap"),
                        "area_ratio": q.get("area_ratio"),
                    }
                    for s, c, q in scored[:5]
                ],
            }
            selected_records.append(selected_record)
            family_counts[str(cand.get("family") or "")] += 1
            action_counts[action] += 1
            stage_outputs["qwen_pipe_candidate_quality_selector"] = {
                "applied": True,
                "candidate_source": args.candidate_source,
                "apply_mode": args.apply_mode,
                "action": action,
                "replace_policy": args.replace_policy,
                "score_threshold": args.score_threshold,
                "selected": selected,
                "gt_free": True,
                "policy": "GT-free candidate-quality reranker over exhaustive OCR/line/token/evidence candidates; GT oracle path is diagnostic only and not used for selection.",
            }
            row["stage_outputs"] = stage_outputs
            changed += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(input_path),
        "candidate_jsonl": str(candidate_path),
        "output_jsonl": str(output_path),
        "candidate_source": args.candidate_source,
        "apply_mode": args.apply_mode,
        "replace_policy": args.replace_policy,
        "score_threshold": args.score_threshold,
        "max_area_ratio": args.max_area_ratio,
        "rows_seen": rows_seen,
        "rows_with_candidate_diag": rows_with_candidates,
        "changed": changed,
        "rows_below_threshold": rows_below_threshold,
        "selected_family_counts": dict(family_counts),
        "action_counts": dict(action_counts),
        "reject_counts": dict(reject_counts),
        "mean_selected_vs_best_oracle_iou": (sum(oracle_iou_values) / len(oracle_iou_values)) if oracle_iou_values else None,
        "selected_records": selected_records,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
