#!/usr/bin/env python3
"""GT-free Thai redundant-box replacement with broad OCR/evidence candidates.

The v263 diagnostic showed that some Thai low-localization wins come from
replacing redundant same-line or duplicate boxes with a broad candidate from a
different region.  This script turns that observation into an inference-side
heuristic: only samples with clear existing-box redundancy are eligible, and
candidate selection uses OCR/evidence/image features only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
    generate_candidates,
    page_area_ratio,
    read_jsonl,
    sample_id_from_row,
    select_apply_extras,
)
from qwen_pair_replace_model import conclusion_is_forged, language_code  # noqa: E402
from qwen_text_crop_verify import area, resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


VISUAL_RE = re.compile(r"artifact|red|black|edge|blur|pixel|font|bold|weight|glitch|block", re.I)


def load_languages(eval_json: str | None) -> dict[str, str]:
    if not eval_json:
        return {}
    path = resolve_pipe_path(eval_json)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(sample.get("sample_id") or ""): str(sample.get("language_code") or "")
        for sample in data.get("samples") or []
        if sample.get("sample_id")
    }


def rank_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    meta = candidate.get("meta") or {}
    return meta.get("v145ocranchor_rank") or meta.get("v87b_rank") or meta.get("v87_rank") or {}


def box_area(box: list[int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def intersection(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def contain_fraction(inner: list[int], outer: list[int]) -> float:
    return intersection(inner, outer) / max(1.0, box_area(inner))


def y_overlap_fraction(a: list[int], b: list[int]) -> float:
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def redundant_indices(boxes: list[list[int]], *, duplicate_iou: float, same_row_contain: float) -> dict[int, float]:
    scores: dict[int, float] = {}
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i >= j:
                continue
            iou = box_iou(a, b)
            same_row = y_overlap_fraction(a, b) >= 0.65
            contain_ab = contain_fraction(a, b)
            contain_ba = contain_fraction(b, a)
            if iou >= duplicate_iou:
                # Later duplicate is usually lower-value while preserving count.
                keep = j if box_area(a) >= box_area(b) else i
                scores[keep] = max(scores.get(keep, 0.0), 2.0 + iou)
            elif same_row and max(contain_ab, contain_ba) >= same_row_contain:
                smaller = i if box_area(a) <= box_area(b) else j
                scores[smaller] = max(scores.get(smaller, 0.0), 1.0 + max(contain_ab, contain_ba))
    return scores


def candidate_support(candidate: dict[str, Any], args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    rank = rank_payload(candidate)
    source = str(candidate.get("source") or "")
    text = str(candidate.get("text") or "")
    visual_hits_raw = rank.get("visual_hits") or []
    visual_hits = [str(v) for v in visual_hits_raw if str(v).lower() not in {"row", "layout", "table", "spacing"}]
    number_hits = rank.get("number_hits") or []
    token_hits = int(rank.get("token_hit_count") or (candidate.get("meta") or {}).get("token_hit_count") or 0)
    query_score = float(rank.get("query_score") or 0.0)
    rank_score = float(rank.get("score") or candidate.get("score") or 0.0)
    visual_signal = 1.0 if visual_hits or VISUAL_RE.search(text) else 0.0
    numeric_signal = 1.0 if number_hits else 0.0
    support = rank_score + args.visual_bonus * visual_signal + 0.75 * numeric_signal + 0.15 * token_hits
    if (
        args.verbose_visual_evidence_penalty > 0
        and args.verbose_visual_evidence_penalty_chars > 0
        and source == "stage_evidence_candidate"
        and visual_signal > 0.0
        and len(text) > args.verbose_visual_evidence_penalty_chars
    ):
        support -= args.verbose_visual_evidence_penalty
    return support, {
        "rank_score": rank_score,
        "query_score": query_score,
        "token_hit_count": token_hits,
        "number_hits": number_hits,
        "visual_hits": visual_hits,
        "visual_signal": visual_signal,
        "numeric_signal": numeric_signal,
    }


def candidate_is_supported(candidate: dict[str, Any], support_meta: dict[str, Any], args: argparse.Namespace) -> bool:
    family = str(candidate.get("family") or "")
    source = str(candidate.get("source") or "")
    text = str(candidate.get("text") or "")
    if args.candidate_family_allowlist and family not in args.candidate_family_allowlist:
        return False
    if args.candidate_source_allowlist and not any(source.startswith(prefix) for prefix in args.candidate_source_allowlist):
        return False
    if args.candidate_text_include_regex and not re.search(args.candidate_text_include_regex, text, re.I):
        return False
    if family not in {"evidence", "ocr", "linegrid", "row", "patch"}:
        return False
    if family == "row" and support_meta["visual_signal"] <= 0.0 and support_meta["numeric_signal"] <= 0.0:
        return False
    if (
        args.reject_verbose_visual_evidence_chars > 0
        and source == "stage_evidence_candidate"
        and support_meta["visual_signal"] > 0.0
        and len(text) > args.reject_verbose_visual_evidence_chars
    ):
        return False
    if (
        source.startswith("stage_evidence_span_id")
        and support_meta["visual_signal"] > 0.0
        and len(text) > args.max_visual_span_union_chars
    ):
        return False
    if (
        args.reject_long_nonvisual_linegrid_subline_chars > 0
        and source == "ocr_linegrid:subline_window"
        and support_meta["visual_signal"] <= 0.0
        and support_meta["numeric_signal"] > 0.0
        and len(text) >= args.reject_long_nonvisual_linegrid_subline_chars
    ):
        return False
    if support_meta["visual_signal"] <= 0.0 and len(text) > args.max_nonvisual_chars:
        return False
    if support_meta["rank_score"] < args.min_rank_score:
        return False
    if (
        support_meta["visual_signal"] <= 0.0
        and support_meta["numeric_signal"] <= 0.0
        and support_meta["query_score"] < args.min_query_score
        and support_meta["token_hit_count"] < args.min_token_hits
    ):
        return False
    return True


def choose_replacement(
    *,
    boxes: list[list[int]],
    candidate: dict[str, Any],
    redundant: dict[int, float],
    args: argparse.Namespace,
) -> tuple[int | None, str]:
    cbox = [int(v) for v in candidate.get("box") or []]
    if len(cbox) < 4:
        return None, "bad_candidate_box"
    if any(box_iou(cbox, box) >= args.max_candidate_iou_existing for box in boxes):
        return None, "candidate_duplicates_existing"

    eligible: list[tuple[float, int]] = []
    for idx, rscore in redundant.items():
        # Avoid replacing a box if the candidate would swallow multiple other
        # existing boxes; that pattern caused wide-row Thai regressions.
        contained_others = 0
        for j, old in enumerate(boxes):
            if j == idx:
                continue
            if contain_fraction(old, cbox) >= args.max_candidate_contains_existing:
                contained_others += 1
        if contained_others > args.max_contained_others:
            continue
        ox, oy = center(boxes[idx])
        cx, cy = center(cbox)
        distance = abs(ox - cx) / max(1, max(box[2] for box in boxes)) + abs(oy - cy) / max(1, max(box[3] for box in boxes))
        eligible.append((rscore + 0.05 * distance + 0.001 * idx, idx))
    if not eligible:
        return None, "no_safe_redundant_index"
    eligible.sort(reverse=True)
    return eligible[0][1], "selected"


def is_ambiguous_numeric_linegrid(
    candidate: dict[str, Any],
    support_meta: dict[str, Any],
    extras: list[dict[str, Any]],
    args: argparse.Namespace,
) -> bool:
    if not args.skip_ambiguous_numeric_linegrid:
        return False
    if str(candidate.get("source") or "") != "ocr_linegrid:subline_window":
        return False
    if support_meta["visual_signal"] > 0.0 or support_meta["numeric_signal"] <= 0.0:
        return False
    text = str(candidate.get("text") or "")
    if len(text) < args.ambiguous_linegrid_min_chars:
        return False
    cbox = [int(v) for v in candidate.get("box") or []]
    if len(cbox) < 4:
        return False
    cy = (cbox[1] + cbox[3]) / 2.0
    bands: list[float] = []
    for other in extras:
        obox = [int(v) for v in other.get("box") or []]
        if len(obox) < 4:
            continue
        source = str(other.get("source") or "")
        if not source.startswith("stage_evidence_span_id"):
            continue
        _, meta = candidate_support(other, args)
        if meta["visual_signal"] > 0.0 or meta["numeric_signal"] <= 0.0:
            continue
        if len(str(other.get("text") or "")) < args.ambiguous_linegrid_min_chars:
            continue
        oy = (obox[1] + obox[3]) / 2.0
        if all(abs(oy - seen) > args.ambiguous_linegrid_y_gap for seen in bands):
            bands.append(oy)
    far_bands = [y for y in bands if abs(y - cy) > args.ambiguous_linegrid_y_gap]
    return len(bands) >= args.ambiguous_linegrid_min_bands and len(far_bands) >= 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--eval-json", default="", help="Optional: read language_code only.")
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--ocr-layout-cache-dir", default=str(DEFAULT_OCR_LAYOUT_CACHE))
    parser.add_argument("--ocr-layout-model", default="qwen-vl-ocr")
    parser.add_argument(
        "--route-language",
        action="append",
        default=[],
        help="Language(s) to route. Defaults to th only when this option is omitted.",
    )
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel", "auto"], default="normalized-1000")
    parser.add_argument("--max-candidates", type=int, default=2500)
    parser.add_argument("--top-k", type=int, default=140)
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument("--max-area-ratio", type=float, default=0.08)
    parser.add_argument("--candidate-duplicate-iou", type=float, default=0.50)
    parser.add_argument("--candidate-min-area", type=float, default=0.0)
    parser.add_argument("--candidate-family-allowlist", default="")
    parser.add_argument("--candidate-source-allowlist", default="")
    parser.add_argument("--candidate-text-include-regex", default="")
    parser.add_argument("--duplicate-iou", type=float, default=0.78)
    parser.add_argument("--same-row-contain", type=float, default=0.55)
    parser.add_argument("--min-rank-score", type=float, default=6.5)
    parser.add_argument("--min-query-score", type=float, default=2.5)
    parser.add_argument("--min-token-hits", type=int, default=1)
    parser.add_argument("--visual-bonus", type=float, default=7.0)
    parser.add_argument("--max-nonvisual-chars", type=int, default=120)
    parser.add_argument("--max-visual-span-union-chars", type=int, default=120)
    parser.add_argument(
        "--reject-verbose-visual-evidence-chars",
        type=int,
        default=0,
        help="Reject long direct evidence candidates with visual hits; 0 disables.",
    )
    parser.add_argument("--verbose-visual-evidence-penalty-chars", type=int, default=160)
    parser.add_argument("--verbose-visual-evidence-penalty", type=float, default=4.0)
    parser.add_argument(
        "--reject-long-nonvisual-linegrid-subline-chars",
        type=int,
        default=0,
        help="Reject long numeric ocr_linegrid:subline_window candidates without visual cues; 0 disables.",
    )
    parser.add_argument("--skip-ambiguous-numeric-linegrid", action="store_true")
    parser.add_argument("--ambiguous-linegrid-min-chars", type=int, default=60)
    parser.add_argument("--ambiguous-linegrid-min-bands", type=int, default=3)
    parser.add_argument("--ambiguous-linegrid-y-gap", type=float, default=80.0)
    parser.add_argument("--max-candidate-iou-existing", type=float, default=0.35)
    parser.add_argument("--max-candidate-contains-existing", type=float, default=0.72)
    parser.add_argument("--max-contained-others", type=int, default=1)
    parser.add_argument("--stage-name", default="qwen_pipe_th_redundant_broad_replace")
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    languages = load_languages(args.eval_json)
    if not args.route_language:
        args.route_language = ["th"]
    args.candidate_family_allowlist = {
        part.strip() for part in str(args.candidate_family_allowlist or "").split(",") if part.strip()
    }
    args.candidate_source_allowlist = [
        part.strip() for part in str(args.candidate_source_allowlist or "").split(",") if part.strip()
    ]
    route_langs = set(args.route_language or [])
    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    stats = {
        "rows": 0,
        "eligible_language": 0,
        "forged_rows": 0,
        "redundant_rows": 0,
        "changed": 0,
        "skipped": {},
    }

    with out_path.open("w", encoding="utf-8") as out:
        for row in read_jsonl(resolve_pipe_path(args.input_jsonl)):
            stats["rows"] += 1
            sid = sample_id_from_row(row)
            lang = languages.get(sid) or language_code(row)
            selected: dict[str, Any] | None = None
            skip_reason = ""
            if lang not in route_langs:
                skip_reason = "language"
            elif not conclusion_is_forged(row):
                skip_reason = "not_pred_forged"
            else:
                stats["eligible_language"] += 1
                stats["forged_rows"] += 1
                boxes = report_boxes(str(row.get("raw_output") or ""))
                redundant = redundant_indices(
                    boxes,
                    duplicate_iou=args.duplicate_iou,
                    same_row_contain=args.same_row_contain,
                )
                if not boxes or not redundant:
                    skip_reason = "no_redundant_boxes"
                else:
                    stats["redundant_rows"] += 1
                    image_path = resolve_image_path(row, debug_root)
                    if not image_path:
                        skip_reason = "missing_image"
                    else:
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
                            language_code=lang,
                        )
                        top = choose_topk_v145ocranchor(row, candidates, args.top_k, width, height, language_code=lang)
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
                            boxes,
                            width=width,
                            height=height,
                            limit=args.candidate_limit,
                            max_area_ratio=args.max_area_ratio,
                            duplicate_iou=args.candidate_duplicate_iou,
                        )
                        ranked: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
                        reject_counts: dict[str, int] = {}
                        for cand in extras:
                            cbox = [int(v) for v in cand.get("box") or []]
                            if len(cbox) < 4:
                                continue
                            if args.candidate_min_area > 0 and area(cbox) < args.candidate_min_area:
                                reject_counts["small_candidate"] = reject_counts.get("small_candidate", 0) + 1
                                continue
                            if page_area_ratio(cbox, width, height) > args.max_area_ratio:
                                continue
                            support, meta = candidate_support(cand, args)
                            if not candidate_is_supported(cand, meta, args):
                                reject_counts["weak_support"] = reject_counts.get("weak_support", 0) + 1
                                continue
                            replace_index, reason = choose_replacement(
                                boxes=boxes,
                                candidate=cand,
                                redundant=redundant,
                                args=args,
                            )
                            if replace_index is None:
                                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                                continue
                            ranked.append((support + redundant.get(replace_index, 0.0), replace_index, cand, meta))
                        if not ranked:
                            skip_reason = "no_candidate"
                            selected = {"reject_counts": reject_counts, "redundant": redundant}
                        else:
                            ranked.sort(key=lambda item: item[0], reverse=True)
                            _score, replace_index, cand, meta = ranked[0]
                            if is_ambiguous_numeric_linegrid(cand, meta, extras, args):
                                skip_reason = "ambiguous_numeric_linegrid"
                                selected = {
                                    "reject_counts": reject_counts,
                                    "redundant": redundant,
                                    "candidate": {
                                        "label": cand.get("label"),
                                        "box": [int(v) for v in cand.get("box") or []][:4],
                                        "family": cand.get("family"),
                                        "source": cand.get("source"),
                                        "text": str(cand.get("text") or "")[:180],
                                        "support_meta": meta,
                                    },
                                    "policy": "Skipped redundant-box replacement because a nonvisual numeric linegrid candidate is ambiguous across repeated evidence rows.",
                                }
                                stats["skipped"][skip_reason] = int(stats["skipped"].get(skip_reason, 0)) + 1
                                summary_rows.append(
                                    {
                                        "sample_id": sid,
                                        "language_code": lang,
                                        "skip_reason": skip_reason,
                                        "selected": selected,
                                    }
                                )
                                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                                continue
                            cbox = [int(v) for v in cand.get("box") or []][:4]
                            new_report, replaced = replace_groundings(str(row.get("raw_output") or ""), {replace_index: cbox})
                            if replaced:
                                row = dict(row)
                                row["raw_output"] = new_report
                                row["parsed"] = parse_cct_report(new_report)
                                selected = {
                                    "replace_index": replace_index,
                                    "candidate": {
                                        "label": cand.get("label"),
                                        "box": cbox,
                                        "family": cand.get("family"),
                                        "source": cand.get("source"),
                                        "text": str(cand.get("text") or "")[:180],
                                        "support_meta": meta,
                                    },
                                    "redundant": redundant,
                                    "reject_counts": reject_counts,
                                    "policy": "GT-free Thai broad candidate replacement only when existing grounding boxes are redundant.",
                                }
                                stage_outputs = dict(row.get("stage_outputs") or {})
                                stage_outputs[args.stage_name] = {"applied": True, **selected}
                                row["stage_outputs"] = stage_outputs
                                stats["changed"] += 1
                            else:
                                skip_reason = "replace_failed"
            if skip_reason:
                stats["skipped"][skip_reason] = int(stats["skipped"].get(skip_reason, 0)) + 1
            if selected or (skip_reason and lang in route_langs):
                summary_rows.append(
                    {
                        "sample_id": sid,
                        "language_code": lang,
                        "skip_reason": skip_reason,
                        "selected": selected,
                    }
                )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    params = dict(vars(args))
    params["candidate_family_allowlist"] = sorted(args.candidate_family_allowlist)
    params["candidate_source_allowlist"] = list(args.candidate_source_allowlist)
    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "output_jsonl": str(out_path),
        "params": params,
        "stats": stats,
        "rows": summary_rows,
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
