#!/usr/bin/env python3
"""GT-free Thai evidence/linegrid grounding route.

This pass converts two diagnostic Thai localization lessons into narrow
inference-time rules:

* high-confidence edge/redaction evidence can replace a weak broad primary box;
* when a Thai report points to a line-number/digit rendering defect, a nearby
  previous OCR linegrid crop that contains the same numeric token can replace a
  redundant same-line extra box.

No GT labels, masks, reports, or loc scores are read.  The optional eval JSON is
used only to get language_code.
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
from qwen_evidence_multibox_refine import area, iou, report_boxes  # noqa: E402
from qwen_exhaustive_recall import generate_candidates, sample_id_from_row  # noqa: E402
from qwen_issue_refine import evidence_candidates, parse_box, project_box  # noqa: E402
from qwen_pair_replace_model import conclusion_is_forged  # noqa: E402
from qwen_text_crop_verify import resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


ANOMALY_RE = re.compile(r"^###\s+ANOMALY[^\n]*", re.I | re.M)
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[[^\]]+\]", re.I)
EDGE_RE = re.compile(
    r"(edge_artifact|redaction|obscur|solid (?:black|grey|gray)|grey bar|gray bar|"
    r"black pixels|ปิดบัง|แถบสีเทา|สีเทาทึบ|กล่องสี่เหลี่ยม)",
    re.I,
)
LINE_DEFECT_RE = re.compile(r"(บรรทัดที่|ตัวเลข|glitch|rendering|เรนเดอร์|ลวดลาย|รอยแตก)", re.I)
THAI_DIGIT_RE = re.compile(r"[๐-๙]{2,}")


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


def report_blocks(report: str) -> list[str]:
    matches = list(ANOMALY_RE.finditer(report or ""))
    blocks: list[str] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        block = report[match.start() : end]
        if GROUNDING_RE.search(block):
            blocks.append(block)
    return blocks


def candidate_text(cand: dict[str, Any]) -> str:
    return " ".join(str(cand.get(key) or "") for key in ("category", "evidence", "notes", "reason"))


def y_gap_above(candidate_box: list[int], anchor_box: list[int]) -> int:
    return anchor_box[1] - candidate_box[3]


def x_overlap_ratio(a: list[int], b: list[int]) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return overlap / max(1, min(a[2] - a[0], b[2] - b[0]))


def best_edge_candidate(
    row: dict[str, Any],
    *,
    min_confidence: float,
    max_area_ratio: float,
    duplicate_iou: float,
    existing_boxes: list[list[int]],
) -> tuple[dict[str, Any], list[int], float] | None:
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    page_area = max(1, width * height)
    scored: list[tuple[float, dict[str, Any], list[int]]] = []
    for cand in evidence_candidates(row):
        text = candidate_text(cand)
        category = str(cand.get("category") or "").lower()
        if category != "edge_artifact" and not EDGE_RE.search(text):
            continue
        if category not in {"edge_artifact", "copy_paste_boundary", "rendering_artifact"}:
            continue
        try:
            confidence = float(cand.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            continue
        raw_box = parse_box(cand.get("bbox"))
        if not raw_box:
            continue
        box = project_box(raw_box, width, height, "normalized-1000")
        if not box:
            continue
        if area(box) / page_area > max_area_ratio:
            continue
        if any(iou(box, old) >= duplicate_iou for old in existing_boxes):
            continue
        nonprimary_support = max((iou(box, old) for old in existing_boxes[1:]), default=0.0)
        if category == "rendering_artifact" and nonprimary_support < 0.08:
            # Plain rendering-artifact redaction descriptions are often already
            # localized by the primary anomaly; stage evidence coordinates can
            # be vertically shifted.  Require an independent existing extra box
            # near the same cell before replacing the primary slot.
            continue
        edge_bonus = 0.25 if category == "edge_artifact" else 0.0
        support_bonus = min(0.12, nonprimary_support)
        redaction_bonus = 0.15 if EDGE_RE.search(text) else 0.0
        compact_bonus = min(0.15, 0.15 * (1.0 - area(box) / max(1, page_area * max_area_ratio)))
        scored.append((confidence + edge_bonus + redaction_bonus + compact_bonus + support_bonus, cand, box))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    score, cand, box = scored[0]
    return cand, box, score


def choose_edge_replace_index(blocks: list[str], boxes: list[list[int]], edge_box: list[int]) -> int | None:
    if not boxes:
        return None
    primary = boxes[0]
    primary_text = blocks[0].lower() if blocks else ""
    primary_far = iou(edge_box, primary) < 0.05
    primary_is_visual = any(term in primary_text for term in ("visual", "render", "เรนเดอร์", "ผิดปกติ"))
    if primary_far and primary_is_visual:
        return 0
    for idx, block in enumerate(blocks[: len(boxes)]):
        lower = block.lower()
        if "anomaly_v87_extra" in lower and "evidence" not in lower:
            return idx
    return 0 if primary_far else None


def report_core_thai_numbers(report: str) -> set[str]:
    cleaned = re.sub(r"\[GROUNDING\]\s*:\s*\[[^\]]+\]", " ", report or "", flags=re.I)
    out = set(THAI_DIGIT_RE.findall(cleaned))
    # Parenthesized line ordinals are useful routing signals but weak targets.
    for ordinal in re.findall(r"\(([๐-๙]{2,})\)", cleaned):
        if len(out) > 1:
            out.discard(ordinal)
    return out


def find_previous_digit_linegrid(
    *,
    row: dict[str, Any],
    image: Image.Image,
    debug_root: Path,
    ocr_layout_cache_dir: Path,
    ocr_layout_model: str,
    coord_mode: str,
    anchor_box: list[int],
    existing_boxes: list[list[int]],
    core_numbers: set[str],
    max_candidates: int,
    min_width: int,
    max_width: int,
    max_height: int,
    max_gap: int,
    duplicate_iou: float,
) -> Any | None:
    candidates = generate_candidates(
        row,
        image,
        debug_root,
        ocr_layout_cache_dir,
        ocr_layout_model,
        coord_mode,
        max_candidates,
        enable_token_candidates=True,
        enable_linegrid_candidates=True,
        enable_scriptgrid_candidates=True,
        language_code="th",
    )
    scored: list[tuple[float, Any]] = []
    for cand in candidates:
        if cand.family != "linegrid":
            continue
        if not (
            str(cand.source).startswith("ocr_scriptgrid:physical_window")
            or str(cand.source).startswith("ocr_linegrid:subline_window")
        ):
            continue
        box = [int(v) for v in cand.box[:4]]
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width < min_width or width > max_width or height <= 0 or height > max_height:
            continue
        gap = y_gap_above(box, anchor_box)
        if gap < -8 or gap > max_gap:
            continue
        if x_overlap_ratio(box, anchor_box) < 0.25:
            continue
        if any(iou(box, old) >= duplicate_iou for old in existing_boxes):
            continue
        text = str(cand.text or "")
        if core_numbers and not any(num in text for num in core_numbers):
            continue
        meta = cand.meta or {}
        try:
            match_score = float(meta.get("match_score") or 0.0)
        except (TypeError, ValueError):
            match_score = 0.0
        # Prefer the closest previous line.  Match score prevents arbitrary
        # neighboring rows, but closeness is the key correction for this subtype.
        score = 4.0 * max(0, max_gap - abs(gap)) + match_score + min(width, max_width) / 100.0
        scored.append((score, cand))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def choose_digit_replace_index(blocks: list[str], boxes: list[list[int]], anchor_idx: int) -> int | None:
    candidates: list[tuple[int, int]] = []
    for idx, block in enumerate(blocks[: len(boxes)]):
        if idx == anchor_idx:
            continue
        lower = block.lower()
        if "anomaly_v87_extra" in lower:
            # Replace the right-side duplicate first; keep the left-side token
            # because it usually anchors the reported number text.
            candidates.append((boxes[idx][0], idx))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def apply_replacement(row: dict[str, Any], replace_index: int, new_box: list[int], stage_payload: dict[str, Any]) -> dict[str, Any] | None:
    debug_root = Path(stage_payload.pop("_debug_root"))
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    report = str(row.get("raw_output") or "")
    new_report, replaced = replace_groundings(report, {replace_index: new_box})
    if not replaced:
        return None
    out = dict(row)
    out["raw_output"] = new_report
    out["parsed"] = parse_cct_report(new_report)
    stage_outputs = dict(out.get("stage_outputs") or {})
    stage_outputs[str(stage_payload.pop("_stage_name"))] = stage_payload
    out["stage_outputs"] = stage_outputs
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--ocr-layout-cache-dir", default=str(DEFAULT_OCR_LAYOUT_CACHE))
    parser.add_argument("--ocr-layout-model", default="qwen-vl-ocr")
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel", "auto"], default="normalized-1000")
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--edge-min-confidence", type=float, default=0.84)
    parser.add_argument("--edge-max-area-ratio", type=float, default=0.06)
    parser.add_argument("--duplicate-iou", type=float, default=0.20)
    parser.add_argument("--digit-min-width", type=int, default=120)
    parser.add_argument("--digit-max-width", type=int, default=320)
    parser.add_argument("--digit-max-height", type=int, default=72)
    parser.add_argument("--digit-max-gap", type=int, default=90)
    parser.add_argument("--stage-name", default="qwen_pipe_th_stage_evidence_route")
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    languages = load_languages(args.eval_json)
    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"rows": 0, "eligible": 0, "changed": 0, "by_action": {}, "skipped": {}}
    changes: list[dict[str, Any]] = []

    with resolve_pipe_path(args.input_jsonl).open("r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            sid = sample_id_from_row(row)
            lang = (languages.get(sid) or "").lower()
            skip = ""
            changed_row: dict[str, Any] | None = None
            if lang != "th":
                skip = "language"
            elif not conclusion_is_forged(row):
                skip = "not_pred_forged"
            else:
                stats["eligible"] += 1
                report = str(row.get("raw_output") or "")
                boxes = report_boxes(report)
                blocks = report_blocks(report)
                edge = best_edge_candidate(
                    row,
                    min_confidence=args.edge_min_confidence,
                    max_area_ratio=args.edge_max_area_ratio,
                    duplicate_iou=args.duplicate_iou,
                    existing_boxes=boxes,
                )
                if edge:
                    cand, edge_box, score = edge
                    replace_index = choose_edge_replace_index(blocks, boxes, edge_box)
                    if replace_index is not None:
                        payload = {
                            "_debug_root": str(debug_root),
                            "_stage_name": args.stage_name,
                            "applied": True,
                            "action": "edge_artifact_replace",
                            "replace_index": replace_index,
                            "old_box": boxes[replace_index] if replace_index < len(boxes) else None,
                            "new_box": edge_box,
                            "candidate": {
                                "id": cand.get("id"),
                                "category": cand.get("category"),
                                "confidence": cand.get("confidence"),
                                "score": score,
                                "evidence": str(cand.get("evidence") or "")[:240],
                            },
                            "policy": "GT-free Thai high-confidence edge/redaction evidence replaces a weak/far grounding slot.",
                        }
                        changed_row = apply_replacement(row, replace_index, edge_box, payload)
                if changed_row is None and LINE_DEFECT_RE.search(report):
                    core_numbers = report_core_thai_numbers(report)
                    anchor_idx = 0 if boxes else -1
                    if boxes and core_numbers:
                        image_path = resolve_image_path(row, debug_root)
                        if image_path:
                            with Image.open(image_path) as im:
                                image = im.convert("RGB")
                            cand = find_previous_digit_linegrid(
                                row=row,
                                image=image,
                                debug_root=debug_root,
                                ocr_layout_cache_dir=Path(args.ocr_layout_cache_dir).expanduser().resolve(),
                                ocr_layout_model=args.ocr_layout_model,
                                coord_mode=args.coord_mode,
                                anchor_box=boxes[anchor_idx],
                                existing_boxes=boxes,
                                core_numbers=core_numbers,
                                max_candidates=args.max_candidates,
                                min_width=args.digit_min_width,
                                max_width=args.digit_max_width,
                                max_height=args.digit_max_height,
                                max_gap=args.digit_max_gap,
                                duplicate_iou=args.duplicate_iou,
                            )
                            replace_index = choose_digit_replace_index(blocks, boxes, anchor_idx) if cand else None
                            if cand and replace_index is not None:
                                new_box = [int(v) for v in cand.box[:4]]
                                payload = {
                                    "_debug_root": str(debug_root),
                                    "_stage_name": args.stage_name,
                                    "applied": True,
                                    "action": "previous_digit_linegrid_replace",
                                    "anchor_index": anchor_idx,
                                    "replace_index": replace_index,
                                    "old_box": boxes[replace_index],
                                    "new_box": new_box,
                                    "core_numbers": sorted(core_numbers),
                                    "candidate": {
                                        "family": cand.family,
                                        "source": cand.source,
                                        "text": str(cand.text or "")[:220],
                                        "meta": cand.meta or {},
                                    },
                                    "policy": "GT-free Thai line-number/digit defect route replaces a redundant same-line extra with a nearby previous-line linegrid candidate.",
                                }
                                changed_row = apply_replacement(row, replace_index, new_box, payload)
                if changed_row is not None:
                    row = changed_row
                    stage = row.get("stage_outputs", {}).get(args.stage_name, {})
                    action = str(stage.get("action") or "unknown")
                    stats["changed"] += 1
                    stats["by_action"][action] = int(stats["by_action"].get(action, 0)) + 1
                    changes.append(
                        {
                            "sample_id": sid,
                            "action": action,
                            "replace_index": stage.get("replace_index"),
                            "old_box": stage.get("old_box"),
                            "new_box": stage.get("new_box"),
                            "candidate": stage.get("candidate"),
                        }
                    )
                else:
                    skip = skip or "no_route"
            if skip:
                stats["skipped"][skip] = int(stats["skipped"].get(skip, 0)) + 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {**stats, "changes": changes, "output_jsonl": str(out_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
