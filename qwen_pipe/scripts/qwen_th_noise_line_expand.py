#!/usr/bin/env python3
"""GT-free Thai noise-artifact line expansion.

Some Thai failures localize only a tiny black/noise artifact even though the
report text describes a corrupted paragraph/line.  This DocShield-style visual
cue grounding pass expands one tiny Thai noise box to the OCR text line at the
same vertical position, while preserving the existing verdict and report text.
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
from qwen_exhaustive_recall import (  # noqa: E402
    RecallCandidate,
    generate_candidates,
    read_jsonl,
    sample_id_from_row,
)
from qwen_pair_replace_model import conclusion_is_forged  # noqa: E402
from qwen_text_crop_verify import resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


ANOMALY_RE = re.compile(r"^###\s+ANOMALY[^\n]*", re.I | re.M)
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[[^\]]+\]", re.I)
NOISE_RE = re.compile(r"(จุดดำ|จุดสีดำ|noise artifact|visual noise|สิ่งแปลกปลอม)", re.I)


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


def is_tiny_noise_block(block: str, box: list[int], max_area: int, max_width: int, max_height: int) -> bool:
    if not NOISE_RE.search(block or ""):
        return False
    return area(box) <= max_area and (box[2] - box[0]) <= max_width and (box[3] - box[1]) <= max_height


def y_overlap(a: list[int], b: list[int]) -> float:
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def find_line_candidate(
    *,
    row: dict[str, Any],
    image: Image.Image,
    debug_root: Path,
    ocr_layout_cache_dir: Path,
    ocr_layout_model: str,
    coord_mode: str,
    anchor_box: list[int],
    existing_boxes: list[list[int]],
    max_candidates: int,
    min_width: int,
    max_height: int,
    max_width_ratio: float,
) -> RecallCandidate | None:
    candidates = generate_candidates(
        row,
        image,
        debug_root,
        ocr_layout_cache_dir,
        ocr_layout_model,
        coord_mode,
        max_candidates,
        enable_token_candidates=False,
        enable_linegrid_candidates=False,
        enable_scriptgrid_candidates=False,
        language_code="th",
    )
    valid: list[tuple[float, RecallCandidate]] = []
    ax = (anchor_box[0] + anchor_box[2]) / 2.0
    ay = (anchor_box[1] + anchor_box[3]) / 2.0
    for cand in candidates:
        box = [int(v) for v in cand.box[:4]]
        if len(box) != 4:
            continue
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width < min_width or height <= 0 or height > max_height:
            continue
        if width / max(1, image.size[0]) > max_width_ratio:
            continue
        if cand.family not in {"ocr", "row"}:
            continue
        if "qwen_ocr" not in cand.source and "ocr_row" not in cand.source:
            continue
        if y_overlap(anchor_box, box) < 0.35:
            continue
        if any(iou(box, old) >= 0.45 for old in existing_boxes):
            continue
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        score = width - 1.5 * abs(cy - ay) - 0.1 * abs(cx - ax)
        if cand.family == "row":
            score += 80.0
        if "expanded" in cand.source:
            score -= 120.0
        valid.append((score, cand))
    if not valid:
        return None
    valid.sort(key=lambda item: item[0], reverse=True)
    return valid[0][1]


def choose_replace_index(blocks: list[str], boxes: list[list[int]], anchor_idx: int) -> int:
    for idx, block in enumerate(blocks[: len(boxes)]):
        if idx == anchor_idx:
            continue
        lower = block.lower()
        if "anomaly_v87_extra" in lower and "evidence" not in lower:
            return idx
    for idx, block in enumerate(blocks[: len(boxes)]):
        if idx != anchor_idx and "anomaly_v87_extra" in block.lower():
            return idx
    return anchor_idx


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
    parser.add_argument("--max-candidates", type=int, default=2500)
    parser.add_argument("--max-anchor-area", type=int, default=4000)
    parser.add_argument("--max-anchor-width", type=int, default=150)
    parser.add_argument("--max-anchor-height", type=int, default=60)
    parser.add_argument("--min-line-width", type=int, default=500)
    parser.add_argument("--max-line-height", type=int, default=90)
    parser.add_argument("--max-line-width-ratio", type=float, default=0.92)
    parser.add_argument("--stage-name", default="qwen_pipe_th_noise_line_expand")
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    languages = load_languages(args.eval_json)
    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"rows": 0, "eligible": 0, "changed": 0, "skipped": {}}
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
            if lang != "th":
                skip = "language"
            elif not conclusion_is_forged(row):
                skip = "not_pred_forged"
            else:
                stats["eligible"] += 1
                report = str(row.get("raw_output") or "")
                boxes = report_boxes(report)
                blocks = report_blocks(report)
                anchors = [
                    (idx, boxes[idx])
                    for idx in range(min(len(blocks), len(boxes)))
                    if is_tiny_noise_block(
                        blocks[idx],
                        boxes[idx],
                        args.max_anchor_area,
                        args.max_anchor_width,
                        args.max_anchor_height,
                    )
                ]
                if not anchors:
                    skip = "no_tiny_noise_anchor"
                else:
                    image_path = resolve_image_path(row, debug_root)
                    if not image_path:
                        skip = "missing_image"
                    else:
                        with Image.open(image_path) as im:
                            image = im.convert("RGB")
                        best: tuple[float, int, list[int], RecallCandidate] | None = None
                        for idx, anchor in anchors:
                            cand = find_line_candidate(
                                row=row,
                                image=image,
                                debug_root=debug_root,
                                ocr_layout_cache_dir=Path(args.ocr_layout_cache_dir).expanduser().resolve(),
                                ocr_layout_model=args.ocr_layout_model,
                                coord_mode=args.coord_mode,
                                anchor_box=anchor,
                                existing_boxes=boxes,
                                max_candidates=args.max_candidates,
                                min_width=args.min_line_width,
                                max_height=args.max_line_height,
                                max_width_ratio=args.max_line_width_ratio,
                            )
                            if not cand:
                                continue
                            cbox = [int(v) for v in cand.box[:4]]
                            score = (cbox[2] - cbox[0]) - abs(((cbox[1] + cbox[3]) - (anchor[1] + anchor[3])) / 2.0)
                            if best is None or score > best[0]:
                                best = (score, idx, anchor, cand)
                        if best is None:
                            skip = "no_line_candidate"
                        else:
                            _score, anchor_idx, anchor_box, cand = best
                            replace_index = choose_replace_index(blocks, boxes, anchor_idx)
                            cbox = [int(v) for v in cand.box[:4]]
                            new_report, replaced = replace_groundings(report, {replace_index: cbox})
                            if not replaced:
                                skip = "replace_failed"
                            else:
                                row = dict(row)
                                row["raw_output"] = new_report
                                row["parsed"] = parse_cct_report(new_report)
                                stage_outputs = dict(row.get("stage_outputs") or {})
                                stage_outputs[args.stage_name] = {
                                    "applied": True,
                                    "anchor_index": anchor_idx,
                                    "replace_index": replace_index,
                                    "anchor_box": anchor_box,
                                    "candidate": {
                                        "box": cbox,
                                        "text": str(cand.text or "")[:180],
                                        "source": cand.source,
                                        "family": cand.family,
                                    },
                                    "policy": "GT-free Thai tiny-noise artifact expansion to same OCR line.",
                                }
                                row["stage_outputs"] = stage_outputs
                                changes.append(
                                    {
                                        "sample_id": sid,
                                        "anchor_index": anchor_idx,
                                        "replace_index": replace_index,
                                        "old_box": boxes[replace_index],
                                        "anchor_box": anchor_box,
                                        "new_box": cbox,
                                        "candidate_text": str(cand.text or "")[:180],
                                        "candidate_source": cand.source,
                                    }
                                )
                                stats["changed"] += 1
            if skip:
                stats["skipped"][skip] = int(stats["skipped"].get(skip, 0)) + 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {**stats, "changes": changes, "output_jsonl": str(out_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
