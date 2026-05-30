#!/usr/bin/env python3
"""GT-free URL/source-row localization replacement.

Some news/document samples expose tampering in source/header rows such as
social URLs.  The generic visual-evidence replacement intentionally ignores
nonvisual row candidates, so this narrow router only considers top-page OCR rows
that contain URL/social handles and replaces a weak generated extra grounding.
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
    RecallCandidate,
    box_iou,
    generate_candidates,
    page_area_ratio,
    read_jsonl,
    sample_id_from_row,
)
from qwen_pair_replace_model import conclusion_is_forged, language_code  # noqa: E402
from qwen_text_crop_verify import area, resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


URL_RE = re.compile(r"(https?://|www\.|twitter\.com|facebook\.com|instagram\.com|@[A-Za-z0-9_]+|[A-Za-z0-9_-]+\.(?:com|org|net))", re.I)
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[[^\]]+\]", re.I)
ANOMALY_RE = re.compile(r"^###\s+ANOMALY[^\n]*", re.I | re.M)


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
    if not matches:
        return []
    blocks: list[str] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        block = report[match.start() : end]
        if GROUNDING_RE.search(block):
            blocks.append(block)
    return blocks


def choose_replace_index(report: str, boxes: list[list[int]]) -> int | None:
    blocks = report_blocks(report)
    for idx, block in enumerate(blocks):
        lower = block.lower()
        if "anomaly_v87_extra" in lower and ("(token)" in lower or "from token" in lower):
            return idx
    for idx, block in enumerate(blocks):
        if "anomaly_v87_extra" in block.lower():
            return idx
    if boxes:
        return min(range(len(boxes)), key=lambda i: area(boxes[i]))
    return None


def candidate_score(candidate: RecallCandidate, width: int, height: int) -> float:
    box = [int(v) for v in candidate.box[:4]]
    text = str(candidate.text or "")
    y_center = (box[1] + box[3]) / 2.0 / max(1, height)
    url_hits = len(URL_RE.findall(text))
    score = 10.0 * url_hits
    score += 2.0 if candidate.family == "row" else 0.5
    score += 1.5 if "twitter.com" in text.lower() else 0.0
    score += max(0.0, 2.0 - 12.0 * y_center)
    score -= 20.0 * page_area_ratio(box, width, height)
    return score


def find_url_candidate(
    *,
    row: dict[str, Any],
    image: Image.Image,
    debug_root: Path,
    ocr_layout_cache_dir: Path,
    ocr_layout_model: str,
    coord_mode: str,
    max_candidates: int,
    width: int,
    height: int,
    existing_boxes: list[list[int]],
    language_code: str,
    max_top_y: float,
    max_area_ratio: float,
    candidate_family_allowlist: set[str],
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
        language_code=language_code,
    )
    filtered: list[RecallCandidate] = []
    for cand in candidates:
        if candidate_family_allowlist and cand.family not in candidate_family_allowlist:
            continue
        if not candidate_family_allowlist and cand.family not in {"row", "ocr"}:
            continue
        text = str(cand.text or "")
        if not URL_RE.search(text):
            continue
        box = [int(v) for v in cand.box[:4]]
        if len(box) < 4:
            continue
        if (box[1] + box[3]) / 2.0 > max_top_y * height:
            continue
        if page_area_ratio(box, width, height) > max_area_ratio:
            continue
        if any(box_iou(box, old) >= 0.25 for old in existing_boxes):
            continue
        filtered.append(cand)
    if not filtered:
        return None
    filtered.sort(key=lambda cand: candidate_score(cand, width, height), reverse=True)
    return filtered[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--eval-json", default="", help="Optional: read language_code only.")
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--ocr-layout-cache-dir", default=str(DEFAULT_OCR_LAYOUT_CACHE))
    parser.add_argument("--ocr-layout-model", default="qwen-vl-ocr")
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel", "auto"], default="normalized-1000")
    parser.add_argument("--max-candidates", type=int, default=2500)
    parser.add_argument("--route-language", action="append", default=[])
    parser.add_argument("--max-top-y", type=float, default=0.18)
    parser.add_argument("--max-area-ratio", type=float, default=0.012)
    parser.add_argument("--candidate-family-allowlist", default="")
    parser.add_argument("--stage-name", default="qwen_pipe_ar_url_source_row_replace")
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    languages = load_languages(args.eval_json)
    if not args.route_language:
        args.route_language = ["ar"]
    route_langs = {str(lang).strip().lower() for lang in args.route_language if str(lang).strip()}
    candidate_family_allowlist = {
        part.strip() for part in str(args.candidate_family_allowlist or "").split(",") if part.strip()
    }
    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"rows": 0, "eligible_ar_forged": 0, "changed": 0, "skipped": {}}
    summary_rows: list[dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as out:
        for row in read_jsonl(resolve_pipe_path(args.input_jsonl)):
            stats["rows"] += 1
            sid = sample_id_from_row(row)
            lang = languages.get(sid) or language_code(row)
            skip_reason = ""
            selected: dict[str, Any] | None = None
            if lang not in route_langs:
                skip_reason = "language"
            elif not conclusion_is_forged(row):
                skip_reason = "not_pred_forged"
            else:
                stats["eligible_ar_forged"] += 1
                report = str(row.get("raw_output") or "")
                boxes = report_boxes(report)
                replace_index = choose_replace_index(report, boxes)
                image_path = resolve_image_path(row, debug_root)
                if not boxes or replace_index is None:
                    skip_reason = "no_replace_slot"
                elif not image_path:
                    skip_reason = "missing_image"
                else:
                    with Image.open(image_path) as im:
                        image = im.convert("RGB")
                    width, height = image.size
                    cand = find_url_candidate(
                        row=row,
                        image=image,
                        debug_root=debug_root,
                        ocr_layout_cache_dir=Path(args.ocr_layout_cache_dir).expanduser().resolve(),
                        ocr_layout_model=args.ocr_layout_model,
                        coord_mode=args.coord_mode,
                        max_candidates=args.max_candidates,
                        width=width,
                        height=height,
                        existing_boxes=boxes,
                        language_code=lang,
                        max_top_y=args.max_top_y,
                        max_area_ratio=args.max_area_ratio,
                        candidate_family_allowlist=candidate_family_allowlist,
                    )
                    if not cand:
                        skip_reason = "no_url_candidate"
                    else:
                        cbox = [int(v) for v in cand.box[:4]]
                        new_report, replaced = replace_groundings(report, {replace_index: cbox})
                        if replaced:
                            row = dict(row)
                            row["raw_output"] = new_report
                            row["parsed"] = parse_cct_report(new_report)
                            selected = {
                                "replace_index": replace_index,
                                "candidate": {
                                    "box": cbox,
                                    "family": cand.family,
                                    "source": cand.source,
                                    "text": str(cand.text or "")[:180],
                                    "score": candidate_score(cand, width, height),
                                },
                            "policy": "GT-free top-page URL/source-row replacement.",
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

    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "output_jsonl": str(out_path),
        "params": vars(args),
        "stats": stats,
        "rows": summary_rows,
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
