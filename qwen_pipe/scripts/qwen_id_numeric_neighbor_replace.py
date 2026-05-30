#!/usr/bin/env python3
"""GT-free Indonesian numeric-neighbor localization replacement.

The v290 diagnostic showed that several Indonesian failures can be improved by
grounding nearby page/number spans instead of only the representative anomaly
returned by the VLM.  This script does not read GT labels or donor selections:
it uses the current report text plus cached OCR action candidates to find short
numeric spans mentioned in the report and adjacent to an already localized
numeric/page anomaly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_candidate_delta_model import replace_groundings  # noqa: E402
from qwen_evidence_multibox_refine import iou, report_boxes  # noqa: E402
from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_pair_replace_model import conclusion_is_forged  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


ANOMALY_RE = re.compile(r"^###\s+ANOMALY[^\n]*", re.I | re.M)
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[[^\]]+\]", re.I)
NUM_RE = re.compile(r"\b\d{1,3}(?:/\d{1,3})?\b")
NUMERIC_TRIGGER_RE = re.compile(
    r"(\b angka\s+halaman|nomor\s+halaman|page\s+number|halaman\s*['\"]?\d|"
    r"kualitas\s+render\s+\b angka\b|ketidakseragaman\s+kualitas\s+render\s+\b angka\b|"
    r"blur[^\n]{0,80}\b angka\b|buram[^\n]{0,80}\b angka\b|"
    r"kontras[^\n]{0,80}\b angka\b|baris[^\n]{0,80}\b angka\b)",
    re.I,
)


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


def load_observations(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(resolve_pipe_path(path).read_text(encoding="utf-8"))
    by_sid: dict[str, list[dict[str, Any]]] = {}
    for obs in data.get("observations") or []:
        sid = str(obs.get("sample_id") or "")
        if sid:
            by_sid.setdefault(sid, []).append(obs)
    return by_sid


def report_blocks(report: str) -> list[str]:
    matches = list(ANOMALY_RE.finditer(report or ""))
    blocks: list[str] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        block = report[match.start() : end]
        if GROUNDING_RE.search(block):
            blocks.append(block)
    return blocks


def extract_numbers(text: str) -> set[str]:
    out: set[str] = set()
    for match in NUM_RE.finditer(text or ""):
        token = match.group(0)
        out.add(token)
        if "/" in token:
            out.update(part for part in token.split("/") if part)
    return out


def core_report_numbers(numbers: set[str]) -> set[str]:
    """Keep numbers that can anchor a document value sequence, not boilerplate."""
    out: set[str] = set()
    for num in numbers:
        if num in {"1", "2", "75", "85", "95"}:
            continue
        if "/" in num:
            out.add(num)
            continue
        try:
            value = int(num)
        except ValueError:
            continue
        if value >= 10:
            out.add(num)
    return out


def semantic_block_text(block: str) -> str:
    """Remove coordinate-only grounding lines before semantic matching."""
    return GROUNDING_RE.sub(" ", block or "")


def box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def center(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def block_is_numeric(block: str) -> bool:
    return bool(NUMERIC_TRIGGER_RE.search(semantic_block_text(block)))


def numeric_anchor_indices(blocks: list[str], boxes: list[list[int]]) -> list[int]:
    return [idx for idx, block in enumerate(blocks[: len(boxes)]) if block_is_numeric(block)]


def choose_replace_index(blocks: list[str], boxes: list[list[int]], numeric_indices: list[int]) -> int | None:
    numeric_set = set(numeric_indices)
    for idx, block in enumerate(blocks[: len(boxes)]):
        lower = block.lower()
        if idx not in numeric_set and "anomaly_v87_extra" not in lower:
            return idx
    for idx, block in enumerate(blocks[: len(boxes)]):
        if "anomaly_v87_extra" in block.lower():
            return idx
    if boxes:
        return min(range(len(boxes)), key=lambda i: box_area(boxes[i]))
    return None


def rank_for(obs: dict[str, Any]) -> dict[str, Any]:
    cand = obs.get("candidate") or {}
    meta = cand.get("meta") or {}
    return meta.get("v145ocranchor_rank") or meta.get("v87b_rank") or meta.get("v87_rank") or {}


def obs_numeric_tokens(obs: dict[str, Any]) -> set[str]:
    cand = obs.get("candidate") or {}
    text = str(cand.get("text") or "")
    return extract_numbers(text)


def candidate_score(
    obs: dict[str, Any],
    *,
    report_numbers: set[str],
    anchor_boxes: list[list[int]],
    existing_boxes: list[list[int]],
    max_x_distance: float,
    max_y_distance: float,
) -> tuple[float, str]:
    cand = obs.get("candidate") or {}
    family = str(cand.get("family") or "")
    source = str(cand.get("source") or "")
    box = [int(v) for v in (cand.get("box") or [])[:4]]
    text = str(cand.get("text") or "")
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        return -1e9, "bad_box"
    if str(obs.get("action") or "") != "replace":
        return -1e9, "not_replace"
    if family != "ocr":
        return -1e9, "not_ocr"
    if "qwen_ocr" not in source and "number" not in source:
        return -1e9, "source"
    nums = obs_numeric_tokens(obs)
    if not nums:
        return -1e9, "no_number"
    if not (nums & report_numbers):
        return -1e9, "number_not_in_report"
    if len(re.sub(r"[\d/\\s.,:;()\\-]", "", text)) > 14:
        return -1e9, "too_textual"
    if any(iou(box, old) >= 0.30 for old in existing_boxes):
        return -1e9, "duplicate"
    cx, cy = center(box)
    nearest_x = min(abs(cx - center(anchor)[0]) for anchor in anchor_boxes)
    nearest_y = min(abs(cy - center(anchor)[1]) for anchor in anchor_boxes)
    if nearest_x > max_x_distance:
        return -1e9, "x_far"
    if nearest_y > max_y_distance:
        return -1e9, "y_far"
    rank = rank_for(obs)
    query = float(rank.get("query_score") or 0.0)
    anchor = float(rank.get("score") or 0.0)
    short_bonus = 3.0 if len(text.strip()) <= 12 else 0.0
    slash_bonus = 1.5 if "/" in text else 0.0
    score = anchor + 1.5 * query + short_bonus + slash_bonus - 0.02 * nearest_y - 0.01 * nearest_x
    return score, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--eval-json", default="", help="Optional: read language_code only.")
    parser.add_argument(
        "--observation-cache",
        default="outputs/diagnostics/qwen_pipe_v223_pair_v145_on_v218_observations.json",
    )
    parser.add_argument("--route-language", action="append", default=[])
    parser.add_argument("--max-x-distance", type=float, default=70.0)
    parser.add_argument("--max-y-distance", type=float, default=260.0)
    parser.add_argument("--min-core-report-numbers", type=int, default=2)
    parser.add_argument("--stage-name", default="qwen_pipe_id_numeric_neighbor_replace")
    parser.add_argument("--debug-root", default=str(PIPE_ROOT.parent / "debug_distribution"))
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    languages = load_languages(args.eval_json)
    observations = load_observations(args.observation_cache)
    route_langs = {lang.lower() for lang in (args.route_language or ["id"])}

    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "rows": 0,
        "eligible": 0,
        "changed": 0,
        "skipped": {},
        "route_language": sorted(route_langs),
    }
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
            if lang not in route_langs:
                skip = "language"
            elif not conclusion_is_forged(row):
                skip = "not_pred_forged"
            else:
                stats["eligible"] += 1
                report = str(row.get("raw_output") or "")
                boxes = report_boxes(report)
                blocks = report_blocks(report)
                numeric_indices = numeric_anchor_indices(blocks, boxes)
                if not boxes or not numeric_indices:
                    skip = "no_numeric_anchor"
                else:
                    anchor_boxes = [boxes[idx] for idx in numeric_indices]
                    anchor_text = " ".join(semantic_block_text(blocks[idx]) for idx in numeric_indices)
                    report_numbers = extract_numbers(anchor_text)
                    core_numbers = core_report_numbers(report_numbers)
                    if len(core_numbers) < args.min_core_report_numbers:
                        skip = "few_report_numbers"
                    else:
                        best: tuple[float, dict[str, Any], str] | None = None
                        reasons: dict[str, int] = {}
                        for obs in observations.get(sid, []):
                            score, reason = candidate_score(
                                obs,
                                report_numbers=core_numbers,
                                anchor_boxes=anchor_boxes,
                                existing_boxes=boxes,
                                max_x_distance=args.max_x_distance,
                                max_y_distance=args.max_y_distance,
                            )
                            if reason != "ok":
                                reasons[reason] = reasons.get(reason, 0) + 1
                                continue
                            if best is None or score > best[0]:
                                best = (score, obs, reason)
                        if best is None:
                            skip = "no_candidate"
                        else:
                            replace_index = choose_replace_index(blocks, boxes, numeric_indices)
                            if replace_index is None:
                                skip = "no_replace_slot"
                            else:
                                score, obs, _ = best
                                cand = obs.get("candidate") or {}
                                cbox = [int(v) for v in (cand.get("box") or [])[:4]]
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
                                        "replace_index": replace_index,
                                        "numeric_anchor_indices": numeric_indices,
                                        "report_numbers": sorted(report_numbers),
                                        "core_report_numbers": sorted(core_numbers),
                                        "candidate": {
                                            "box": cbox,
                                            "text": str(cand.get("text") or ""),
                                            "source": str(cand.get("source") or ""),
                                            "family": str(cand.get("family") or ""),
                                            "score": score,
                                        },
                                        "policy": "GT-free Indonesian numeric-neighbor replacement from current report text and OCR action cache.",
                                    }
                                    row["stage_outputs"] = stage_outputs
                                    changes.append(
                                        {
                                            "sample_id": sid,
                                            "replace_index": replace_index,
                                            "old_box": boxes[replace_index],
                                            "new_box": cbox,
                                            "candidate_text": str(cand.get("text") or ""),
                                            "score": score,
                                            "report_numbers": sorted(report_numbers),
                                            "core_report_numbers": sorted(core_numbers),
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
