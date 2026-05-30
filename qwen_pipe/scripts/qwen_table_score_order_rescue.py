#!/usr/bin/env python3
"""Rescue authentic predictions when an OCR score table violates row order.

This module is GT-free.  It uses only the final report verdict and OCR layout
stage output.  The motivating failure mode is an event/results table where the
model describes a layout issue as benign, but the OCR rows reveal a descending
score table with a later row scoring higher than the row above it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_issue_refine import clamp_box, resolve_pipe_path, setup_debug_import  # noqa: E402


RANK_TERMS = (
    "champion",
    "runner-up",
    "runner up",
    "winner",
    "veteran",
    "place/category",
)
SCORE_RE = re.compile(r"(?<!\d)(\d{1,3})(?:\s*/\s*(\d{1,3}))?(?!\d)")


def parsed_conclusion(row: dict[str, Any]) -> str:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper()


def ocr_parsed(row: dict[str, Any]) -> dict[str, Any]:
    return (((row.get("stage_outputs") or {}).get("ocr_layout") or {}).get("parsed") or {})


def parse_score(text: str) -> int | None:
    matches = list(SCORE_RE.finditer(text or ""))
    if not matches:
        return None
    fraction_matches = [m for m in matches if m.group(2)]
    if fraction_matches:
        return int(fraction_matches[-1].group(1))
    last = matches[-1]
    value = int(last.group(1))
    if value > 100:
        return None
    # The score should be a trailing cell value, not a date or body number.
    if len((text or "")[last.end():].strip()) > 4:
        return None
    return value


def norm_text(value: Any) -> str:
    return str(value or "").lower()


def project_norm_box(box: list[float], width: int, height: int) -> list[int] | None:
    return clamp_box(
        [
            int(round(box[0] * width / 1000.0)),
            int(round(box[1] * height / 1000.0)),
            int(round(box[2] * width / 1000.0)),
            int(round(box[3] * height / 1000.0)),
        ],
        width,
        height,
    )


def detect_score_order_violation(row: dict[str, Any]) -> dict[str, Any] | None:
    if parsed_conclusion(row) != "AUTHENTIC":
        return None
    parsed = ocr_parsed(row)
    spans = list(parsed.get("text_spans") or [])
    if not spans:
        return None
    all_text = " ".join(str(s.get("text") or "") for s in spans).lower()
    if "score" not in all_text or not any(term in all_text for term in RANK_TERMS):
        return None

    score_header = None
    for span in spans:
        text = norm_text(span.get("text"))
        if "score" in text and str(span.get("role") or "").lower() == "table":
            score_header = span
            break
    if not score_header or not isinstance(score_header.get("bbox"), list):
        return None

    rows: list[dict[str, Any]] = []
    header_y = float(score_header["bbox"][1])
    for span in spans:
        if str(span.get("role") or "").lower() != "table":
            continue
        bbox = span.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        if float(bbox[1]) <= header_y:
            continue
        text = str(span.get("text") or "")
        score = parse_score(text)
        if score is None:
            continue
        if len(re.findall(r"[A-Za-z]", text)) < 3:
            continue
        rows.append({"id": span.get("id"), "text": text, "bbox": bbox, "score": score})
    rows.sort(key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))
    if len(rows) < 3:
        return None

    violations = []
    for prev, cur in zip(rows, rows[1:]):
        if int(cur["score"]) > int(prev["score"]):
            violations.append((prev, cur))
    if not violations:
        return None

    # Prefer a violation where a ranked/category row appears after an unlabeled
    # row.  This is a stronger forgery cue than a generic local disorder.
    selected_prev, selected_cur = violations[-1]
    for prev, cur in violations:
        prev_text = norm_text(prev["text"])
        cur_text = norm_text(cur["text"])
        if not any(term in prev_text for term in RANK_TERMS) and any(term in cur_text for term in RANK_TERMS):
            selected_prev, selected_cur = prev, cur
            break

    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    if width <= 0 or height <= 0:
        return None

    header_box = score_header["bbox"]
    cur_box = selected_cur["bbox"]
    norm_box = [
        max(0.0, float(header_box[0]) - 40.0),
        max(0.0, float(cur_box[1]) - 3.0),
        min(1000.0, float(header_box[0]) + 30.0),
        min(1000.0, float(cur_box[3]) + 4.0),
    ]
    native_box = project_norm_box(norm_box, width, height)
    if not native_box:
        return None

    return {
        "prev_row": selected_prev,
        "current_row": selected_cur,
        "score_header": score_header,
        "native_box": native_box,
        "norm_box": norm_box,
        "rows_considered": len(rows),
    }


def build_forged_report(row: dict[str, Any], detail: dict[str, Any]) -> str:
    sid = sample_id_from_row(row)
    prev = detail["prev_row"]
    cur = detail["current_row"]
    box = detail["native_box"]
    return f"""# FORGERY ANALYSIS REPORT

**Report ID:** TABLE-SCORE-ORDER-{sid}
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** FORGED
    **[RISK_SCORE]:** 72

---

## DETAILED ANOMALY ANALYSIS

### ANOMALY_001: Logical Fraud (Score Order Violation)
[GROUNDING]:[{box[0]}, {box[1]}, {box[2]}, {box[3]}]
[REASON]: The OCR table is a results/score table, but the row "{cur["text"]}" carries score {cur["score"]} after the preceding row "{prev["text"]}" with score {prev["score"]}. A later row scoring higher than the row immediately above it breaks the expected descending results order and is treated as localized score tampering rather than a benign layout issue.

---

## SUMMARY
The document is classified as forged because the OCR table structure exposes a localized score-order contradiction in the score column. This table-level validation converts the previously benign layout observation into a concrete, grounded anomaly.

---
**END OF REPORT**
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--stage-name", default="qwen_pipe_table_score_order_rescue")
    args = parser.parse_args()

    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"rows": 0, "changed": 0, "skipped": {}}
    changes: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as out:
        for row in read_jsonl(input_path):
            stats["rows"] += 1
            detail = detect_score_order_violation(row)
            if detail:
                row = dict(row)
                report = build_forged_report(row, detail)
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs[args.stage_name] = {
                    "applied": True,
                    "prev_row": detail["prev_row"],
                    "current_row": detail["current_row"],
                    "norm_box": detail["norm_box"],
                    "native_box": detail["native_box"],
                    "rows_considered": detail["rows_considered"],
                    "policy": "GT-free OCR score-table order rescue for authentic predictions with descending-rank violations.",
                }
                row["stage_outputs"] = stage_outputs
                stats["changed"] += 1
                changes.append({"sample_id": sample_id_from_row(row), **stage_outputs[args.stage_name]})
            else:
                stats["skipped"]["no_score_order_violation"] = int(stats["skipped"].get("no_score_order_violation", 0)) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {**stats, "changes": changes, "output_jsonl": str(output_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
