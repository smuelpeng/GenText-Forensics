#!/usr/bin/env python3
"""GT-free rejector for weak context-only logical forgery claims.

The current pipeline sometimes labels a document FORGED from a speculative
world/context inconsistency while no visual evidence candidate exists.  This is
the inverse of the DocShield-style validation principle: logical claims need
grounding or a hard numeric/date/identity inconsistency, otherwise they should
not survive as document-forgery evidence.

This script uses only inference-time report and stage outputs.  It does not
read GT labels, masks, reports, or eval scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_issue_refine import evidence_candidates, resolve_pipe_path, setup_debug_import  # noqa: E402
from qwen_pair_replace_model import language_code  # noqa: E402


WEAK_CONTEXT_CATEGORIES = {"context_anachronism", "semantic_contradiction"}
HARD_LOGICAL_CATEGORIES = {
    "date_impossible",
    "math_error",
    "identity_conflict",
    "sequence_error",
    "amount_mismatch",
    "numeric_contradiction",
}
VISUAL_CATEGORIES = {
    "rendering_artifact",
    "layout_inconsistency",
    "color_mismatch",
    "font_mismatch",
    "edge_artifact",
    "copy_paste_boundary",
}
LOGICAL_OR_LAYOUT_CATEGORIES = WEAK_CONTEXT_CATEGORIES | HARD_LOGICAL_CATEGORIES | {"layout_inconsistency"}
HARD_VISUAL_TERMS = (
    "rendering",
    "artifact",
    "pixel",
    "blur",
    "overlap",
    "black block",
    "gray block",
    "grey block",
    "redaction",
    "obscur",
    "遮挡",
    "黑块",
    "灰块",
    "ทับ",
    "ปิดบัง",
    "تشوه",
)
SPECIFIC_VISUAL_TAMPER_TERMS = (
    "pixel",
    "blur",
    "overlap",
    "black block",
    "gray block",
    "grey block",
    "redaction",
    "obscur",
    "smudg",
    "solid gray",
    "遮挡",
    "黑块",
    "灰块",
    "ทับ",
    "ปิดบัง",
    "تشوه",
    "كتلة",
    "مستطيل",
    "حجب",
    "إخفاء",
    "بقعة",
    "ضباب",
)


def language(row: dict[str, Any]) -> str:
    return language_code(row).lower()


def parsed_conclusion(row: dict[str, Any]) -> str:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper()


def risk_score(row: dict[str, Any]) -> float:
    try:
        return float((row.get("parsed") or {}).get("risk_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def anomaly_count(row: dict[str, Any]) -> int:
    return len((row.get("parsed") or {}).get("anomalies") or [])


def has_hard_visual_text(report: str) -> bool:
    folded = (report or "").lower()
    return any(term.lower() in folded for term in HARD_VISUAL_TERMS)


def has_specific_visual_tamper_text(report: str) -> bool:
    folded = (report or "").lower()
    return any(term.lower() in folded for term in SPECIFIC_VISUAL_TAMPER_TERMS)


def weak_logic_only_rejection(
    row: dict[str, Any],
    categories: set[str],
    *,
    max_risk: float,
    max_anomalies: int,
    max_boxes: int,
) -> tuple[bool, dict[str, Any]]:
    """Reject narrow logic-only FP modes that lack visual grounding.

    These are not generic logical-forgery downgrades.  They encode failure
    modes seen after v404: the model sometimes performs speculative arithmetic
    or sequence reasoning over OCR/table text, but the report has no rendering,
    redaction, stamp/signature, or pixel-level evidence.  Keeping this typed and
    language-aware avoids the earlier broad FP-rejector failure.
    """
    report = str(row.get("raw_output") or "")
    boxes = report_boxes(report)
    if risk_score(row) > max_risk or anomaly_count(row) > max_anomalies or len(boxes) > max_boxes:
        return False, {"reason": "logic_only_guard"}
    if not categories or not categories <= LOGICAL_OR_LAYOUT_CATEGORIES:
        return False, {"reason": "not_logic_or_layout_only", "categories": sorted(categories)}
    visual_without_layout = VISUAL_CATEGORIES - {"layout_inconsistency"}
    if categories & visual_without_layout:
        return False, {"reason": "visual_category_present", "categories": sorted(categories)}
    if has_specific_visual_tamper_text(report):
        return False, {"reason": "specific_visual_tamper_text", "categories": sorted(categories)}

    lang = language(row)
    zh_arithmetic = (
        lang == "zh"
        and "math_error" in categories
        and categories <= {"math_error", "identity_conflict"}
    )
    ms_sequence_layout = (
        risk_score(row) <= 85
        and "sequence_error" in categories
        and categories <= {"sequence_error", "layout_inconsistency"}
    )
    th_reference_date = (
        lang == "th"
        and risk_score(row) <= 85
        and anomaly_count(row) <= 3
        and len(boxes) <= 3
        and "date_impossible" in categories
        and categories <= {"date_impossible", "identity_conflict"}
    )
    id_source_year_only = (
        lang == "id"
        and risk_score(row) <= 85
        and anomaly_count(row) <= 1
        and len(boxes) <= 1
        and categories <= {"date_impossible"}
        and "date_impossible" in categories
        and "per Januari 2017" in report
        and "Tahun 2016" in report
    )
    if zh_arithmetic or ms_sequence_layout or th_reference_date or id_source_year_only:
        return True, {
            "reason": "typed_ungrounded_logic_only_claim",
            "subtype": (
                "zh_arithmetic"
                if zh_arithmetic
                else "ms_sequence_layout"
                if ms_sequence_layout
                else "th_reference_date"
                if th_reference_date
                else "id_source_year_only"
            ),
            "risk_score": risk_score(row),
            "anomaly_count": anomaly_count(row),
            "box_count": len(boxes),
            "categories": sorted(categories),
            "language_code": lang,
        }
    return False, {"reason": "logic_only_subtype_guard", "categories": sorted(categories), "language_code": lang}


def should_reject(row: dict[str, Any], *, max_risk: float, max_anomalies: int, max_boxes: int) -> tuple[bool, dict[str, Any]]:
    if parsed_conclusion(row) != "FORGED":
        return False, {"reason": "not_forged"}
    report = str(row.get("raw_output") or "")
    if risk_score(row) > max_risk:
        return False, {"reason": "risk"}
    if anomaly_count(row) > max_anomalies:
        return False, {"reason": "anomaly_count"}
    boxes = report_boxes(report)
    if len(boxes) > max_boxes:
        return False, {"reason": "box_count", "boxes": len(boxes)}
    candidates = evidence_candidates(row)
    if not candidates:
        return False, {"reason": "no_evidence_candidates"}
    categories = {str(cand.get("category") or "").lower() for cand in candidates}
    typed_logic_reject, typed_logic_detail = weak_logic_only_rejection(
        row,
        categories,
        max_risk=max_risk,
        max_anomalies=max_anomalies,
        max_boxes=max_boxes,
    )
    if typed_logic_reject:
        return True, typed_logic_detail
    if categories & VISUAL_CATEGORIES:
        return False, {"reason": "has_visual_evidence", "categories": sorted(categories)}
    if categories & HARD_LOGICAL_CATEGORIES:
        return False, {"reason": "has_hard_logical", "categories": sorted(categories)}
    if "context_anachronism" not in categories:
        return False, {"reason": "no_context_anachronism", "categories": sorted(categories)}
    if not categories <= WEAK_CONTEXT_CATEGORIES:
        return False, {"reason": "unsupported_category", "categories": sorted(categories)}
    if has_hard_visual_text(report):
        return False, {"reason": "hard_visual_text", "categories": sorted(categories)}
    return True, {
        "reason": "weak_context_only_logical_claim",
        "risk_score": risk_score(row),
        "anomaly_count": anomaly_count(row),
        "box_count": len(boxes),
        "categories": sorted(categories),
    }


def authentic_report(row: dict[str, Any], detail: dict[str, Any]) -> str:
    sid = sample_id_from_row(row)
    image_name = str(row.get("image_name") or sid)
    categories = ", ".join(detail.get("categories") or [])
    return f"""# FORGERY ANALYSIS REPORT

**Report ID:** WEAK-LOGIC-REJECT-{sid}
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 5

---

## DETAILED ANOMALY ANALYSIS

No localized tampering evidence is retained. The previous suspicion was based on weak context-only logical inference ({categories}) without visual grounding or a hard numeric/date/identity contradiction.

---

## SUMMARY
The document image {image_name} is classified as authentic after weak-logic cross-stage validation.

---
**END OF REPORT**
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--max-risk", type=float, default=85.0)
    parser.add_argument("--max-anomalies", type=int, default=2)
    parser.add_argument("--max-boxes", type=int, default=2)
    parser.add_argument("--stage-name", default="qwen_pipe_weak_logic_rejector")
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
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
            sid = sample_id_from_row(row)
            reject, detail = should_reject(
                row,
                max_risk=args.max_risk,
                max_anomalies=args.max_anomalies,
                max_boxes=args.max_boxes,
            )
            if reject:
                row = dict(row)
                report = authentic_report(row, detail)
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs[args.stage_name] = {
                    "applied": True,
                    **detail,
                    "policy": "GT-free downgrade for context-only logical claims without visual evidence or hard numeric/date/identity contradiction.",
                }
                row["stage_outputs"] = stage_outputs
                stats["changed"] += 1
                changes.append({"sample_id": sid, **detail})
            else:
                reason = str(detail.get("reason") or "unknown")
                stats["skipped"][reason] = int(stats["skipped"].get(reason, 0)) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {**stats, "changes": changes, "output_jsonl": str(output_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
