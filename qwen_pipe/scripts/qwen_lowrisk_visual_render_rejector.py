#!/usr/bin/env python3
"""Reject low-risk visual-quality reports without hard tamper evidence.

This is a GT-free validation pass. It targets a narrow false-positive mode
left after the benign style/typo rejector: reports that classify a page as
FORGED only because of generic rendering quality concerns, with no OCR-grounded
date/amount/identity/sequence/signature/redaction evidence.
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

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_issue_refine import evidence_candidates, resolve_pipe_path, setup_debug_import  # noqa: E402
from qwen_pair_replace_model import language_code  # noqa: E402


TITLE_RE = re.compile(r"###\s*ANOMALY[^:]*:\s*(.*)", re.IGNORECASE)
REASON_RE = re.compile(
    r"\[REASON\]\s*:\s*(.*?)(?=\n\s*###|\n\s*---|\n\s*\[|\Z)",
    re.IGNORECASE | re.DOTALL,
)

VISUAL_ONLY_TITLE_TERMS = (
    "visual clumsy",
    "evidence-level local grounding",
    "exhaustive recall candidate",
)
HARD_TAMPER_TERMS = (
    "black block",
    "redaction",
    "signature",
    "stamp",
    "amount",
    "total",
    "score",
    "rank",
    "date",
    "period",
    "sequence",
    "identity",
    "id number",
    "logo",
    "math",
    "calculation",
    "copy-paste",
    "copy paste",
    "overlap",
    "obscur",
    "obscuring",
    "smudged",
    "blurred",
    "solid gray",
    "pixelated noise",
    "block of pixels",
    "كتلة",
    "مستطيل",
    "حجب",
    "إخفاء",
    "بقعة",
    "ضباب",
)


def parsed_conclusion(row: dict[str, Any]) -> str:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper()


def risk_score(row: dict[str, Any]) -> float:
    try:
        return float((row.get("parsed") or {}).get("risk_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def anomaly_count(row: dict[str, Any]) -> int:
    return len((row.get("parsed") or {}).get("anomalies") or [])


def full_report_text(row: dict[str, Any]) -> str:
    return str(row.get("raw_output") or "")


def anomaly_titles(report: str) -> list[str]:
    return [title.strip() for title in TITLE_RE.findall(report or "") if title.strip()]


def evidence_text(report: str) -> str:
    titles = " ".join(anomaly_titles(report))
    reasons = " ".join(match.group(1) for match in REASON_RE.finditer(report or ""))
    return f"{titles} {reasons}".lower()


def evidence_categories(row: dict[str, Any]) -> set[str]:
    return {str(cand.get("category") or "").lower() for cand in evidence_candidates(row)}


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def has_hard_tamper_term(text: str) -> bool:
    for term in HARD_TAMPER_TERMS:
        if re.fullmatch(r"[a-z0-9 -]+", term):
            pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                return True
        elif term in text:
            return True
    return False


def is_visual_only_report(report: str) -> bool:
    titles = anomaly_titles(report)
    if not titles:
        return False
    for title in titles:
        lower = title.lower()
        if not any(term in lower for term in VISUAL_ONLY_TITLE_TERMS):
            return False
    return True


def arabic_script_count(text: str) -> int:
    return sum(1 for char in text if "\u0600" <= char <= "\u06ff")


def should_reject(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if parsed_conclusion(row) != "FORGED":
        return False, {"reason": "not_forged"}
    if anomaly_count(row) > 3:
        return False, {"reason": "many_anomalies"}

    report = full_report_text(row)
    cats = evidence_categories(row)
    if cats != {"rendering_artifact"}:
        return False, {"reason": "not_pure_rendering_artifact", "categories": sorted(cats)}
    if not is_visual_only_report(report):
        return False, {"reason": "not_visual_only"}

    evidence = evidence_text(report)
    if has_hard_tamper_term(evidence):
        return False, {"reason": "hard_tamper_terms"}

    risk = risk_score(row)
    lang = language_code(row).lower()
    ar_like = lang == "ar" or arabic_script_count(report) >= 20
    low_risk = ar_like and risk <= 85
    if not low_risk:
        return False, {"reason": "risk_or_language_guard", "language_code": lang, "risk_score": risk}

    return True, {
        "reason": "benign_lowrisk_visual_rendering_artifact",
        "categories": sorted(cats),
        "risk_score": risk,
        "language_code": lang or ("ar-script" if ar_like else ""),
        "anomaly_count": anomaly_count(row),
        "box_count": len(report_boxes(report)),
    }


def authentic_report(row: dict[str, Any], detail: dict[str, Any]) -> str:
    sid = sample_id_from_row(row)
    image_name = str(row.get("image_name") or sid)
    categories = ", ".join(detail.get("categories") or [])
    return f"""# FORGERY ANALYSIS REPORT

**Report ID:** BENIGN-LOWRISK-VISUAL-{sid}
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 5

---

## DETAILED ANOMALY ANALYSIS

No localized tampering evidence is retained. The previous suspicion is downgraded by cross-stage validation because the report contains only low-risk visual rendering quality observations ({categories}) and no hard evidence such as altered dates, amounts, scores, identities, signatures, stamps, redactions, or OCR-grounded logical contradictions.

---

## SUMMARY
The document image {image_name} is classified as authentic after low-risk visual rendering validation. Generic rendering, font, or layout quality observations are not treated as sufficient evidence of forgery without a specific manipulated field or object.

---
**END OF REPORT**
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--stage-name", default="qwen_pipe_lowrisk_visual_render_rejector")
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
            reject, detail = should_reject(row)
            if reject:
                row = dict(row)
                report = authentic_report(row, detail)
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs[args.stage_name] = {
                    "applied": True,
                    **detail,
                    "policy": "GT-free downgrade for low-risk visual rendering-only reports without hard tamper evidence.",
                }
                row["stage_outputs"] = stage_outputs
                stats["changed"] += 1
                changes.append({"sample_id": sample_id_from_row(row), **stage_outputs[args.stage_name]})
            else:
                reason = str(detail.get("reason") or "unknown")
                stats["skipped"][reason] = int(stats["skipped"].get(reason, 0)) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {**stats, "changes": changes, "output_jsonl": str(output_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
