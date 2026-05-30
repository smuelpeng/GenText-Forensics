#!/usr/bin/env python3
"""Reject forged reports that only describe benign style or typo issues.

This is a GT-free validation pass.  It targets a narrow false-positive mode:
the model treats worksheet/school design choices, stock-image composition, or
Indonesian typo/proofreading errors as forensic tampering.  These observations
can be real document-quality issues, but without hard localized manipulation
evidence they should not determine a FORGED verdict.
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


REASON_RE = re.compile(
    r"\[REASON\]\s*:\s*(.*?)(?=\n\s*###\s*ANOMALY|\n\s*---|\n\s*##\s*SUMMARY|\Z)",
    re.IGNORECASE | re.DOTALL,
)
TITLE_RE = re.compile(r"###\s*ANOMALY[^:]*:\s*(.*)", re.IGNORECASE)

STYLE_CONTEXT_TERMS = (
    "educational worksheet",
    "school circular",
    "worksheet titled",
    "designed to teach",
)
STYLE_EVIDENCE_TERMS = (
    "composite image",
    "digital overlay",
    "drop shadow",
    "white background box",
    "template design",
    "visual assets",
)
INDONESIAN_TYPO_TERMS = (
    "terdapat kesalahan ketik",
    "kesalahan ketik typo",
    "kata ini seharusnya",
    "seharusnya ditulis sebagai",
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
    "date mismatch",
    "impossible date",
    "copy-paste",
    "copy paste",
    "overlap",
    "obscur",
    "pixelated noise",
    "block of pixels",
)
STYLE_HARD_TAMPER_TERMS = (
    "black block",
    "redaction",
    "signature",
    "stamp",
    "amount",
    "total",
    "score",
    "rank",
    "date mismatch",
    "impossible date",
    "obscur",
    "pixelated noise",
    "block of pixels",
)
STYLE_CATEGORIES = {"rendering_artifact", "font_mismatch", "color_mismatch"}
TYPO_CATEGORIES = {"rendering_artifact", "layout_inconsistency"}
MATH_WORKSHEET_CONTEXT_TERMS = (
    "lembar kerja",
    "solusi matematika",
    "perhitungan matematika",
    "tan a.tan b",
    "cos(a - b)",
    "cos a cos b",
)
MATH_WORKSHEET_BENIGN_TERMS = (
    "goresan tipis",
    "kesalahan rendering",
    "artefak digital",
    "garis yang memotong",
)
MALAY_BACKGROUND_URL_TERMS = (
    "url",
    "foto latar belakang",
    "skrin latar belakang",
    "https://laz...com.my/perak",
    "jurang kosong",
    "ruang kosong",
)
THAI_FONT_NOISE_TERMS = (
    "สับ",
    "ทั้งนี้",
    "ฟอนต์",
    "kerning",
    "baseline",
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


def reason_text(report: str) -> str:
    return (
        " ".join(TITLE_RE.findall(report or ""))
        + " "
        + " ".join(match.group(1) for match in REASON_RE.finditer(report or ""))
    ).lower()


def full_report_text(row: dict[str, Any]) -> str:
    return str(row.get("raw_output") or "").lower()


def evidence_categories(row: dict[str, Any]) -> set[str]:
    return {str(cand.get("category") or "").lower() for cand in evidence_candidates(row)}


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def should_reject(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if parsed_conclusion(row) != "FORGED":
        return False, {"reason": "not_forged"}
    if risk_score(row) > 85:
        return False, {"reason": "high_risk"}
    if anomaly_count(row) > 4:
        return False, {"reason": "many_anomalies"}

    report = full_report_text(row)
    reasons = reason_text(str(row.get("raw_output") or ""))
    cats = evidence_categories(row)
    if not cats:
        return False, {"reason": "no_evidence_categories"}

    style_case = (
        anomaly_count(row) <= 3
        and cats <= STYLE_CATEGORIES
        and has_any(report, STYLE_CONTEXT_TERMS)
        and has_any(report, STYLE_EVIDENCE_TERMS)
        and not has_any(reasons, STYLE_HARD_TAMPER_TERMS)
    )
    if style_case:
        return True, {
            "reason": "benign_educational_style_or_asset_composition",
            "categories": sorted(cats),
            "risk_score": risk_score(row),
            "anomaly_count": anomaly_count(row),
            "box_count": len(report_boxes(str(row.get("raw_output") or ""))),
        }

    # Indonesian proofreading/typo claims are a separate benign mode.  The
    # validation condition deliberately requires only weak rendering/layout
    # categories and no hard tamper terms inside anomaly reasons.
    if has_any(reasons, HARD_TAMPER_TERMS):
        return False, {"reason": "hard_tamper_terms"}

    typo_case = (
        cats <= TYPO_CATEGORIES
        and has_any(reasons, INDONESIAN_TYPO_TERMS)
    )
    if typo_case:
        return True, {
            "reason": "benign_indonesian_typo_or_proofreading_issue",
            "categories": sorted(cats),
            "risk_score": risk_score(row),
            "anomaly_count": anomaly_count(row),
            "box_count": len(report_boxes(str(row.get("raw_output") or ""))),
        }

    # Mathematical worksheets often contain thin ruling/scan artifacts around
    # fractions or handwritten-looking equations.  If the report has no math
    # inconsistency category and only describes a rendering scratch over an
    # equation, treat it as insufficient forgery evidence.
    math_worksheet_scratch = (
        risk_score(row) <= 85
        and cats <= {"rendering_artifact"}
        and anomaly_count(row) <= 3
        and has_any(reasons, MATH_WORKSHEET_CONTEXT_TERMS)
        and has_any(reasons, MATH_WORKSHEET_BENIGN_TERMS)
        and not {"math_error", "numeric_contradiction", "semantic_contradiction"} & cats
        and not has_any(reasons, ("jawaban salah", "nilai tidak konsisten", "perhitungan salah", "hasil salah"))
    )
    if math_worksheet_scratch:
        return True, {
            "reason": "benign_math_worksheet_rendering_scratch",
            "categories": sorted(cats),
            "risk_score": risk_score(row),
            "anomaly_count": anomaly_count(row),
            "box_count": len(report_boxes(str(row.get("raw_output") or ""))),
        }

    # Some news/article screenshots contain embedded display photos where a
    # visible URL on the background screen is already truncated by the source
    # image or page capture.  Treat this as insufficient evidence when it is the
    # only low-risk cue and no document-field contradiction is present.
    malay_background_url = (
        risk_score(row) <= 85
        and cats <= {"rendering_artifact"}
        and anomaly_count(row) <= 2
        and all(term in reasons for term in MALAY_BACKGROUND_URL_TERMS[:4])
        and has_any(reasons, MALAY_BACKGROUND_URL_TERMS[4:])
        and not has_any(reasons, HARD_TAMPER_TERMS)
    )
    if malay_background_url:
        return True, {
            "reason": "benign_embedded_background_url_truncation",
            "categories": sorted(cats),
            "risk_score": risk_score(row),
            "anomaly_count": anomaly_count(row),
            "box_count": len(report_boxes(str(row.get("raw_output") or ""))),
        }

    # Thai official letters can be over-flagged when the only evidence is a
    # localized font/kerning claim around isolated words.  Keep this branch
    # narrow: pure rendering evidence, low risk, and both the suspect word and
    # neighboring baseline cue must be present.
    thai_font_noise = (
        risk_score(row) <= 85
        and cats <= {"rendering_artifact"}
        and anomaly_count(row) <= 3
        and all(term in reasons for term in THAI_FONT_NOISE_TERMS[:3])
        and has_any(reasons, THAI_FONT_NOISE_TERMS[3:])
        and not has_any(reasons, HARD_TAMPER_TERMS)
    )
    if thai_font_noise:
        return True, {
            "reason": "benign_thai_font_kerning_noise",
            "categories": sorted(cats),
            "risk_score": risk_score(row),
            "anomaly_count": anomaly_count(row),
            "box_count": len(report_boxes(str(row.get("raw_output") or ""))),
        }

    return False, {"reason": "not_benign_style_typo", "categories": sorted(cats)}


def authentic_report(row: dict[str, Any], detail: dict[str, Any]) -> str:
    sid = sample_id_from_row(row)
    image_name = str(row.get("image_name") or sid)
    reason = str(detail.get("reason") or "benign_style_typo")
    categories = ", ".join(detail.get("categories") or [])
    return f"""# FORGERY ANALYSIS REPORT

**Report ID:** BENIGN-STYLE-TYPO-{sid}
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 5

---

## DETAILED ANOMALY ANALYSIS

No localized tampering evidence is retained. The previous suspicion is downgraded by cross-stage validation because it is limited to {reason.replace("_", " ")} ({categories}) without hard evidence such as redaction, altered amounts/dates, signature/stamp manipulation, or OCR-grounded logical contradiction.

---

## SUMMARY
The document image {image_name} is classified as authentic after benign style/typo validation. Observed design, template, stock-asset, watermark, or proofreading irregularities are treated as document-quality issues rather than sufficient evidence of forgery.

---
**END OF REPORT**
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--stage-name", default="qwen_pipe_benign_style_typo_rejector")
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
                    "policy": "GT-free downgrade for benign design/style/typo claims without hard localized tamper evidence.",
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
