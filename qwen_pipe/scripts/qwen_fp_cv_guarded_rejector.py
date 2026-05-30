#!/usr/bin/env python3
"""Apply CV false-positive scores with hard-evidence protection.

The cross-validated FP score is useful for surfacing weak FORGED predictions,
but the highest-scoring bucket can still contain true forgeries with hard
localized visual evidence.  This applier keeps the score as a GT-free
inference artifact and adds a report-text guard for non-downgradable evidence
such as unreadable/gibberish fields, stamps, signatures, redactions, or solid
blocks.
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
from qwen_pair_replace_model import language_code  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


HARD_EVIDENCE_TERMS = (
    "gibberish",
    "unreadable",
    "illegible",
    "nonsense text",
    "black block",
    "redaction",
    "signature",
    "stamp",
    "seal",
    "barcode",
    "qr code",
    "غير مفهوم",
    "غير مقروء",
    "مشوه",
    "مشوّه",
    "رقم الوثيقة",
    "كتلة",
    "مستطيل",
    "حجب",
    "إخفاء",
)

ARABIC_GENERIC_DISTORTION_TERMS = (
    "غير مفهوم",
    "غير مقروء",
    "مشوه",
    "مشوّه",
)


def parsed_conclusion(row: dict[str, Any]) -> str:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper()


def has_hard_evidence(report: str, *, soft_arabic_generic_distortion: bool = False) -> bool:
    folded = report.lower()
    for term in HARD_EVIDENCE_TERMS:
        if soft_arabic_generic_distortion and term in ARABIC_GENERIC_DISTORTION_TERMS:
            continue
        if term.lower() in folded:
            return True
    return False


def authentic_report(row: dict[str, Any], score: float, detail: dict[str, Any]) -> str:
    sid = sample_id_from_row(row)
    image_name = str(row.get("image_name") or sid)
    return f"""# FORGERY ANALYSIS REPORT

**Report ID:** CV-GUARDED-FP-{sid}
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 5

---

## DETAILED ANOMALY ANALYSIS

No localized tampering evidence is retained after guarded false-positive validation. The previous FORGED decision is downgraded because the cross-stage FP score is high ({score:.4f}) and the report lacks protected hard evidence such as unreadable/gibberish fields, stamps, signatures, redactions, solid blocks, or other non-downgradable visual tamper cues.

---

## SUMMARY
The document image {image_name} is classified as authentic by the guarded CV false-positive rejector. Detail: {json.dumps(detail, ensure_ascii=False)}.

---
**END OF REPORT**
"""


def read_scores(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("sample_id") or "")
        if sid:
            out[sid] = row
    return out


def parse_language_thresholds(values: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values:
        if not value:
            continue
        if "=" not in value:
            raise SystemExit(f"--language-threshold must be LANG=FLOAT, got {value!r}")
        lang, raw_threshold = value.split("=", 1)
        try:
            out[lang.strip().lower()] = float(raw_threshold)
        except ValueError as exc:
            raise SystemExit(f"Invalid threshold in {value!r}") from exc
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--score-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--threshold", type=float, default=0.535)
    parser.add_argument(
        "--language-threshold",
        action="append",
        default=[],
        help="Optional language-specific threshold, e.g. ar=0.345. Repeated values are allowed.",
    )
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--stage-name", default="qwen_pipe_fp_cv_guarded_rejector")
    parser.add_argument(
        "--soft-arabic-generic-distortion",
        action="store_true",
        help=(
            "Do not treat generic Arabic distortion words as protected evidence by themselves. "
            "Specific hard cues such as stamp, redaction, block, document number, or gibberish remain protected."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    scores = read_scores(resolve_pipe_path(args.score_jsonl))
    out_path = resolve_pipe_path(args.output_jsonl)
    summary_path = resolve_pipe_path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "rows": 0,
        "scored_forged": 0,
        "selected_by_score": 0,
        "changed": 0,
        "protected_hard_evidence": 0,
        "threshold": args.threshold,
        "language_thresholds": parse_language_thresholds(args.language_threshold),
        "soft_arabic_generic_distortion": bool(args.soft_arabic_generic_distortion),
    }
    changed: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for row in read_jsonl(resolve_pipe_path(args.input_jsonl)):
            stats["rows"] += 1
            sid = sample_id_from_row(row)
            score_row = scores.get(sid)
            if score_row and parsed_conclusion(row) == "FORGED":
                stats["scored_forged"] += 1
                score = float(score_row.get("reject_score") or 0.0)
                lang = language_code(row).lower()
                threshold = stats["language_thresholds"].get(lang, args.threshold)
                if score >= threshold:
                    stats["selected_by_score"] += 1
                    report = str(row.get("raw_output") or "")
                    protected = has_hard_evidence(
                        report,
                        soft_arabic_generic_distortion=bool(args.soft_arabic_generic_distortion),
                    )
                    detail = {
                        "reject_score": score,
                        "threshold": threshold,
                        "language_code": lang,
                        "protected_hard_evidence": protected,
                    }
                    if protected:
                        stats["protected_hard_evidence"] += 1
                    else:
                        row = dict(row)
                        new_report = authentic_report(row, score, detail)
                        row["raw_output"] = new_report
                        row["parsed"] = parse_cct_report(new_report)
                        stage_outputs = dict(row.get("stage_outputs") or {})
                        stage_outputs[args.stage_name] = {
                            "applied": True,
                            **detail,
                            "policy": "Apply CV FP score only when no protected hard-evidence terms are present in the report.",
                        }
                        row["stage_outputs"] = stage_outputs
                        stats["changed"] += 1
                        changed.append({"sample_id": sid, **detail})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["changed_rows"] = changed
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
