#!/usr/bin/env python3
"""Apply selected replacement candidates with GT-free typed filters.

This is used to combine a strong base run with a narrow rescue selector.  The
selected JSONL comes from ``qwen_candidate_delta_model.py`` diagnostics and
contains deploy-time candidate metadata plus the replacement index chosen
without GT.  This script never reads GT labels, masks, reports, or eval scores.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_candidate_delta_model import replace_groundings  # noqa: E402
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d|[\u0660-\u0669]")
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def csv_set(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def option_applies_to_family(value: str, family: str) -> bool:
    families = csv_set(value)
    return not families or family in families


def row_language(row: dict[str, Any]) -> str:
    return str(row.get("language_code") or ((row.get("metadata") or {}).get("language_code")) or "").lower()


def box_area(box: list[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def term_text(candidate: dict[str, Any]) -> str:
    meta = candidate.get("meta") or {}
    return str(meta.get("term") or str(candidate.get("text") or "").split("::", 1)[0]).strip()


def script_flags(text: str) -> set[str]:
    flags: set[str] = set()
    if ARABIC_RE.search(text):
        flags.add("arabic")
    if THAI_RE.search(text):
        flags.add("thai")
    if LATIN_RE.search(text):
        flags.add("latin")
    if DIGIT_RE.search(text):
        flags.add("digit")
    if CJK_RE.search(text):
        flags.add("cjk")
    return flags


def candidate_allowed(
    *,
    row: dict[str, Any],
    selected: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    cand = selected.get("candidate") or {}
    box = [int(v) for v in cand.get("box") or []]
    if len(box) < 4:
        return False, "invalid_box"
    lang = row_language(row)
    allowed_languages = csv_set(args.allow_languages)
    rejected_languages = csv_set(args.reject_languages)
    if allowed_languages and lang not in allowed_languages:
        return False, "language_not_allowed"
    if rejected_languages and lang in rejected_languages:
        return False, "language_rejected"
    allowed_families = csv_set(args.allow_families)
    family = str(cand.get("family") or "")
    if allowed_families and family not in allowed_families:
        return False, "family_not_allowed"
    source = str(cand.get("source") or "")
    if args.require_source_prefix and not source.startswith(args.require_source_prefix):
        return False, "source_prefix"
    pred_delta = float(selected.get("pred_delta") or 0.0)
    risk_score = float(selected.get("risk_score") if selected.get("risk_score") is not None else 1.0)
    if pred_delta < args.min_pred_delta:
        return False, "pred_delta_low"
    if risk_score < args.min_risk_score:
        return False, "risk_low"
    area = box_area(box)
    if args.max_area > 0 and area > args.max_area:
        return False, "area_high"
    if args.min_area > 0 and area < args.min_area:
        return False, "area_low"
    term = term_text(cand)
    cand_text = str(cand.get("text") or "")
    if args.min_term_chars > 0 and len(term) < args.min_term_chars:
        return False, "term_short"
    if args.max_term_chars > 0 and len(term) > args.max_term_chars:
        return False, "term_long"
    if args.max_candidate_text_chars > 0 and len(cand_text) > args.max_candidate_text_chars:
        return False, "candidate_text_long"
    if args.require_term_regex and not re.search(args.require_term_regex, term, flags=re.IGNORECASE):
        return False, "term_regex_missing"
    if args.reject_term_regex and re.search(args.reject_term_regex, term, flags=re.IGNORECASE):
        return False, "term_regex_rejected"
    if args.require_candidate_regex and not re.search(args.require_candidate_regex, cand_text, flags=re.IGNORECASE):
        return False, "candidate_regex_missing"
    if args.reject_candidate_regex and re.search(args.reject_candidate_regex, cand_text, flags=re.IGNORECASE):
        return False, "candidate_regex_rejected"
    flags = script_flags(term)
    if (
        args.reject_mixed_alnum_term
        and option_applies_to_family(args.reject_mixed_alnum_families, family)
        and "latin" in flags
        and "digit" in flags
    ):
        return False, "mixed_alnum_term"
    required_scripts = csv_set(args.require_term_script)
    rejected_scripts = csv_set(args.reject_term_script)
    if required_scripts and not (flags & required_scripts):
        return False, "term_script_missing"
    if rejected_scripts and (flags & rejected_scripts):
        return False, "term_script_rejected"
    if args.reject_numeric_term and flags <= {"digit"}:
        return False, "numeric_term"
    if args.require_alpha_term and not (flags & {"latin", "arabic", "thai", "cjk"}):
        return False, "alpha_term_missing"
    if args.linegrid_require_regex and family == "linegrid":
        if not re.search(args.linegrid_require_regex, str(cand.get("text") or ""), flags=re.IGNORECASE):
            return False, "linegrid_regex_missing"
    return True, "kept"


def load_selected(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or "")
        for selected in row.get("selected") or []:
            cand = selected.get("candidate") or {}
            if cand.get("box") and selected.get("replace_index") is not None:
                out[sid].append(selected)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-raw-jsonl", required=True)
    parser.add_argument("--selected-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--stage-name", default="qwen_apply_filtered_replacements")
    parser.add_argument("--allow-languages", default="")
    parser.add_argument("--reject-languages", default="")
    parser.add_argument("--allow-families", default="")
    parser.add_argument("--require-source-prefix", default="")
    parser.add_argument("--min-pred-delta", type=float, default=0.0)
    parser.add_argument("--min-risk-score", type=float, default=0.0)
    parser.add_argument("--min-area", type=float, default=0.0)
    parser.add_argument("--max-area", type=float, default=0.0)
    parser.add_argument("--require-term-script", default="")
    parser.add_argument("--reject-term-script", default="")
    parser.add_argument("--reject-numeric-term", action="store_true")
    parser.add_argument("--reject-mixed-alnum-term", action="store_true")
    parser.add_argument(
        "--reject-mixed-alnum-families",
        default="",
        help="Optional comma-separated families for --reject-mixed-alnum-term. Empty means all families.",
    )
    parser.add_argument("--require-alpha-term", action="store_true")
    parser.add_argument("--min-term-chars", type=int, default=0)
    parser.add_argument("--max-term-chars", type=int, default=0)
    parser.add_argument("--max-candidate-text-chars", type=int, default=0)
    parser.add_argument("--require-term-regex", default="")
    parser.add_argument("--reject-term-regex", default="")
    parser.add_argument("--require-candidate-regex", default="")
    parser.add_argument("--reject-candidate-regex", default="")
    parser.add_argument(
        "--linegrid-require-regex",
        default="",
        help="Optional regex that linegrid candidate text must match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    base_path = resolve_pipe_path(args.base_raw_jsonl)
    selected_path = resolve_pipe_path(args.selected_jsonl)
    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected_by_sid = load_selected(selected_path)
    counts: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for row in read_jsonl(base_path):
            sid = sample_id_from_row(row)
            replacements: dict[int, list[int]] = {}
            kept_items: list[dict[str, Any]] = []
            existing = report_boxes(str(row.get("raw_output") or ""))
            for item in selected_by_sid.get(sid) or []:
                ok, reason = candidate_allowed(row=row, selected=item, args=args)
                counts[reason] += 1
                if not ok:
                    continue
                replace_index = int(item.get("replace_index"))
                cand = item.get("candidate") or {}
                box = [int(v) for v in cand.get("box") or []][:4]
                if replace_index < 0 or replace_index >= len(existing):
                    counts["bad_replace_index"] += 1
                    continue
                replacements[replace_index] = box
                kept_items.append(item)
            if replacements:
                row = dict(row)
                report, replaced = replace_groundings(str(row.get("raw_output") or ""), replacements)
                if replaced:
                    row["raw_output"] = report
                    row["parsed"] = parse_cct_report(report)
                    stage_outputs = dict(row.get("stage_outputs") or {})
                    stage_outputs[args.stage_name] = {
                        "applied": True,
                        "selected_jsonl": str(selected_path),
                        "replaced": replaced,
                        "filters": {
                            "allow_languages": args.allow_languages,
                            "reject_languages": args.reject_languages,
                            "allow_families": args.allow_families,
                            "require_source_prefix": args.require_source_prefix,
                            "min_pred_delta": args.min_pred_delta,
                            "min_risk_score": args.min_risk_score,
                            "min_area": args.min_area,
                            "max_area": args.max_area,
                            "require_term_script": args.require_term_script,
                            "reject_term_script": args.reject_term_script,
                            "reject_numeric_term": bool(args.reject_numeric_term),
                            "reject_mixed_alnum_term": bool(args.reject_mixed_alnum_term),
                            "reject_mixed_alnum_families": args.reject_mixed_alnum_families,
                            "require_alpha_term": bool(args.require_alpha_term),
                            "linegrid_require_regex": args.linegrid_require_regex,
                            "min_term_chars": args.min_term_chars,
                            "max_term_chars": args.max_term_chars,
                            "max_candidate_text_chars": args.max_candidate_text_chars,
                            "require_term_regex": args.require_term_regex,
                            "reject_term_regex": args.reject_term_regex,
                            "require_candidate_regex": args.require_candidate_regex,
                            "reject_candidate_regex": args.reject_candidate_regex,
                        },
                        "kept": [
                            {
                                "replace_index": int(item.get("replace_index")),
                                "candidate": item.get("candidate"),
                                "pred_delta": item.get("pred_delta"),
                                "risk_score": item.get("risk_score"),
                            }
                            for item in kept_items
                        ],
                        "policy": "GT-free typed filter over selected replacement candidates.",
                    }
                    row["stage_outputs"] = stage_outputs
                    changed.append(
                        {
                            "sample_id": sid,
                            "language_code": row_language(row),
                            "replacements": replacements,
                            "kept_count": len(kept_items),
                        }
                    )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "base_raw_jsonl": str(base_path),
        "selected_jsonl": str(selected_path),
        "output_jsonl": str(out_path),
        "filter_counts": dict(counts),
        "changed_count": len(changed),
        "changed": changed,
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
