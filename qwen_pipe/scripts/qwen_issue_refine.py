#!/usr/bin/env python3
"""Issue-level detection refinement for Qwen-pipe.

This postprocess is GT-blind at inference time. It borrows two DocShield /
FakeShield ideas:

* keep discarded hard visual candidates for a second-pass rescue, instead of
  letting validation erase all localization;
* downgrade only narrow world-knowledge/template claims that lack hard local
  visual evidence.

The script reads current raw JSONL records and rewrites final reports only from
existing stage outputs. GT eval files are not read by this script.
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
DEFAULT_INPUT = PIPE_ROOT / "outputs/raw/qwen_pipe_v74_ocr_cluster_gated_300.jsonl"


HARD_VISUAL_TERMS = (
    "cover",
    "covered",
    "covering",
    "obscur",
    "unreadable",
    "unclear",
    "floating",
    "not integrated",
    "broken",
    "abnormal",
    "abnormally",
    "gray block",
    "grey block",
    "dark spot",
    "dark gray",
    "redaction",
    "pixelated",
    "rendering artifact",
    "layout error",
    "word wrap",
    "spacing",
    "遮挡",
    "遮蔽",
    "覆盖",
    "灰块",
    "黑块",
    "乱码",
    "无法清晰",
    "ผิดปกติ",
    "ไม่ชัด",
    "เว้นวรรคผิดปกติ",
    "จุดดำ",
    "จุดสีดำ",
    "بقعة",
    "مربع",
    "تغطي",
    "غير واضحة",
    "محجوبة",
)

BENIGN_RESCUE_EXCLUDE_TERMS = (
    "logo",
    "standard typography",
    "newspaper",
    "sequence of dashes",
    "dashes",
    "placeholder",
    "replaced by placeholders",
    "replacing numerical data",
    "normal stock",
    "missing data",
)

HARD_FINAL_TERMS = (
    "pixel",
    "pixelated",
    "blur",
    "blurry",
    "ghost",
    "jagged",
    "unreadable",
    "obscur",
    "covered",
    "covering",
    "black block",
    "gray block",
    "grey block",
    "redaction",
    "floating",
    "not integrated",
    "overlap",
    "乱码",
    "模糊",
    "重影",
    "像素",
    "遮挡",
    "黑块",
    "灰块",
    "ผิดปกติ",
    "ไม่ชัด",
    "بقعة",
    "مربع",
    "تغطي",
)

WORLD_ONLY_TERMS = (
    "official",
    "government",
    "logo",
    "language",
    "ontario",
    "template placeholder",
    "page number",
    "官方",
    "政府",
    "徽标",
    "语言",
    "模板",
    "占位符",
    "页码",
)

NUMERIC_PROTECT_TERMS = (
    "impossible",
    "contradiction",
    "conflicting",
    "math",
    "sum",
    "total",
    "amount",
    "date",
    "negative",
    "percentage",
    "金额",
    "日期",
    "数字",
    "合计",
    "总计",
    "矛盾",
    "不可能",
    "编号",
)


def resolve_pipe_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PIPE_ROOT / p


def setup_debug_import(debug_root: Path) -> None:
    docshield_dir = debug_root / "baselines" / "DocShield"
    if not docshield_dir.exists():
        raise SystemExit(f"DocShield helper directory not found: {docshield_dir}")
    sys.path.insert(0, str(docshield_dir))
    sys.path.insert(0, str(PIPE_ROOT / "scripts"))


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    folded = text.lower()
    return any(term.lower() in folded for term in terms)


def parsed_conclusion(row: dict[str, Any]) -> str:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper()


def parsed_anomaly_count(row: dict[str, Any]) -> int:
    return len((row.get("parsed") or {}).get("anomalies") or [])


def evidence_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = (((row.get("stage_outputs") or {}).get("evidence_candidates") or {}).get("parsed") or {})
    out: list[dict[str, Any]] = []
    for key in ("visual_candidates", "logical_candidates"):
        for cand in parsed.get(key) or []:
            if isinstance(cand, dict):
                item = dict(cand)
                item["_source_list"] = key
                out.append(item)
    return out


def validation_discard_ids(row: dict[str, Any]) -> set[str]:
    parsed = (((row.get("stage_outputs") or {}).get("validation") or {}).get("parsed") or {})
    ids: set[str] = set()
    for discarded in parsed.get("discarded_candidates") or []:
        if isinstance(discarded, dict) and discarded.get("id"):
            ids.add(str(discarded.get("id")))
    return ids


def candidate_text(cand: dict[str, Any]) -> str:
    return " ".join(
        str(cand.get(key) or "")
        for key in ("category", "evidence", "notes", "reason")
    )


def parse_box(value: Any) -> list[float] | None:
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(nums) < 4:
            return None
        return [float(v) for v in nums[:4]]
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[i]) for i in range(4)]
        except (TypeError, ValueError):
            return None
    return None


def hard_visual_candidates(row: dict[str, Any], *, min_confidence: float) -> list[dict[str, Any]]:
    discarded_ids = validation_discard_ids(row)
    candidates = evidence_candidates(row)
    if discarded_ids:
        candidates = [cand for cand in candidates if str(cand.get("id") or "") in discarded_ids]

    hard: list[dict[str, Any]] = []
    for cand in candidates:
        category = str(cand.get("category") or "").lower()
        if category not in {"rendering_artifact", "color_mismatch", "layout_inconsistency"}:
            continue
        try:
            confidence = float(cand.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            continue
        if not parse_box(cand.get("bbox")):
            continue
        text = candidate_text(cand)
        if has_any(text, BENIGN_RESCUE_EXCLUDE_TERMS):
            continue
        if has_any(text, HARD_VISUAL_TERMS):
            hard.append(cand)
    return hard


def should_rescue(row: dict[str, Any], *, min_hard_candidates: int, min_confidence: float) -> tuple[bool, list[dict[str, Any]], str]:
    if parsed_conclusion(row) != "AUTHENTIC":
        return False, [], "non_authentic"
    hard = hard_visual_candidates(row, min_confidence=min_confidence)
    if len(hard) < min_hard_candidates:
        return False, hard, "not_enough_hard_candidates"
    return True, hard, "hard_discarded_candidate_rescue"


def should_downgrade_world_only(row: dict[str, Any]) -> tuple[bool, str]:
    if parsed_conclusion(row) != "FORGED":
        return False, "non_forged"
    report = str(row.get("raw_output") or "")
    detail = report
    marker = re.search(r"DETAILED ANOMALY ANALYSIS", report, re.IGNORECASE)
    if marker:
        detail = report[marker.end():]
    summary = re.search(r"\n\s*-{3,}\s*\n\s*##\s*SUMMARY|\n\s*##\s*SUMMARY", detail, re.IGNORECASE)
    if summary:
        detail = detail[: summary.start()]
    folded = detail.lower()
    if parsed_anomaly_count(row) > 3:
        return False, "too_many_anomalies"
    if not has_any(folded, WORLD_ONLY_TERMS):
        return False, "no_world_only_terms"
    if has_any(folded, HARD_FINAL_TERMS):
        return False, "has_hard_visual_terms"
    if has_any(folded, NUMERIC_PROTECT_TERMS):
        return False, "has_numeric_protect_terms"
    return True, "world_knowledge_or_template_only"


def clamp_box(box: list[int], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    out = [
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    ]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def project_box(raw_box: list[float], width: int, height: int, mode: str) -> list[int] | None:
    if mode == "normalized-1000":
        return clamp_box(
            [
                int(round(raw_box[0] * width / 1000.0)),
                int(round(raw_box[1] * height / 1000.0)),
                int(round(raw_box[2] * width / 1000.0)),
                int(round(raw_box[3] * height / 1000.0)),
            ],
            width,
            height,
        )
    return clamp_box([int(round(v)) for v in raw_box[:4]], width, height)


def build_rescue_report(row: dict[str, Any], candidates: list[dict[str, Any]], *, max_anomalies: int, coord_mode: str) -> str:
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    sample_id = str(row.get("sample_id") or "unknown")
    selected: list[tuple[dict[str, Any], list[int]]] = []
    for cand in candidates:
        raw_box = parse_box(cand.get("bbox"))
        if not raw_box:
            continue
        box = project_box(raw_box, width, height, coord_mode)
        if box:
            selected.append((cand, box))
        if len(selected) >= max_anomalies:
            break

    lines = [
        "# FORGERY ANALYSIS REPORT",
        f"**Report ID:** QWEN-ISSUE-REFINE-{sample_id}",
        "**Case Type:** Document Authentication & Fraud Analysis",
        "",
        "## Overall Assessment",
        "**[Conclusion]:** FORGED",
        "**[RISK_SCORE]:** 85",
        "",
        "## DETAILED ANOMALY ANALYSIS",
    ]
    for idx, (cand, box) in enumerate(selected, start=1):
        category = str(cand.get("category") or "visual_candidate")
        evidence = re.sub(r"\s+", " ", str(cand.get("evidence") or "Hard visual candidate retained after validation discard.")).strip()
        lines.extend(
            [
                f"### ANOMALY_{idx:03d}: Hard Visual Candidate ({category})",
                f"[GROUNDING]:{box}",
                f"[REASON]: {evidence} This candidate was initially detected as localized visual evidence and is retained because multiple hard local artifacts are present.",
                "",
            ]
        )
    lines.extend(
        [
            "---",
            "## SUMMARY",
            f"The second-pass issue reviewer retained {len(selected)} hard localized visual candidates that were discarded by the validation stage. The document is therefore treated as forged with explicit local grounding.",
            "",
            "---",
            "**END OF REPORT**",
        ]
    )
    return "\n".join(lines)


def build_authentic_report(row: dict[str, Any], reason: str) -> str:
    sample_id = str(row.get("sample_id") or "unknown")
    return "\n".join(
        [
            "# FORGERY ANALYSIS REPORT",
            f"**Report ID:** QWEN-ISSUE-REFINE-{sample_id}",
            "**Case Type:** Document Authentication & Fraud Analysis",
            "",
            "## Overall Assessment",
            "**[Conclusion]:** AUTHENTIC",
            "**[RISK_SCORE]:** 5",
            "",
            "## DETAILED ANOMALY ANALYSIS",
            "No anomalies detected.",
            "",
            "---",
            "## SUMMARY",
            f"The second-pass issue reviewer downgraded the prior forged decision because it was supported only by weak document-practice or world-knowledge cues without hard localized tamper evidence. Reason: {reason}.",
            "",
            "---",
            "**END OF REPORT**",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--enable-rescue", action="store_true")
    parser.add_argument("--enable-world-downgrade", action="store_true")
    parser.add_argument("--min-hard-candidates", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--max-rescue-anomalies", type=int, default=6)
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel"], default="normalized-1000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    from postprocess import parse_cct_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "rows": 0,
        "rescued": 0,
        "downgraded": 0,
        "unchanged": 0,
        "rescue_reasons": {},
        "downgrade_reasons": {},
        "enable_rescue": args.enable_rescue,
        "enable_world_downgrade": args.enable_world_downgrade,
    }

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            out = dict(row)
            action = "unchanged"
            action_reason = "none"
            hard_count = 0

            if args.enable_rescue:
                ok, hard, reason = should_rescue(
                    row,
                    min_hard_candidates=args.min_hard_candidates,
                    min_confidence=args.min_confidence,
                )
                hard_count = len(hard)
                stats["rescue_reasons"][reason] = stats["rescue_reasons"].get(reason, 0) + 1
                if ok:
                    report = build_rescue_report(
                        row,
                        hard,
                        max_anomalies=args.max_rescue_anomalies,
                        coord_mode=args.coord_mode,
                    )
                    out["raw_output"] = report
                    out["parsed"] = parse_cct_report(report)
                    action = "rescued"
                    action_reason = reason
                    stats["rescued"] += 1

            if action == "unchanged" and args.enable_world_downgrade:
                ok, reason = should_downgrade_world_only(row)
                stats["downgrade_reasons"][reason] = stats["downgrade_reasons"].get(reason, 0) + 1
                if ok:
                    report = build_authentic_report(row, reason)
                    out["raw_output"] = report
                    out["parsed"] = parse_cct_report(report)
                    action = "downgraded"
                    action_reason = reason
                    stats["downgraded"] += 1

            if action == "unchanged":
                stats["unchanged"] += 1

            stage_outputs = dict(out.get("stage_outputs") or {})
            stage_outputs["qwen_pipe_issue_refine"] = {
                "action": action,
                "reason": action_reason,
                "hard_candidate_count": hard_count,
                "policy": "DocShield/FakeShield-inspired issue rescue/downgrade using only existing candidates and final report text; no GT fields are read.",
            }
            out["stage_outputs"] = stage_outputs
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
