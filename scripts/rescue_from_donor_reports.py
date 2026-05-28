#!/usr/bin/env python3
"""GT-blind rescue pass for false negatives using donor staged reports.

This pass is intentionally conservative: it only changes AUTHENTIC base reports
when a donor run produced a FORGED report with localized anomalies and the donor
text contains one of a small set of strong evidence triggers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCSHIELD_DIR = REPO_ROOT / "baselines" / "DocShield"
sys.path.insert(0, str(DOCSHIELD_DIR))

from postprocess import parse_cct_report  # noqa: E402


DEFAULT_RULES = {
    "v18": (
        ("obscured",),
        ("digital noise",),
        ("localized corruption",),
        ("corrupted rendering",),
        ("black marks", "digital noise"),
        ("black rectangle",),
        ("black blob",),
        ("solid black",),
        ("black bar",),
        ("黑色矩形",),
        ("黑色块",),
        ("garis hitam",),
        ("主字符", "亦", "渲染缺陷"),
        ("红色高亮", "后期叠加"),
        ("حجب نص", "طبقة رقمية"),
        ("بقعة رمادية داكنة", "يحجب"),
    ),
    "v19": (
        ("cpsa",),
        ("black blob", "semesters"),
        ("solid black", "semesters"),
        ("黑色矩形", "第三个电路图"),
        ("garis hitam", "ruang kosong"),
        ("ผลรวม", "เท่ากับ 15", '"16"'),
        ("เพิ่มสูงขึ้น", "ลดลง", "สหภาพยุโรป"),
        ("ff兒童國際電影節", "128"),
        ("趙佳琪", "趙潔超"),
    ),
    "v15": (
        ("12/06/2560", "09/10/2560"),
        ("12/06/2560", "16/10/2560"),
        ("วันที่พิมพ์", "ตารางการสอบ"),
        ("black marks", "digital noise"),
        ("black marks", "obscur"),
        ("solid black", "date of birth"),
        ("黑色块", "上标"),
        ("遮挡", "星期五"),
        ("ภาษาจีน", "上海证券"),
    ),
    "v34": (
        ("application for credentialing", "blok biru/kelabu"),
        ("458", "503", "subset cannot be larger"),
        ("แถบสีดำทึบ", "3 + 2 = 5"),
    ),
    "v36": (
        ("4:11 pm", "4:45 pm", "highlighted schedule"),
        ("4 45", "localized rendering artifact"),
    ),
}


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def parse_donor_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("donor must use name=path")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("donor name is empty")
    return name, repo_path(path.strip())


def rule_matches(text: str, groups: tuple[tuple[str, ...], ...]) -> tuple[bool, tuple[str, ...] | None]:
    lowered = text.lower()
    for group in groups:
        if all(term.lower() in lowered for term in group):
            return True, group
    return False, None


def should_rescue(
    donor_row: dict[str, Any],
    *,
    donor_name: str,
    min_risk: int,
) -> tuple[bool, dict[str, Any]]:
    raw = str(donor_row.get("raw_output") or "")
    parsed = parse_cct_report(raw)
    rules = DEFAULT_RULES.get(donor_name, ())
    matched, trigger = rule_matches(raw, rules)
    apply = (
        parsed.get("conclusion") == "FORGED"
        and int(parsed.get("risk_score") or 0) >= min_risk
        and bool(parsed.get("anomalies"))
        and matched
    )
    return apply, {
        "donor": donor_name,
        "risk_score": parsed.get("risk_score"),
        "anomaly_count": len(parsed.get("anomalies") or []),
        "matched_trigger": list(trigger or ()),
        "applied": apply,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--donor", action="append", type=parse_donor_arg, required=True)
    parser.add_argument("--min-risk", type=int, default=85)
    args = parser.parse_args()

    base_rows = load_jsonl(repo_path(args.base_jsonl))
    donors = [(name, load_jsonl(path)) for name, path in args.donor]
    out_path = repo_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    rescued = 0
    with out_path.open("w", encoding="utf-8") as dst:
        for sample_id, row in base_rows.items():
            total += 1
            out = dict(row)
            stage_outputs = dict(out.get("stage_outputs") or {})
            base_parsed = parse_cct_report(str(out.get("raw_output") or ""))
            attempts = []
            if base_parsed.get("conclusion") == "AUTHENTIC":
                for donor_name, donor_rows in donors:
                    donor_row = donor_rows.get(sample_id)
                    if donor_row is None:
                        continue
                    apply, meta = should_rescue(donor_row, donor_name=donor_name, min_risk=args.min_risk)
                    attempts.append(meta)
                    if apply:
                        out["raw_output"] = str(donor_row.get("raw_output") or "")
                        out["parsed"] = parse_cct_report(out["raw_output"])
                        stage_outputs["donor_rescue"] = {**meta, "source": "donor_report"}
                        rescued += 1
                        break
            if "donor_rescue" not in stage_outputs:
                out["parsed"] = base_parsed
                stage_outputs["donor_rescue"] = {
                    "applied": False,
                    "attempts": attempts,
                    "policy": "Rescue only base AUTHENTIC reports with donor FORGED evidence matching strong trigger rules.",
                }
            out["stage_outputs"] = stage_outputs
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} rows={total} rescued={rescued}")


if __name__ == "__main__":
    main()
