#!/usr/bin/env python3
"""GT-blind grounding-box graft for staged DocShield outputs.

This reproduces the v13 diagnostic pattern: keep the base detector/report
verdict, but replace final report [GROUNDING] boxes with boxes from a separate
OCR-assisted grounding run when both runs independently predict FORGED.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCSHIELD_DIR = REPO_ROOT / "baselines" / "DocShield"
sys.path.insert(0, str(DOCSHIELD_DIR))

from postprocess import parse_cct_report  # noqa: E402


GROUNDING_RE = re.compile(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", re.IGNORECASE)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("image_name") or Path(str(row.get("image_path") or "")).stem)


def conclusion(row: dict[str, Any]) -> str:
    parsed = row.get("parsed") or {}
    if parsed.get("conclusion"):
        return str(parsed.get("conclusion")).upper()
    return str(parse_cct_report(str(row.get("raw_output") or "")).get("conclusion") or "UNKNOWN").upper()


def grounding_literals(report: str) -> list[str]:
    boxes: list[str] = []
    for match in GROUNDING_RE.finditer(report or ""):
        boxes.append(f"{match.group(1)}[{match.group(2)}]")
    return boxes


def graft_boxes(base_report: str, donor_report: str, *, mode: str) -> tuple[str, int, int, int]:
    donor_boxes = grounding_literals(donor_report)
    donor_idx = 0
    replaced = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal donor_idx, replaced
        if donor_idx >= len(donor_boxes):
            return match.group(0)
        box = donor_boxes[donor_idx]
        donor_idx += 1
        replaced += 1
        return box

    if mode == "none":
        return base_report, 0, len(grounding_literals(base_report)), len(donor_boxes)

    if mode == "replace_sequential":
        return GROUNDING_RE.sub(replace, base_report), replaced, len(grounding_literals(base_report)), len(donor_boxes)

    if mode == "replace_all":
        if not donor_boxes:
            return base_report, 0, len(grounding_literals(base_report)), 0
        return GROUNDING_RE.sub(replace, base_report), replaced, len(grounding_literals(base_report)), len(donor_boxes)

    raise ValueError(f"Unknown graft mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", required=True, help="Base detector/report JSONL.")
    parser.add_argument("--grounding-jsonl", required=True, help="Donor JSONL providing replacement boxes.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--mode",
        choices=["replace_sequential", "replace_all", "none"],
        default="replace_sequential",
        help="replace_sequential replaces up to min(base boxes, donor boxes), preserving extra base boxes.",
    )
    parser.add_argument(
        "--require-both-forged",
        action="store_true",
        help="Only graft boxes when both base and donor reports predict FORGED.",
    )
    args = parser.parse_args()

    base_rows = read_jsonl(repo_path(args.base_jsonl))
    donor_rows = {row_key(row): row for row in read_jsonl(repo_path(args.grounding_jsonl))}
    out_path = repo_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grafted = 0
    replaced_total = 0
    missing_donor = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in base_rows:
            key = row_key(row)
            donor = donor_rows.get(key)
            if not donor:
                missing_donor += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            should_graft = True
            if args.require_both_forged:
                should_graft = conclusion(row) == "FORGED" and conclusion(donor) == "FORGED"

            out = dict(row)
            stage_outputs = dict(out.get("stage_outputs") or {})
            if should_graft:
                new_report, replaced, base_count, donor_count = graft_boxes(
                    str(row.get("raw_output") or ""),
                    str(donor.get("raw_output") or ""),
                    mode=args.mode,
                )
                if replaced:
                    out["raw_output"] = new_report
                    out["parsed"] = parse_cct_report(new_report)
                    grafted += 1
                    replaced_total += replaced
                stage_outputs["grounding_graft"] = {
                    "donor_sample_id": row_key(donor),
                    "donor_model": donor.get("model"),
                    "donor_pipeline": donor.get("pipeline"),
                    "mode": args.mode,
                    "required_both_forged": args.require_both_forged,
                    "applied": bool(replaced),
                    "replaced_boxes": replaced,
                    "base_boxes": base_count,
                    "donor_boxes": donor_count,
                    "base_verdict": conclusion(row),
                    "donor_verdict": conclusion(donor),
                }
                out["stage_outputs"] = stage_outputs
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        f"wrote {out_path} rows={len(base_rows)} grafted={grafted} "
        f"replaced_boxes={replaced_total} missing_donor={missing_donor}"
    )


if __name__ == "__main__":
    main()
