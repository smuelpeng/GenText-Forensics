#!/usr/bin/env python3
"""Route only grounding boxes from a donor raw variant by language.

Full language routing can improve localization but hurt detection/reporting if
the donor variant changes the verdict or explanation.  This script keeps the
default report text and verdict, and only replaces its [GROUNDING] coordinates
with boxes from a donor variant for selected languages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_candidate_delta_model import replace_groundings  # noqa: E402
from qwen_evidence_multibox_refine import report_boxes  # noqa: E402


def box_area(box: list[int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def mean_area_ratio(default_boxes: list[list[int]], donor_boxes: list[list[int]], limit: int) -> float:
    if limit <= 0:
        return 0.0
    ratios = [box_area(donor_boxes[idx]) / max(1.0, box_area(default_boxes[idx])) for idx in range(limit)]
    return sum(ratios) / len(ratios)


def resolve_pipe_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PIPE_ROOT / p


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)


def load_languages(eval_json: Path) -> dict[str, str]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for sample in data.get("samples") or []:
        sid = str(sample.get("sample_id") or "")
        if sid:
            out[sid] = str(sample.get("language_code") or "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-raw", required=True)
    parser.add_argument("--donor-raw", required=True)
    parser.add_argument("--eval-json", required=True, help="Read language_code only; no GT scores are used.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--route-language", action="append", default=[])
    parser.add_argument("--mode", choices=["same_count", "prefix"], default="same_count")
    parser.add_argument(
        "--max-mean-area-ratio",
        type=float,
        default=0.0,
        help="Optional geometry acceptor: only route if donor/default mean area ratio over replaced prefix is <= this value. 0 disables.",
    )
    parser.add_argument("--stage-name", default="qwen_pipe_language_box_variant_route")
    parser.add_argument("--default-name", default="default")
    parser.add_argument("--donor-name", default="donor")
    parser.add_argument(
        "--protect-stage-key",
        action="append",
        default=[],
        help=(
            "Skip donor routing when this stage output already exists on the default row. "
            "Use to preserve stronger local validators such as crop-audit remaps."
        ),
    )
    args = parser.parse_args()

    default_path = resolve_pipe_path(args.default_raw)
    donor_path = resolve_pipe_path(args.donor_raw)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    languages = load_languages(resolve_pipe_path(args.eval_json))
    donor_rows = {sample_id(row): row for row in read_jsonl(donor_path)}
    route_langs = set(args.route_language or [])

    stats = {
        "rows": 0,
        "routed_language_rows": 0,
        "changed": 0,
        "skipped_count_mismatch": 0,
        "skipped_no_donor_boxes": 0,
        "skipped_area_ratio": 0,
        "route_language": sorted(route_langs),
        "mode": args.mode,
        "max_mean_area_ratio": args.max_mean_area_ratio,
        "protect_stage_key": sorted(args.protect_stage_key),
    }
    with default_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            sid = sample_id(row)
            lang = languages.get(sid) or str(row.get("language_code") or ((row.get("metadata") or {}).get("language_code")) or "")
            selected = args.default_name
            donor_boxes: list[list[int]] = []
            default_boxes = report_boxes(str(row.get("raw_output") or ""))
            stage_outputs = dict(row.get("stage_outputs") or {})
            protected_keys = [key for key in args.protect_stage_key if key in stage_outputs]
            if lang in route_langs and sid in donor_rows:
                stats["routed_language_rows"] += 1
                if protected_keys:
                    stats["skipped_protected_stage"] = stats.get("skipped_protected_stage", 0) + 1
                else:
                    donor_boxes = report_boxes(str(donor_rows[sid].get("raw_output") or ""))
                if protected_keys:
                    pass
                elif not donor_boxes:
                    stats["skipped_no_donor_boxes"] += 1
                elif args.mode == "same_count" and len(donor_boxes) != len(default_boxes):
                    stats["skipped_count_mismatch"] += 1
                else:
                    limit = min(len(default_boxes), len(donor_boxes))
                    ratio = mean_area_ratio(default_boxes, donor_boxes, limit)
                    if args.max_mean_area_ratio > 0 and ratio > args.max_mean_area_ratio:
                        stats["skipped_area_ratio"] += 1
                    else:
                        replacements = {idx: donor_boxes[idx] for idx in range(limit)}
                        new_report, replaced = replace_groundings(str(row.get("raw_output") or ""), replacements)
                        if replaced:
                            row = dict(row)
                            row["raw_output"] = new_report
                            selected = args.donor_name
                            stats["changed"] += 1
            stage_outputs[args.stage_name] = {
                "selected_variant": selected,
                "language_code": lang,
                "route_language": sorted(route_langs),
                "mode": args.mode,
                "protected_stage_keys": protected_keys,
                "default_raw": str(default_path),
                "donor_raw": str(donor_path),
                "default_box_count": len(default_boxes),
                "donor_box_count": len(donor_boxes),
                "mean_area_ratio": mean_area_ratio(default_boxes, donor_boxes, min(len(default_boxes), len(donor_boxes))),
                "policy": "GT-blind language route that copies only donor grounding coordinates while preserving default verdict and report text.",
            }
            row["stage_outputs"] = stage_outputs
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["output_jsonl"] = str(output_path)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
