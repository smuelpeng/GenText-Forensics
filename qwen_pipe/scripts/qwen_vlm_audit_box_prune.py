#!/usr/bin/env python3
"""Prune or remap individual anomaly boxes using a GT-blind VLM crop audit.

This script does not reject a whole sample.  It operates only when at least
one sibling box in the same report was confirmed YES.  All-NO samples are left
unchanged because prior diagnostics showed that semantic/layout forgeries may
have no hard visual crop evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_bad_box_prune import BLOCK_RE, iter_grounding_entries  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path  # noqa: E402


def setup_debug_import(debug_root: Path) -> None:
    docshield_dir = debug_root / "baselines" / "DocShield"
    if not docshield_dir.exists():
        raise SystemExit(f"DocShield helper directory not found: {docshield_dir}")
    if str(docshield_dir) not in sys.path:
        sys.path.insert(0, str(docshield_dir))


def read_audit(path: Path, min_yes_conf: float, max_no_conf: float | None) -> dict[str, dict[int, dict[str, Any]]]:
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("sample_id") or "")
        per_index: dict[int, dict[str, Any]] = {}
        for box in row.get("boxes") or []:
            try:
                idx = int(box.get("index"))
            except (TypeError, ValueError):
                continue
            verdict = str(box.get("verdict") or "").upper()
            try:
                conf = float(box.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            keep_yes = verdict == "YES" and conf >= min_yes_conf
            keep_no = verdict == "NO" and (max_no_conf is None or conf <= max_no_conf)
            per_index[idx] = {**box, "is_yes": keep_yes, "is_no": keep_no, "confidence_float": conf}
        rows[sid] = per_index
    return rows


def remove_blocks(report: str, remove_indices: set[int]) -> tuple[str, int]:
    if not remove_indices:
        return report, 0
    entries = iter_grounding_entries(report)
    spans: list[tuple[int, int]] = []
    for entry in entries:
        if int(entry["index"]) not in remove_indices:
            continue
        block_span = entry.get("block_span")
        if block_span:
            spans.append(tuple(block_span))
    if not spans:
        return report, 0

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    out_parts: list[str] = []
    cursor = 0
    removed = 0
    for start, end in merged:
        out_parts.append(report[cursor:start])
        cursor = end
        removed += 1
    out_parts.append(report[cursor:])
    new_report = "".join(out_parts)
    new_report = re.sub(r"\n{4,}", "\n\n\n", new_report).strip() + "\n"
    return new_report, removed


def remap_no_to_yes(report: str, replacements: dict[int, list[int]]) -> tuple[str, int]:
    if not replacements:
        return report, 0
    cur = 0
    changed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal cur, changed
        idx = cur
        cur += 1
        box = replacements.get(idx)
        if not box:
            return match.group(0)
        changed += 1
        return f"{match.group(1)}{box}"

    return re.sub(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", repl, report, flags=re.IGNORECASE), changed


def bbox_area(box: list[int] | tuple[int, ...] | None) -> int:
    if not box or len(box) < 4:
        return 0
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def bbox_center_y(box: list[int] | tuple[int, ...] | None) -> float:
    if not box or len(box) < 4:
        return 0.0
    return (float(box[1]) + float(box[3])) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--diagnostic-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(PIPE_ROOT.parent / "debug_distribution"))
    parser.add_argument("--min-yes-conf", type=float, default=0.5)
    parser.add_argument("--max-no-conf", type=float, default=1.0)
    parser.add_argument("--min-yes-boxes", type=int, default=1)
    parser.add_argument("--max-remove-per-sample", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=["remove_blocks", "remap_no_to_best_yes"],
        default="remove_blocks",
        help="remove_blocks drops the NO anomaly blocks; remap_no_to_best_yes preserves report text and changes only coordinates.",
    )
    parser.add_argument(
        "--include-language",
        action="append",
        default=[],
        help="Optional OCR-layout document_language whitelist. Can be repeated.",
    )
    parser.add_argument("--min-report-boxes", type=int, default=0, help="Skip reports with fewer parsed grounding boxes.")
    parser.add_argument(
        "--max-remap-area-ratio",
        type=float,
        default=0.0,
        help=(
            "For remap_no_to_best_yes, skip a NO box if its area is more than this "
            "multiple of the selected YES box area. 0 disables the guard."
        ),
    )
    parser.add_argument(
        "--max-remap-y-center-delta",
        type=float,
        default=0.0,
        help=(
            "For remap_no_to_best_yes, skip a NO box if its vertical center is farther "
            "than this many pixels from the selected YES box center. 0 disables the guard."
        ),
    )
    return parser.parse_args()


def row_language(row: dict[str, Any]) -> str:
    layout = ((row.get("stage_outputs") or {}).get("ocr_layout") or {})
    parsed = layout.get("parsed") if isinstance(layout, dict) else {}
    if isinstance(parsed, dict):
        lang = str(parsed.get("document_language") or "").strip().lower()
        if lang:
            return lang
    return str(row.get("language_code") or "").strip().lower()


def main() -> None:
    args = parse_args()
    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    audit = read_audit(resolve_pipe_path(args.audit_jsonl), args.min_yes_conf, args.max_no_conf)
    out_path = resolve_pipe_path(args.output_jsonl)
    diag_path = resolve_pipe_path(args.diagnostic_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows": 0,
        "rows_with_audit": 0,
        "rows_changed": 0,
        "blocks_removed": 0,
        "skipped_all_no": 0,
        "skipped_no_yes": 0,
    }
    diag_rows: list[dict[str, Any]] = []
    with resolve_pipe_path(args.input_jsonl).open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            sid = str(row.get("sample_id") or Path(str(row.get("image_name") or "")).stem)
            per_index = audit.get(sid)
            out = dict(row)
            if not per_index:
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue
            stats["rows_with_audit"] += 1
            lang = row_language(row)
            if args.include_language and lang not in {v.lower() for v in args.include_language}:
                stats["skipped_language"] = stats.get("skipped_language", 0) + 1
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue
            if args.min_report_boxes > 0 and len(iter_grounding_entries(str(row.get("raw_output") or ""))) < args.min_report_boxes:
                stats["skipped_min_report_boxes"] = stats.get("skipped_min_report_boxes", 0) + 1
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue
            yes_indices = {idx for idx, box in per_index.items() if box.get("is_yes")}
            no_indices = [idx for idx, box in sorted(per_index.items()) if box.get("is_no")]
            if not yes_indices:
                stats["skipped_all_no"] += 1
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue
            if len(yes_indices) < args.min_yes_boxes:
                stats["skipped_no_yes"] += 1
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue
            report = str(row.get("raw_output") or "")
            if args.mode == "remove_blocks":
                remove_indices = set(no_indices[: args.max_remove_per_sample])
                new_report, changed = remove_blocks(report, remove_indices)
                removed = changed
                remapped = 0
                best_yes_box = None
                area_rejected: list[dict[str, Any]] = []
            else:
                yes_boxes = sorted(
                    (box for idx, box in per_index.items() if idx in yes_indices),
                    key=lambda box: float(box.get("confidence_float") or 0.0),
                    reverse=True,
                )
                if not yes_boxes:
                    dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                    continue
                best_yes_box = [int(v) for v in yes_boxes[0].get("box")]
                yes_area = max(1, bbox_area(best_yes_box))
                yes_center_y = bbox_center_y(best_yes_box)
                remove_indices_list: list[int] = []
                area_rejected = []
                y_rejected: list[dict[str, Any]] = []
                for idx in no_indices:
                    no_box = per_index.get(idx, {}).get("box")
                    no_area = bbox_area(no_box)
                    area_ratio = no_area / yes_area
                    if args.max_remap_area_ratio > 0 and area_ratio > args.max_remap_area_ratio:
                        area_rejected.append(
                            {
                                "index": idx,
                                "box": no_box,
                                "area": no_area,
                                "best_yes_box": best_yes_box,
                                "best_yes_area": yes_area,
                                "area_ratio": area_ratio,
                            }
                        )
                        continue
                    y_delta = abs(bbox_center_y(no_box) - yes_center_y)
                    if args.max_remap_y_center_delta > 0 and y_delta > args.max_remap_y_center_delta:
                        y_rejected.append(
                            {
                                "index": idx,
                                "box": no_box,
                                "best_yes_box": best_yes_box,
                                "y_delta": y_delta,
                            }
                        )
                        continue
                    remove_indices_list.append(idx)
                    if len(remove_indices_list) >= args.max_remove_per_sample:
                        break
                if area_rejected:
                    stats["skipped_area_ratio"] = stats.get("skipped_area_ratio", 0) + len(area_rejected)
                if y_rejected:
                    stats["skipped_y_delta"] = stats.get("skipped_y_delta", 0) + len(y_rejected)
                remove_indices = set(remove_indices_list)
                if not remove_indices:
                    dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                    continue
                new_report, changed = remap_no_to_yes(report, {idx: best_yes_box for idx in remove_indices})
                removed = 0
                remapped = changed
            if changed and new_report != report:
                out["raw_output"] = new_report
                out["parsed"] = parse_cct_report(new_report)
                stage_outputs = dict(out.get("stage_outputs") or {})
                stage_outputs["qwen_pipe_vlm_audit_box_prune"] = {
                    "applied": True,
                    "mode": args.mode,
                    "yes_indices": sorted(yes_indices),
                    "changed_indices": sorted(remove_indices),
                    "max_remap_area_ratio": args.max_remap_area_ratio,
                    "max_remap_y_center_delta": args.max_remap_y_center_delta,
                    "area_rejected": area_rejected,
                    "y_rejected": y_rejected,
                    "policy": (
                        "Operate on NO-audited boxes only when sibling YES-audited boxes exist; "
                        "never reject all-NO samples; optional area-ratio and vertical-band guards "
                        "prevent broad or cross-section context boxes from snapping to confirmed artifacts."
                    ),
                }
                out["stage_outputs"] = stage_outputs
                stats["rows_changed"] += 1
                stats["blocks_removed"] += removed
                stats["boxes_remapped"] = stats.get("boxes_remapped", 0) + remapped
                diag_rows.append(
                    {
                        "sample_id": sid,
                        "mode": args.mode,
                        "yes_indices": sorted(yes_indices),
                        "changed_indices": sorted(remove_indices),
                        "removed_blocks": removed,
                        "remapped_boxes": remapped,
                        "best_yes_box": best_yes_box,
                        "area_rejected": area_rejected,
                        "y_rejected": y_rejected,
                        "no_evidence": [per_index[i].get("evidence") for i in sorted(remove_indices) if i in per_index],
                    }
                )
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
    diag_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in diag_rows) + ("\n" if diag_rows else ""), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
