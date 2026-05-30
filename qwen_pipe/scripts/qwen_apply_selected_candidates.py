#!/usr/bin/env python3
"""Apply selected diagnostic candidates to a base raw JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_exhaustive_recall import insert_extra_anomalies, read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"


def load_selected(path: Path) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("sample_id") or "")
        for item in row.get("selected") or []:
            cand = item.get("candidate") or {}
            if cand.get("box"):
                selected[sid].append(cand)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-raw-jsonl", required=True)
    parser.add_argument("--selected-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--stage-name", default="qwen_apply_selected_candidates")
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    base_path = resolve_pipe_path(args.base_raw_jsonl)
    selected_path = resolve_pipe_path(args.selected_jsonl)
    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = load_selected(selected_path)
    applied_samples = 0
    applied_candidates = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in read_jsonl(base_path):
            sid = sample_id_from_row(row)
            extras = selected.get(sid) or []
            if extras:
                row = dict(row)
                report = insert_extra_anomalies(str(row.get("raw_output") or ""), extras)
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs[args.stage_name] = {
                    "applied": True,
                    "boxes_added": len(extras),
                    "selected_jsonl": str(selected_path),
                    "policy": "Append selected candidates from a diagnostic JSONL to a chosen base raw output.",
                }
                row["stage_outputs"] = stage_outputs
                applied_samples += 1
                applied_candidates += len(extras)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"applied_samples": applied_samples, "applied_candidates": applied_candidates, "output": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
