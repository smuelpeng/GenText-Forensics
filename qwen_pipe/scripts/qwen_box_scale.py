#!/usr/bin/env python3
"""GT-blind grounding box scaling for Qwen-pipe reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"


def setup_debug_import(debug_root: Path) -> None:
    docshield_dir = debug_root / "baselines" / "DocShield"
    if not docshield_dir.exists():
        raise SystemExit(f"DocShield helper directory not found: {docshield_dir}")
    sys.path.insert(0, str(docshield_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--scale-x", type=float, required=True)
    parser.add_argument("--scale-y", type=float, required=True)
    return parser.parse_args()


def resolve_pipe_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PIPE_ROOT / p


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)

    from postprocess import parse_cct_report  # type: ignore
    from run_staged_docshield_api import scale_grounding_boxes_in_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            total += 1
            width = int(row.get("width") or 0)
            height = int(row.get("height") or 0)
            old_report = str(row.get("raw_output") or "")
            new_report = old_report
            if width > 0 and height > 0:
                new_report = scale_grounding_boxes_in_report(
                    old_report,
                    width,
                    height,
                    args.scale_x,
                    args.scale_y,
                )
            out = dict(row)
            if new_report != old_report:
                changed += 1
                out["raw_output"] = new_report
                out["parsed"] = parse_cct_report(new_report)
            stage_outputs = dict(out.get("stage_outputs") or {})
            stage_outputs["qwen_pipe_box_scale"] = {
                "scale_x": args.scale_x,
                "scale_y": args.scale_y,
                "changed": new_report != old_report,
                "policy": "Scale existing final report grounding boxes only; do not change verdicts or reasons.",
            }
            out["stage_outputs"] = stage_outputs
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(json.dumps({"rows": total, "changed": changed, "scale_x": args.scale_x, "scale_y": args.scale_y}))


if __name__ == "__main__":
    main()
