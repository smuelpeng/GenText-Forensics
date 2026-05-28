#!/usr/bin/env python3
"""Map crop-level donor report coordinates back to the original image space."""

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


GROUNDING_RE = re.compile(
    r"(\[GROUNDING\]\s*:\s*)\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
    re.IGNORECASE,
)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def remap_box(
    box: list[int | float],
    *,
    offset_x: float,
    offset_y: float,
    scale_x: float,
    scale_y: float,
    max_width: int | None,
    max_height: int | None,
) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    mapped = [
        round(x1 * scale_x + offset_x),
        round(y1 * scale_y + offset_y),
        round(x2 * scale_x + offset_x),
        round(y2 * scale_y + offset_y),
    ]
    if max_width is not None:
        mapped[0] = max(0, min(max_width, mapped[0]))
        mapped[2] = max(0, min(max_width, mapped[2]))
    if max_height is not None:
        mapped[1] = max(0, min(max_height, mapped[1]))
        mapped[3] = max(0, min(max_height, mapped[3]))
    return mapped


def remap_raw_output(raw: str, **kwargs: Any) -> str:
    def replace(match: re.Match[str]) -> str:
        box = [float(match.group(i)) for i in range(2, 6)]
        mapped = remap_box(box, **kwargs)
        return f"{match.group(1)}[{mapped[0]}, {mapped[1]}, {mapped[2]}, {mapped[3]}]"

    return GROUNDING_RE.sub(replace, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--sample-id", required=True, help="Original sample id to assign to remapped rows.")
    parser.add_argument("--offset-x", type=float, default=0.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--scale-x", type=float, default=1.0)
    parser.add_argument("--scale-y", type=float, default=1.0)
    parser.add_argument("--max-width", type=int)
    parser.add_argument("--max-height", type=int)
    args = parser.parse_args()

    in_path = repo_path(args.input_jsonl)
    out_path = repo_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            raw = remap_raw_output(
                str(row.get("raw_output") or ""),
                offset_x=args.offset_x,
                offset_y=args.offset_y,
                scale_x=args.scale_x,
                scale_y=args.scale_y,
                max_width=args.max_width,
                max_height=args.max_height,
            )
            stage_outputs = dict(row.get("stage_outputs") or {})
            stage_outputs["crop_coordinate_remap"] = {
                "source_sample_id": row.get("sample_id"),
                "target_sample_id": args.sample_id,
                "offset_x": args.offset_x,
                "offset_y": args.offset_y,
                "scale_x": args.scale_x,
                "scale_y": args.scale_y,
            }
            out = dict(row)
            out["sample_id"] = args.sample_id
            out["raw_output"] = raw
            out["parsed"] = parse_cct_report(raw)
            out["stage_outputs"] = stage_outputs
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows += 1
    print(f"wrote {out_path} rows={rows}")


if __name__ == "__main__":
    main()
