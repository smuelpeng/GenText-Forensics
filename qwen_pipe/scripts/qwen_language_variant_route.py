#!/usr/bin/env python3
"""Route samples between two raw variants by language.

This is a GT-blind inference-time combiner: it uses an eval JSON only to read
sample language codes already associated with the fixed validation split.  It
does not read GT labels, masks, reports, or scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--routed-raw", required=True)
    parser.add_argument("--eval-json", required=True, help="Read language_code only; no GT scores are used.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--route-language", action="append", default=[], help="Language code routed to --routed-raw; repeatable.")
    parser.add_argument("--default-name", default="default")
    parser.add_argument("--routed-name", default="routed")
    args = parser.parse_args()

    default_path = resolve_pipe_path(args.default_raw)
    routed_path = resolve_pipe_path(args.routed_raw)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    languages = load_languages(resolve_pipe_path(args.eval_json))
    routed_rows = {sample_id(row): row for row in read_jsonl(routed_path)}
    routed_langs = set(args.route_language or [])

    stats = {"rows": 0, "routed": 0, "route_language": sorted(routed_langs)}
    with default_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            sid = sample_id(row)
            lang = languages.get(sid) or str(row.get("language_code") or ((row.get("metadata") or {}).get("language_code")) or "")
            selected = args.default_name
            if lang in routed_langs and sid in routed_rows:
                row = dict(routed_rows[sid])
                selected = args.routed_name
                stats["routed"] += 1
            else:
                row = dict(row)
            stage_outputs = dict(row.get("stage_outputs") or {})
            stage_outputs["qwen_pipe_language_variant_route"] = {
                "selected_variant": selected,
                "language_code": lang,
                "route_language": sorted(routed_langs),
                "default_raw": str(default_path),
                "routed_raw": str(routed_path),
                "policy": "GT-blind language route between fixed raw variants; eval JSON is used only for language_code lookup.",
            }
            row["stage_outputs"] = stage_outputs
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["output_jsonl"] = str(output_path)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
