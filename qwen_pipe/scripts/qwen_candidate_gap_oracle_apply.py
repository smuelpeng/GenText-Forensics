#!/usr/bin/env python3
"""Apply candidate-gap broad-diagnosis best replacements as an upper bound.

This script is explicitly diagnostic, not deployable: it consumes a broad
candidate diagnosis whose best candidate was chosen using local GT evaluation.
It quantifies whether narrow-trigger broad search is worth engineering further.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_candidate_delta_model import replace_groundings  # noqa: E402
from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--broad-diag-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    diag = json.loads(resolve_pipe_path(args.broad_diag_json).read_text(encoding="utf-8"))
    best_by_sid: dict[str, dict[str, Any]] = {}
    for row in diag.get("rows") or []:
        best = row.get("best") or {}
        delta = float(best.get("delta") or 0.0)
        if delta > args.min_delta:
            best_by_sid[str(row.get("sample_id") or "")] = best

    out_path = resolve_pipe_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    changed: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for row in read_jsonl(resolve_pipe_path(args.input_jsonl)):
            sid = sample_id_from_row(row)
            best = best_by_sid.get(sid)
            if best:
                replace_index = int(best.get("replace_index"))
                box = [int(v) for v in best.get("box") or []]
                if len(box) >= 4:
                    row = dict(row)
                    report, changed_count = replace_groundings(str(row.get("raw_output") or ""), {replace_index: box[:4]})
                    if changed_count:
                        row["raw_output"] = report
                        row["parsed"] = parse_cct_report(report)
                        stage_outputs = dict(row.get("stage_outputs") or {})
                        stage_outputs["qwen_pipe_candidate_gap_oracle_apply"] = {
                            "applied": True,
                            "diagnostic_only": True,
                            "replace_index": replace_index,
                            "candidate": best,
                            "policy": "Diagnostic upper bound: best broad candidate selected by local GT delta, never deploy as inference logic.",
                        }
                        row["stage_outputs"] = stage_outputs
                        changed.append({"sample_id": sid, "best": best})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(resolve_pipe_path(args.input_jsonl)),
        "broad_diag_json": str(resolve_pipe_path(args.broad_diag_json)),
        "output_jsonl": str(out_path),
        "min_delta": args.min_delta,
        "candidate_samples": len(best_by_sid),
        "changed_count": len(changed),
        "changed": changed,
        "diagnostic_only": True,
    }
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
