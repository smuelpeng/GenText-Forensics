#!/usr/bin/env python3
"""Targeted broad-candidate diagnosis for lowloc samples with no positive pair.

This is an evaluation-side diagnostic only.  It uses GT masks to check whether
expanding top-k/candidate-limit would create positive replacement candidates for
samples previously classified as candidate gaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
DEFAULT_GT = DEFAULT_DEBUG_ROOT / "data/val_300.jsonl"
DEFAULT_OCR_LAYOUT_CACHE = DEFAULT_DEBUG_ROOT / "outputs/cache/ocr_layouts"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import (  # noqa: E402
    choose_topk_v145ocranchor,
    choose_topk_v87mix,
    choose_topk_v89token,
    generate_candidates,
    mask_stats,
    read_gt_mask,
    read_jsonl,
    sample_id_from_row,
    select_apply_extras,
)
from qwen_pair_replace_model import action_delta, language_code, load_gt_rows  # noqa: E402
from qwen_text_crop_verify import resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402


def candidate_brief(cand: dict[str, Any], delta: float, replace_index: int) -> dict[str, Any]:
    meta = cand.get("meta") or {}
    rank = meta.get("v145ocranchor_rank") or meta.get("v87b_rank") or meta.get("v87_rank") or {}
    return {
        "delta": delta,
        "replace_index": replace_index,
        "family": cand.get("family"),
        "source": cand.get("source"),
        "box": cand.get("box"),
        "text": str(cand.get("text") or "")[:180],
        "rank_score": rank.get("score"),
        "query_score": rank.get("query_score"),
        "token_hit_count": rank.get("token_hit_count"),
        "number_hits": rank.get("number_hits"),
        "visual_hits": rank.get("visual_hits"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-jsonl", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--gt-jsonl", default=str(DEFAULT_GT))
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--ocr-layout-cache-dir", default=str(DEFAULT_OCR_LAYOUT_CACHE))
    parser.add_argument("--ocr-layout-model", default="qwen-vl-ocr")
    parser.add_argument("--coord-mode", choices=["normalized-1000", "pixel", "auto"], default="normalized-1000")
    parser.add_argument("--rank-mode", choices=["v145ocranchor", "v87mix", "v89token"], default="v145ocranchor")
    parser.add_argument("--max-candidates", type=int, default=2500)
    parser.add_argument("--top-k", type=int, default=140)
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument("--apply-max-area-ratio", type=float, default=0.08)
    parser.add_argument("--candidate-duplicate-iou", type=float, default=0.50)
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    coverage = json.loads(resolve_pipe_path(args.coverage_json).read_text(encoding="utf-8"))
    target_ids = [
        str(r.get("sample_id") or "")
        for r in coverage.get("all_rows") or []
        if r.get("bucket") == "candidate_gap_no_positive_pair"
    ]
    raw_rows = {sample_id_from_row(r): r for r in read_jsonl(resolve_pipe_path(args.raw_jsonl))}
    gt_rows = load_gt_rows(args.gt_jsonl, debug_root)
    rows: list[dict[str, Any]] = []

    for sid in target_ids:
        row = raw_rows.get(sid)
        if not row:
            continue
        image_path = resolve_image_path(row, debug_root)
        if not image_path:
            rows.append({"sample_id": sid, "error": "missing_image"})
            continue
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        width, height = image.size
        existing = report_boxes(str(row.get("raw_output") or ""))
        candidates = generate_candidates(
            row,
            image,
            debug_root,
            Path(args.ocr_layout_cache_dir).expanduser().resolve(),
            args.ocr_layout_model,
            args.coord_mode,
            args.max_candidates,
            enable_token_candidates=True,
            enable_linegrid_candidates=True,
            enable_scriptgrid_candidates=True,
            language_code=language_code(row),
        )
        if args.rank_mode == "v145ocranchor":
            top = choose_topk_v145ocranchor(row, candidates, args.top_k, width, height, language_code=language_code(row))
        elif args.rank_mode == "v87mix":
            top = choose_topk_v87mix(row, candidates, args.top_k, width, height)
        else:
            top = choose_topk_v89token(row, candidates, args.top_k, width, height)
        top_dicts = [
            {
                "label": c.label,
                "box": c.box,
                "source": c.source,
                "family": c.family,
                "score": c.score,
                "text": c.text,
                "meta": c.meta,
            }
            for c in top
        ]
        extras = select_apply_extras(
            top_dicts,
            existing,
            width=width,
            height=height,
            limit=args.candidate_limit,
            max_area_ratio=args.apply_max_area_ratio,
            duplicate_iou=args.candidate_duplicate_iou,
        )
        gt_row = gt_rows.get(sid)
        gt_mask = read_gt_mask(gt_row, width, height, debug_root) if gt_row else None
        base_f1 = float(mask_stats(gt_mask, existing, width, height).get("mask_f1") or 0.0) if gt_mask is not None else 0.0
        scored: list[dict[str, Any]] = []
        for cand in extras:
            box = [int(v) for v in cand.get("box") or []]
            if len(box) < 4:
                continue
            for replace_index in range(len(existing)):
                delta, _new_f1, _action = action_delta(
                    gt_mask=gt_mask,
                    existing=existing,
                    candidate_box=box,
                    replace_index=replace_index,
                    width=width,
                    height=height,
                    base_f1=base_f1,
                )
                scored.append(candidate_brief(cand, float(delta), replace_index))
        scored.sort(key=lambda x: float(x.get("delta") or 0.0), reverse=True)
        rows.append(
            {
                "sample_id": sid,
                "language_code": language_code(row),
                "existing_count": len(existing),
                "candidate_count": len(candidates),
                "top_count": len(top),
                "extra_count": len(extras),
                "base_f1": base_f1,
                "positive_count": sum(1 for item in scored if float(item.get("delta") or 0.0) > 0.0),
                "best_delta": float(scored[0].get("delta") or 0.0) if scored else 0.0,
                "best": scored[0] if scored else None,
                "top_positive": [item for item in scored if float(item.get("delta") or 0.0) > 0.0][:10],
            }
        )

    summary = {
        "raw_jsonl": str(resolve_pipe_path(args.raw_jsonl)),
        "coverage_json": str(resolve_pipe_path(args.coverage_json)),
        "target_count": len(target_ids),
        "covered_positive_count": sum(1 for r in rows if float(r.get("best_delta") or 0.0) > 0.0),
        "family_counts_best_positive": dict(Counter(str((r.get("best") or {}).get("family") or "") for r in rows if float(r.get("best_delta") or 0.0) > 0.0)),
        "rows": rows,
        "params": {
            "max_candidates": args.max_candidates,
            "top_k": args.top_k,
            "candidate_limit": args.candidate_limit,
            "apply_max_area_ratio": args.apply_max_area_ratio,
            "rank_mode": args.rank_mode,
        },
    }
    out_path = resolve_pipe_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["target_count", "covered_positive_count", "family_counts_best_positive"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
