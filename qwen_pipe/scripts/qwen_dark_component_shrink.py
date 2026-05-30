#!/usr/bin/env python3
"""Shrink VLM-accepted obstruction boxes to dark connected components.

This is a GT-blind localization postprocess.  It targets the narrow lane that
worked in v463: OCR crop-verifier boxes whose evidence mentions black/dark
smudge, redaction, block, or obscuring.  The script replaces those broad OCR
boxes with image-derived dark component boxes and updates the report text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

PIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_evidence_multibox_refine import report_boxes  # noqa: E402
from qwen_exhaustive_recall import page_area_ratio, read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_image_path, resolve_pipe_path, setup_debug_import  # noqa: E402

GROUNDING_RE = re.compile(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", re.IGNORECASE)
OBSTRUCTION_RE = re.compile(r"black|dark|smudge|redaction|obscur|block", re.I)
REJECT_RE = re.compile(r"font substitution|substitution error|font.*error", re.I)


def parse_box_text(text: str) -> list[int] | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    if len(nums) < 4:
        return None
    try:
        box = [int(round(float(v))) for v in nums[:4]]
    except Exception:
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def clamp_box(box: list[int], width: int, height: int) -> list[int] | None:
    x1 = max(0, min(width - 1, int(box[0])))
    y1 = max(0, min(height - 1, int(box[1])))
    x2 = max(0, min(width, int(box[2])))
    y2 = max(0, min(height, int(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def replace_grounding(report: str, old_box: list[int], new_box: list[int]) -> tuple[str, bool]:
    replaced = False

    def repl(match: re.Match[str]) -> str:
        nonlocal replaced
        if replaced:
            return match.group(0)
        box = parse_box_text(match.group(2))
        if box == old_box:
            replaced = True
            return f"{match.group(1)}{new_box}"
        return match.group(0)

    return GROUNDING_RE.sub(repl, report), replaced


def grayscale_pixels(image: Image.Image) -> list[list[int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    data = list(rgb.getdata())
    rows: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            r, g, b = data[y * width + x]
            row.append(int(0.299 * r + 0.587 * g + 0.114 * b))
        rows.append(row)
    return rows


def connected_components(mask: list[list[bool]]) -> list[dict[str, Any]]:
    h = len(mask)
    w = len(mask[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    comps: list[dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or not mask[y][x]:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cx, cy = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if nx < 0 or ny < 0 or nx >= w or ny >= h:
                            continue
                        if seen[ny][nx] or not mask[ny][nx]:
                            continue
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            x1, x2 = min(xs), max(xs) + 1
            y1, y2 = min(ys), max(ys) + 1
            area = len(xs)
            box_area = max(1, (x2 - x1) * (y2 - y1))
            comps.append(
                {
                    "area": area,
                    "box": [x1, y1, x2, y2],
                    "fill": area / box_area,
                    "w": x2 - x1,
                    "h": y2 - y1,
                }
            )
    return comps


def choose_dark_box(
    image: Image.Image,
    box: list[int],
    *,
    mode: str,
    threshold: int,
    min_component_area: int,
    max_components: int,
    pad: int,
) -> tuple[list[int] | None, dict[str, Any]]:
    width, height = image.size
    box = clamp_box(box, width, height)
    if not box:
        return None, {"reason": "invalid_box"}
    crop = image.crop(tuple(box))
    gray = grayscale_pixels(crop)
    if not gray:
        return None, {"reason": "empty_crop"}
    mask = [[pix <= threshold for pix in row] for row in gray]
    comps = [
        c
        for c in connected_components(mask)
        if c["area"] >= min_component_area and c["w"] >= 2 and c["h"] >= 2
    ]
    if not comps:
        return None, {"reason": "no_components", "threshold": threshold}
    crop_w, crop_h = crop.size

    def score(c: dict[str, Any]) -> float:
        # Favor large, compact dark blobs; downweight tiny text-like marks.
        height_bonus = min(2.0, c["h"] / max(1, crop_h * 0.35))
        fill_bonus = min(1.8, c["fill"] * 3.0)
        return float(c["area"]) * (0.5 + 0.25 * height_bonus + 0.25 * fill_bonus)

    comps.sort(key=score, reverse=True)
    if mode == "largest":
        selected = comps[:1]
    elif mode == "top2":
        selected = comps[:2]
    elif mode == "top3":
        selected = comps[:3]
    else:
        selected = comps[:max_components]

    x1 = min(c["box"][0] for c in selected)
    y1 = min(c["box"][1] for c in selected)
    x2 = max(c["box"][2] for c in selected)
    y2 = max(c["box"][3] for c in selected)
    local = [max(0, x1 - pad), max(0, y1 - pad), min(crop_w, x2 + pad), min(crop_h, y2 + pad)]
    out = clamp_box([local[0] + box[0], local[1] + box[1], local[2] + box[0], local[3] + box[1]], width, height)
    return out, {"components": comps[:8], "selected": selected, "threshold": threshold, "mode": mode}


def accepted_targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    stage = (row.get("stage_outputs") or {}).get("qwen_exhaustive_crop_verifier") or {}
    out: list[dict[str, Any]] = []
    for item in stage.get("accepted") or []:
        source = str(item.get("source") or "")
        family = str(item.get("family") or "")
        text = str(item.get("text") or "")
        if not source.startswith("qwen_exhaustive_crop_verifier"):
            continue
        if family != "ocr":
            continue
        if not OBSTRUCTION_RE.search(text) or REJECT_RE.search(text):
            continue
        box = item.get("box")
        if not isinstance(box, list) or len(box) < 4:
            continue
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--diag-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--debug-root", default=str(PIPE_ROOT.parent / "debug_distribution"))
    parser.add_argument("--mode", choices=["largest", "top2", "top3"], default="largest")
    parser.add_argument("--threshold", type=int, default=85)
    parser.add_argument("--min-component-area", type=int, default=18)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--pad", type=int, default=3)
    parser.add_argument("--max-shrink-ratio", type=float, default=0.75)
    args = parser.parse_args()

    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)
    rows = read_jsonl(resolve_pipe_path(args.input_jsonl))
    out_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for row in rows:
        stats["rows"] += 1
        sid = sample_id_from_row(row)
        targets = accepted_targets(row)
        if not targets:
            out_rows.append(row)
            continue
        image_path = resolve_image_path(row, debug_root)
        if not image_path:
            stats["missing_image"] += 1
            out_rows.append(row)
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        report = str(row.get("raw_output") or "")
        out = dict(row)
        stage_outputs = dict(out.get("stage_outputs") or {})
        shrink_stage: list[dict[str, Any]] = []
        changed = False
        for item in targets:
            old_box = clamp_box([int(float(v)) for v in item["box"][:4]], width, height)
            if not old_box:
                continue
            new_box, meta = choose_dark_box(
                image,
                old_box,
                mode=args.mode,
                threshold=args.threshold,
                min_component_area=args.min_component_area,
                max_components=args.max_components,
                pad=args.pad,
            )
            if not new_box:
                stats["no_shrink_box"] += 1
                shrink_stage.append({"old_box": old_box, "new_box": None, "meta": meta})
                continue
            old_area = max(1, (old_box[2] - old_box[0]) * (old_box[3] - old_box[1]))
            new_area = max(1, (new_box[2] - new_box[0]) * (new_box[3] - new_box[1]))
            if new_area / old_area > args.max_shrink_ratio:
                stats["rejected_not_shrunk"] += 1
                shrink_stage.append({"old_box": old_box, "new_box": new_box, "accepted": False, "meta": meta})
                continue
            updated_report, replaced = replace_grounding(report, old_box, new_box)
            if not replaced:
                stats["replace_miss"] += 1
                shrink_stage.append({"old_box": old_box, "new_box": new_box, "accepted": False, "replace_miss": True, "meta": meta})
                continue
            report = updated_report
            item["box_before_dark_shrink"] = old_box
            item["box"] = new_box
            shrink_stage.append({"old_box": old_box, "new_box": new_box, "accepted": True, "meta": meta})
            stats["boxes_shrunk"] += 1
            changed = True
        if changed:
            out["raw_output"] = report
            verifier = dict((stage_outputs.get("qwen_exhaustive_crop_verifier") or {}))
            verifier["accepted"] = verifier.get("accepted") or []
            # Keep stage metadata aligned with the report for downstream inspection.
            for accepted in verifier["accepted"]:
                for shrunk in shrink_stage:
                    if shrunk.get("accepted") and accepted.get("box") == shrunk.get("old_box"):
                        accepted["box_before_dark_shrink"] = shrunk["old_box"]
                        accepted["box"] = shrunk["new_box"]
                        accepted["dark_shrink"] = True
            stage_outputs["qwen_exhaustive_crop_verifier"] = verifier
            stage_outputs["qwen_dark_component_shrink"] = {
                "applied": True,
                "mode": args.mode,
                "threshold": args.threshold,
                "pad": args.pad,
                "items": shrink_stage,
            }
            out["stage_outputs"] = stage_outputs
            stats["rows_changed"] += 1
        out_rows.append(out)
        diag_rows.append({"sample_id": sid, "targets": len(targets), "changed": changed, "items": shrink_stage})

    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    diag_path = resolve_pipe_path(args.diag_jsonl)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in diag_rows) + "\n", encoding="utf-8")
    summary_path = resolve_pipe_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(stats), ensure_ascii=False))


if __name__ == "__main__":
    main()
