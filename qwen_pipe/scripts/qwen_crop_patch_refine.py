#!/usr/bin/env python3
"""Refine grounding boxes with image crop/patch component candidates.

This GT-blind postprocess searches the source image for high-confidence visual
patches such as solid redaction bars or highlight blocks. It keeps the final
verdict, risk score, reasons, and number of [GROUNDING] entries unchanged; for
each existing grounding it may replace the box with one detected patch box.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"
GROUNDING_RE = re.compile(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", re.IGNORECASE)

BLOCK_TERMS = (
    "redaction",
    "redacted",
    "censored",
    "obscur",
    "cover",
    "covered",
    "black block",
    "black rectangle",
    "solid black",
    "rectangle",
    "rectangular",
    "box",
    "block",
    "kotak",
    "blok",
    "hitam",
    "บดบัง",
    "遮挡",
    "遮蔽",
    "覆盖",
    "黑块",
    "黑色块",
    "实心黑色块",
    "像素块",
)
COLOR_TERMS = (
    "color",
    "colour",
    "gray",
    "grey",
    "red",
    "yellow",
    "highlight",
    "lighter",
    "darker",
    "颜色",
    "色",
    "高亮",
    "灰色",
    "红色",
    "黄色",
    "สี",
)
RENDER_TERMS = (
    "pixelated",
    "jagged",
    "blur",
    "blurry",
    "glitch",
    "rendering",
    "overlap",
    "merged",
    "low-resolution",
    "low resolution",
    "raster",
    "aliasing",
    "乱码",
    "渲染",
    "模糊",
    "锯齿",
    "重叠",
    "混叠",
    "ซ้อน",
)
SCHEDULE_TERMS = ("schedule", "time", "times", "table", "cell", "ตาราง", "表格", "时间")


@dataclass(frozen=True)
class Patch:
    box: list[int]
    kind: str
    fill: float
    pixels: int
    score: float


def setup_debug_import(debug_root: Path) -> None:
    docshield_dir = debug_root / "baselines" / "DocShield"
    if not docshield_dir.exists():
        raise SystemExit(f"DocShield helper directory not found: {docshield_dir}")
    sys.path.insert(0, str(docshield_dir))


def resolve_pipe_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PIPE_ROOT / p


def resolve_image_path(row: dict[str, Any], debug_root: Path) -> Path | None:
    raw = row.get("image_path") or row.get("image_name")
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    if p.is_absolute() and p.exists():
        return p
    for base in (debug_root, debug_root / "data" / "images", PIPE_ROOT.parent):
        candidate = base / p
        if candidate.exists():
            return candidate
    return None


def parse_box(value: str) -> list[int] | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(nums) < 4:
        return None
    vals = [int(round(float(v))) for v in nums[:4]]
    x1, y1, x2, y2 = vals
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def grounding_contexts(report: str) -> list[dict[str, Any]]:
    matches = list(GROUNDING_RE.finditer(report))
    contexts: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        block_start = report.rfind("###", 0, match.start())
        if block_start < 0:
            block_start = max(0, report.rfind("\n", 0, match.start()))
        context = report[block_start:next_start]
        box = parse_box(match.group(2))
        contexts.append({"match": match, "box": box, "context": context})
    return contexts


def expanded_box(box: list[int], width: int, height: int, factor: float) -> list[int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * factor))
    pad_y = int(round(bh * factor))
    return [max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)]


def center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def inside(inner: list[int], outer: list[int]) -> bool:
    cx, cy = center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def dist_norm(a: list[int], b: list[int], width: int, height: int) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot((ax - bx) / max(width, 1), (ay - by) / max(height, 1))


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(term in low for term in terms)


def row_runs(mask: np.ndarray, min_run: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    height, _ = mask.shape
    for y in range(height):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            continue
        gaps = np.flatnonzero(np.diff(xs) > 1)
        starts = np.r_[0, gaps + 1]
        ends = np.r_[gaps, xs.size - 1]
        for start_i, end_i in zip(starts, ends):
            x1 = int(xs[start_i])
            x2 = int(xs[end_i]) + 1
            if x2 - x1 >= min_run:
                runs.append((y, x1, x2))
    return runs


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []

    def add(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        return idx

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def components_from_runs(mask: np.ndarray, min_run: int) -> list[tuple[int, int, int, int, int]]:
    runs = row_runs(mask, min_run)
    uf = UnionFind()
    indexed: list[tuple[int, int, int, int]] = []
    prev: list[tuple[int, int, int]] = []
    prev_y = -2
    for y, x1, x2 in runs:
        idx = uf.add()
        if y == prev_y:
            active_prev = prev
        elif y == prev_y + 1:
            active_prev = prev
        else:
            active_prev = []
        for px1, px2, pidx in active_prev:
            if x1 <= px2 + 1 and x2 + 1 >= px1:
                uf.union(idx, pidx)
        if y != prev_y:
            prev = []
            prev_y = y
        prev.append((x1, x2, idx))
        indexed.append((y, x1, x2, idx))

    stats: dict[int, list[int]] = {}
    for y, x1, x2, idx in indexed:
        root = uf.find(idx)
        if root not in stats:
            stats[root] = [x1, y, x2, y + 1, 0]
        s = stats[root]
        s[0] = min(s[0], x1)
        s[1] = min(s[1], y)
        s[2] = max(s[2], x2)
        s[3] = max(s[3], y + 1)
        s[4] += x2 - x1
    return [tuple(v) for v in stats.values()]


def detect_patches(image: Image.Image, kind: str) -> list[Patch]:
    arr = np.asarray(image.convert("RGB"))
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    height, width = arr.shape[:2]
    if kind == "dark":
        luma = (30 * r + 59 * g + 11 * b) / 100.0
        mask = luma < 55
        min_run = max(8, int(width * 0.008))
        min_h = max(3, int(height * 0.0015))
        min_w = min_run
    elif kind == "yellow":
        mask = (r > 160) & (g > 140) & (b < 145) & ((r - b) > 45) & ((g - b) > 35)
        min_run = max(8, int(width * 0.008))
        min_h = max(4, int(height * 0.002))
        min_w = min_run
    elif kind == "pale_yellow":
        # Timetable/schedule highlights are often low-saturation yellow rather
        # than the strong marker yellow handled above.
        mask = (r > 185) & (g > 175) & (b > 115) & (b < 215) & ((r - b) > 25) & ((g - b) > 20)
        min_run = max(8, int(width * 0.008))
        min_h = max(4, int(height * 0.001))
        min_w = min_run
    elif kind == "red":
        mask = (r > 145) & (g < 135) & (b < 135) & ((r - g) > 35) & ((r - b) > 35)
        min_run = max(5, int(width * 0.004))
        min_h = max(4, int(height * 0.0015))
        min_w = min_run
    else:
        return []

    patches: list[Patch] = []
    for x1, y1, x2, y2, pixels in components_from_runs(mask, min_run):
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_w or bh < min_h:
            continue
        area = bw * bh
        if area <= 0:
            continue
        fill = pixels / area
        if fill < 0.28:
            continue
        if area > width * height * 0.08:
            continue
        # Suppress ordinary page borders and table rules: they are very thin
        # but extremely wide. Redaction/highlight patches are usually thicker.
        if bw > width * 0.75 and bh < height * 0.01:
            continue
        score = pixels * fill * (1.0 + min(2.0, bh / 18.0))
        patches.append(Patch([x1, y1, x2, y2], kind, fill, pixels, score))
    patches.sort(key=lambda p: p.score, reverse=True)
    return patches[:80]


def candidate_kinds(context: str, profile: str) -> tuple[list[str], bool]:
    global_ok = False
    kinds: list[str] = []
    if has_any(context, BLOCK_TERMS):
        kinds.append("dark")
        global_ok = True
    if profile == "block-only":
        return kinds, global_ok
    if has_any(context, COLOR_TERMS):
        kinds.extend(["red", "yellow"])
    if profile == "block-color":
        deduped_color: list[str] = []
        for kind in kinds:
            if kind not in deduped_color:
                deduped_color.append(kind)
        return deduped_color, global_ok
    if has_any(context, SCHEDULE_TERMS):
        kinds.append("yellow")
        global_ok = True
    if has_any(context, RENDER_TERMS):
        kinds.append("dark")
    deduped: list[str] = []
    for kind in kinds:
        if kind not in deduped:
            deduped.append(kind)
    return deduped, global_ok


def choose_patch(
    box: list[int],
    context: str,
    patches_by_kind: dict[str, list[Patch]],
    width: int,
    height: int,
    local_factor: float,
    profile: str,
    min_patch_width_frac: float,
) -> Patch | None:
    kinds, global_ok = candidate_kinds(context, profile)
    if not kinds:
        return None
    local = expanded_box(box, width, height, local_factor)
    candidates: list[tuple[float, Patch]] = []
    for kind in kinds:
        for patch in patches_by_kind.get(kind, []):
            if patch.box[2] - patch.box[0] < width * min_patch_width_frac:
                continue
            if inside(patch.box, local):
                candidates.append((patch.score / (1.0 + 4.0 * dist_norm(patch.box, box, width, height)), patch))
    if not candidates and global_ok:
        for kind in kinds:
            for patch in patches_by_kind.get(kind, [])[:20]:
                if patch.box[2] - patch.box[0] < width * min_patch_width_frac:
                    continue
                candidates.append((patch.score / (1.0 + 1.5 * dist_norm(patch.box, box, width, height)), patch))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def pad_patch(box: list[int], width: int, height: int, pad: int) -> list[int]:
    return [
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(width, box[2] + pad),
        min(height, box[3] + pad),
    ]


def replace_groundings(report: str, boxes: list[list[int]]) -> tuple[str, int]:
    idx = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        if idx >= len(boxes):
            return match.group(0)
        box = boxes[idx]
        idx += 1
        return f"{match.group(1)}{box}"

    return GROUNDING_RE.sub(repl, report), idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--final-forged-only", action="store_true")
    parser.add_argument("--local-factor", type=float, default=2.0)
    parser.add_argument("--patch-pad", type=int, default=3)
    parser.add_argument("--min-replaced-per-row", type=int, default=1)
    parser.add_argument("--min-patch-width-frac", type=float, default=0.0)
    parser.add_argument(
        "--profile",
        choices=["block-only", "block-color", "all"],
        default="all",
        help="Detector trigger profile. block-only is the conservative redaction/occlusion setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)

    from postprocess import parse_cct_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    output_path = resolve_pipe_path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "rows": 0,
        "rows_rewritten": 0,
        "boxes_replaced": 0,
        "skipped_non_forged": 0,
        "missing_images": 0,
        "kind_counts": {},
    }

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            stats["rows"] += 1
            out = dict(row)
            final_verdict = str((row.get("parsed") or {}).get("conclusion") or "").upper()
            report = str(row.get("raw_output") or "")
            contexts = grounding_contexts(report)
            replaced = 0
            patch_kinds: list[str] = []

            if args.final_forged_only and final_verdict != "FORGED":
                stats["skipped_non_forged"] += 1
            elif contexts:
                image_path = resolve_image_path(row, debug_root)
                if image_path is None:
                    stats["missing_images"] += 1
                else:
                    image = Image.open(image_path).convert("RGB")
                    width, height = image.size
                    needed = set()
                    for ctx in contexts:
                        kinds, _ = candidate_kinds(str(ctx.get("context") or ""), args.profile)
                        needed.update(kinds)
                    patches_by_kind = {kind: detect_patches(image, kind) for kind in sorted(needed)}
                    new_boxes: list[list[int]] = []
                    for ctx in contexts:
                        box = ctx.get("box")
                        if not box:
                            continue
                        patch = choose_patch(
                            box,
                            str(ctx.get("context") or ""),
                            patches_by_kind,
                            width,
                            height,
                            args.local_factor,
                            args.profile,
                            args.min_patch_width_frac,
                        )
                        if patch:
                            new_box = pad_patch(patch.box, width, height, args.patch_pad)
                            if new_box != box:
                                replaced += 1
                                patch_kinds.append(patch.kind)
                            new_boxes.append(new_box)
                        else:
                            new_boxes.append(box)
                    if replaced >= args.min_replaced_per_row and len(new_boxes) == len(contexts):
                        new_report, count = replace_groundings(report, new_boxes)
                        if count == len(new_boxes) and new_report != report:
                            out["raw_output"] = new_report
                            out["parsed"] = parse_cct_report(new_report)

            stage_outputs = dict(out.get("stage_outputs") or {})
            applied = out.get("raw_output") != row.get("raw_output")
            stage_outputs["qwen_pipe_crop_patch_refine"] = {
                "applied": applied,
                "boxes_replaced": replaced if applied else 0,
                "patch_kinds": patch_kinds if applied else [],
                "local_factor": args.local_factor,
                "patch_pad": args.patch_pad,
                "min_patch_width_frac": args.min_patch_width_frac,
                "profile": args.profile,
                "policy": "Replace existing grounding boxes with high-confidence visual patch components only; keep verdicts, reasons, and anomaly count unchanged.",
            }
            out["stage_outputs"] = stage_outputs
            if applied:
                stats["rows_rewritten"] += 1
                stats["boxes_replaced"] += replaced
                for kind in patch_kinds:
                    stats["kind_counts"][kind] = stats["kind_counts"].get(kind, 0) + 1
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
