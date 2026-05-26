"""Postprocessing utilities for DocShield CCT outputs.

The model is asked to wrap reasoning inside ``<think>...</think>`` and the
final structured forensic report inside ``<report>...</report>``. We split the
two so that:

* ``thinking`` keeps the chain-of-thought (kept for debugging only — never
  forwarded to the CodaBench submission).
* ``raw_output`` carries the Markdown report that the GenText pipeline expects;
  ``gentext_agent.build_report()`` already knows how to handle a Markdown
  report when it sees ``[Conclusion]`` or ``# FORGERY ANALYSIS``.

This module also exposes a small ``parse_cct_report`` helper that pulls out the
classification label, risk score and bounding boxes for sanity checking on
debug splits.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
REPORT_RE = re.compile(r"<report>(.*?)</report>", re.DOTALL | re.IGNORECASE)
CONCLUSION_RE = re.compile(
    r"\[Conclusion\][:\s*]*\*?\*?\s*(FORGED|AUTHENTIC|PRISTINE)",
    re.IGNORECASE,
)
RISK_RE = re.compile(
    r"\[RISK[_\s]?SCORE\][:\s*]*\*?\*?\s*([0-9]+)",
    re.IGNORECASE,
)
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[([^\[\]]+)\]", re.IGNORECASE)


@dataclass
class CctSplit:
    thinking: str
    report: str
    raw: str
    has_think_tag: bool
    has_report_tag: bool


def split_think_report(raw: str) -> CctSplit:
    """Extract the ``<think>`` and ``<report>`` blocks from raw model output.

    Falls back gracefully when the model forgets the wrappers:

    * If only ``<report>`` is present, ``thinking`` is empty.
    * If only ``<think>`` is present, the report is whatever follows the close
      ``</think>`` tag (or the full text if even that is missing).
    * If neither tag is present, we treat the whole output as the report.
    """

    raw = raw or ""
    think_match = THINK_RE.search(raw)
    report_match = REPORT_RE.search(raw)

    thinking = think_match.group(1).strip() if think_match else ""

    if report_match:
        report = report_match.group(1).strip()
    elif think_match:
        tail = raw[think_match.end():].strip()
        report = tail
    else:
        report = raw.strip()

    if not report:
        report = raw.strip()

    return CctSplit(
        thinking=thinking,
        report=report,
        raw=raw,
        has_think_tag=think_match is not None,
        has_report_tag=report_match is not None,
    )


def normalize_label(text: str | None) -> str:
    if not text:
        return "UNKNOWN"
    upper = text.upper()
    if "FORGED" in upper or "TAMPER" in upper or "FAKE" in upper:
        return "FORGED"
    if "AUTHENTIC" in upper or "PRISTINE" in upper or "REAL" in upper:
        return "AUTHENTIC"
    return "UNKNOWN"


def parse_cct_report(report: str) -> dict[str, Any]:
    """Lightweight extractor used for eval/debug only.

    Returns a dict with: conclusion, risk_score, anomalies (list of {grounding,
    raw_block}). The full pipeline still relies on ``gentext_agent.build_report``
    using the raw Markdown — this helper exists for quick sanity checking.
    """

    conclusion_match = CONCLUSION_RE.search(report)
    conclusion = normalize_label(conclusion_match.group(1) if conclusion_match else None)

    risk_match = RISK_RE.search(report)
    try:
        risk = int(risk_match.group(1)) if risk_match else (80 if conclusion == "FORGED" else 0)
    except ValueError:
        risk = 80 if conclusion == "FORGED" else 0
    risk = max(0, min(100, risk))

    anomalies: list[dict[str, Any]] = []
    for grounding_match in GROUNDING_RE.finditer(report):
        body = grounding_match.group(1)
        nums = re.findall(r"-?\d+(?:\.\d+)?", body)
        box: list[int] | None = None
        if len(nums) >= 4:
            try:
                box = [int(round(float(v))) for v in nums[:4]]
            except ValueError:
                box = None
        anomalies.append({"grounding": box})

    return {
        "conclusion": conclusion,
        "risk_score": risk,
        "anomalies": anomalies,
    }


def to_pipeline_record(
    *,
    sample_id: str | None,
    image_name: str | None,
    image_path: str | None,
    width: int | None,
    height: int | None,
    gold_label: str | None,
    language_code: str | None,
    model_id: str,
    raw_full: str,
    elapsed_sec: float | None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a JSONL record compatible with ``gentext_agent.py make-prediction``.

    Schema mirrors ``cmd_run_qwen``'s output so downstream tooling
    (``make-prediction`` / ``eval`` / ``summarize``) works without any code
    change. Two extra fields are added:

    * ``raw_output_full`` — the original generated string (think + report).
    * ``thinking`` — the extracted ``<think>`` body (or "" when absent).

    The canonical ``raw_output`` field carries the Markdown report only, which
    ``gentext_agent.build_report()`` recognises via ``[Conclusion]`` /
    ``# FORGERY ANALYSIS`` substring matching.

    **Coordinate scaling**: If width/height are provided, all [GROUNDING] boxes
    in the report are scaled from Qwen API resize-space to original resolution.
    """

    if error is not None:
        return {
            "sample_id": sample_id,
            "image_name": image_name,
            "image_path": image_path,
            "gold_label": gold_label,
            "language_code": language_code,
            "model": model_id,
            "error": error,
        }

    split = split_think_report(raw_full)
    report = split.report

    # Scale grounding boxes to original resolution if dimensions are known
    if width and height:
        report = scale_report_boxes(report, width, height)

    parsed = parse_cct_report(report)

    return {
        "sample_id": sample_id,
        "image_name": image_name,
        "image_path": image_path,
        "width": width,
        "height": height,
        "gold_label": gold_label,
        "language_code": language_code,
        "model": model_id,
        "raw_output": report,
        "raw_output_full": raw_full,
        "thinking": split.thinking,
        "has_think_tag": split.has_think_tag,
        "has_report_tag": split.has_report_tag,
        "parsed": parsed,
        "elapsed_sec": elapsed_sec,
    }


def qwen_resize(w: int, h: int, max_pixels: int = 1003520, patch_size: int = 14) -> tuple[int, int]:
    """Replicate Qwen-VL API smart_resize to find the resize target dimensions.

    The API resizes images before inference; model-output coordinates are in
    resize space. We need the inverse scale to project back to original pixels.
    """
    s = math.sqrt(max_pixels / (w * h))
    if s < 1.0:
        nw, nh = int(w * s), int(h * s)
    else:
        nw, nh = w, h
    rw = max(patch_size, round(nw / patch_size) * patch_size)
    rh = max(patch_size, round(nh / patch_size) * patch_size)
    return rw, rh


def scale_bbox_to_original(
    bbox: list[int | float],
    orig_w: int,
    orig_h: int,
    max_pixels: int = 1003520,
    patch_size: int = 14,
) -> list[int]:
    """Scale a bounding box from Qwen resize-space back to original image coordinates."""
    resize_w, resize_h = qwen_resize(orig_w, orig_h, max_pixels=max_pixels, patch_size=patch_size)
    sx = orig_w / resize_w
    sy = orig_h / resize_h
    x1, y1, x2, y2 = bbox[:4]
    return [
        max(0, min(orig_w, int(round(x1 * sx)))),
        max(0, min(orig_h, int(round(y1 * sy)))),
        max(0, min(orig_w, int(round(x2 * sx)))),
        max(0, min(orig_h, int(round(y2 * sy)))),
    ]


def scale_report_boxes(report: str, orig_w: int, orig_h: int) -> str:
    """Rewrite all [GROUNDING]:[x1, y1, x2, y2] in report to original-resolution coords."""
    grounding_re = re.compile(r"(\[GROUNDING\]\s*:\s*)\[([^\[\]]+)\]", re.IGNORECASE)

    def _replace(m: re.Match) -> str:
        prefix = m.group(1)
        body = m.group(2)
        nums = re.findall(r"-?\d+(?:\.\d+)?", body)
        if len(nums) < 4:
            return m.group(0)
        try:
            bbox = [float(v) for v in nums[:4]]
            scaled = scale_bbox_to_original(bbox, orig_w, orig_h)
            return f"{prefix}{scaled}"
        except (ValueError, ZeroDivisionError):
            return m.group(0)

    return grounding_re.sub(_replace, report)


def dumps(obj: Any) -> str:
    """Convenience wrapper used by the runner."""

    return json.dumps(obj, ensure_ascii=False)
