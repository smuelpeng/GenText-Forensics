#!/usr/bin/env python3
"""
Competition-aligned local evaluator for GenText-Forensics.

This script is intentionally stricter than scripts/local_eval_codalab.py:

* S_Det: forged-class F1 plus accuracy for diagnostics.
* S_Loc: pixel F1 and IoU from predicted boxes rasterized against GT masks,
  falling back to GT report box mIoU when masks are unavailable.
* S_Exp: BERTScore F1 when available, otherwise deterministic token-F1.
* S_Rep: deterministic proxy for the hidden LLM-judge report score. It scores
  report structure, verdict-risk consistency, anomaly grounding, explanation
  detail, summary quality, and hallucinated anomalies on authentic samples.

It is not the official CodaBench program, but it is designed to be useful for
model selection without rewarding the old "any box + valid tags" shortcut.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


WEIGHTS = {"S_Det": 0.30, "S_Loc": 0.20, "S_Exp": 0.15, "S_Rep": 0.35}
VALID_LABELS = {"AUTHENTIC", "FORGED"}

# Calibration anchors collected from real CodaBench submissions. Each entry is
# (label, local_proxy_S_Rep, leaderboard_S_Fin, val_proxy_S_Fin) where the
# proxy values come from this evaluator with --skip-bertscore. The damping
# factor for S_Rep_proxy is fitted from these anchors. Append more rows as
# new submissions are scored on the leaderboard.
CALIBRATION_ANCHORS: List[Dict[str, float]] = [
    # v0.2: qwen3-vl-30b, JSON->template build_report, leaderboard test 0.2409.
    # val_1000 proxy: S_Det=0.6300 (acc), S_Loc=0.0186, S_Exp_token=0.1381, S_Rep=0.5342.
    {"label": "v0.2", "S_Rep_proxy": 0.5342, "S_Det": 0.6300,
     "S_Loc": 0.0186, "S_Exp_token": 0.1381, "leaderboard": 0.2409},
]


def calibrate_s_rep(rep_proxy: float) -> float:
    """Damp S_Rep_proxy toward what the official LLM judge actually returns.

    Uses a single-anchor linear fit through the origin: target_rep / proxy_rep
    where target_rep is what S_Rep must be to make the v0.2 leaderboard arithmetic
    close. With only one anchor this is roughly a *0.13 multiplier; once we
    submit more variants we re-fit with least squares in calibrate_proxy.py.
    """
    if not CALIBRATION_ANCHORS:
        return rep_proxy
    a = CALIBRATION_ANCHORS[0]
    other = (
        WEIGHTS["S_Det"] * a["S_Det"]
        + WEIGHTS["S_Loc"] * a["S_Loc"]
        + WEIGHTS["S_Exp"] * a["S_Exp_token"]
    )
    target_rep_weighted = max(0.0, a["leaderboard"] - other)
    target_rep = target_rep_weighted / WEIGHTS["S_Rep"]
    if a["S_Rep_proxy"] <= 0:
        return rep_proxy
    factor = max(0.0, min(1.0, target_rep / a["S_Rep_proxy"]))
    return rep_proxy * factor

CONCLUSION_RE = re.compile(
    r"\[Conclusion\][:\s*]*\*?\*?\s*(FORGED|AUTHENTIC|PRISTINE)",
    re.IGNORECASE,
)
RISK_RE = re.compile(r"\[?RISK[_\s]?SCORE\]?[^\d]{0,12}(\d{1,3})", re.IGNORECASE)
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[([^\[\]]+)\]", re.IGNORECASE)
REASON_RE = re.compile(
    r"\[REASON\]\s*:\s*(.*?)(?=\n\s*###\s*ANOMALY|\n\s*---|\n\s*##\s*SUMMARY|\Z)",
    re.IGNORECASE | re.DOTALL,
)
SUMMARY_RE = re.compile(
    r"##\s*SUMMARY\s*\n(.*?)(?=\n\s*---|\n\s*\*\*END OF REPORT\*\*|\Z)",
    re.IGNORECASE | re.DOTALL,
)
REPORT_RE = re.compile(r"<report>(.*?)</report>", re.IGNORECASE | re.DOTALL)

HEADER_FRAGMENTS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("missing_header", re.compile(r"#\s*FORGERY\s+ANALYSIS\b", re.IGNORECASE)),
    ("missing_detail_section", re.compile(r"DETAILED\s+ANOMALY\s+ANALYSIS", re.IGNORECASE)),
    ("missing_end_marker", re.compile(r"END\s+OF\s+REPORT", re.IGNORECASE)),
]

# Unicode block ranges used for a coarse language-of-text classifier. The
# competition GT only relies on script families, so we bucket each character
# into one of: ar (Arabic), zh (CJK Han), th (Thai), latin, other.
_UNICODE_BLOCKS: List[Tuple[int, int, str]] = [
    (0x0600, 0x06FF, "ar"),
    (0x0750, 0x077F, "ar"),
    (0xFB50, 0xFDFF, "ar"),
    (0xFE70, 0xFEFF, "ar"),
    (0x0E00, 0x0E7F, "th"),
    (0x4E00, 0x9FFF, "zh"),
    (0x3400, 0x4DBF, "zh"),
    (0xF900, 0xFAFF, "zh"),
    (0x3040, 0x30FF, "zh"),  # JP kana, lump under CJK for our purposes
    (0xAC00, 0xD7AF, "zh"),  # Hangul — unlikely in dataset but safe bucket
    (0x0041, 0x007A, "latin"),
    (0x00C0, 0x024F, "latin"),
]

# Map dataset language_code → coarse script bucket.
_LANG_TO_BUCKET: Dict[str, str] = {
    "en": "latin",
    "id": "latin",
    "ms": "latin",
    "vi": "latin",
    "fr": "latin",
    "es": "latin",
    "pt": "latin",
    "de": "latin",
    "tr": "latin",
    "zh": "zh",
    "ja": "zh",
    "ko": "zh",
    "th": "th",
    "ar": "ar",
}


def detect_script_bucket(text: str) -> str:
    """Return dominant script bucket of ``text`` (latin/zh/th/ar/other)."""
    if not text:
        return "other"
    counts: Dict[str, int] = defaultdict(int)
    for ch in text:
        cp = ord(ch)
        for lo, hi, bucket in _UNICODE_BLOCKS:
            if lo <= cp <= hi:
                counts[bucket] += 1
                break
    if not counts:
        return "other"
    return max(counts.items(), key=lambda kv: kv[1])[0]


@dataclass
class ParsedReport:
    conclusion: str = "UNKNOWN"
    risk_score: Optional[int] = None
    boxes: List[List[float]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    summary: str = ""
    report_len: int = 0
    issues: List[str] = field(default_factory=list)


@dataclass
class SampleEval:
    image_name: str
    sample_id: str
    language_code: str
    gt_label: str
    pred_label: str
    det_correct: bool
    loc_score: Optional[float]  # FORGED-only; None for AUTHENTIC and parse failures
    loc_method: str
    box_miou: Optional[float]
    mask_iou: Optional[float]
    mask_f1: Optional[float]
    count_score: Optional[float]
    exp_score: Optional[float]
    rep_score: float
    rep_breakdown: Dict[str, float]
    pred_boxes: int
    gt_boxes: int
    pred_report_len: int
    gt_report_len: int
    pred_script: str
    expected_script: str
    issues: List[str]


def norm_label(value: object) -> str:
    text = str(value or "").upper()
    if "FORGED" in text or "BLACK" in text or "TAMPER" in text or "FAKE" in text:
        return "FORGED"
    if "AUTHENTIC" in text or "PRISTINE" in text or "WHITE" in text or "REAL" in text:
        return "AUTHENTIC"
    return "UNKNOWN"


def _read_jsonl_stream(stream) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Malformed JSONL at line {line_no}: {exc}") from exc
    return rows


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                target = None
                for candidate in ("prediction.jsonl.gz", "prediction.jsonl"):
                    if candidate in names:
                        target = candidate
                        break
                if target is None:
                    # Fallback: any *.jsonl(.gz) file inside.
                    for n in names:
                        if n.endswith(".jsonl.gz") or n.endswith(".jsonl"):
                            target = n
                            break
                if target is None:
                    raise SystemExit(
                        f"Zip {path} does not contain prediction.jsonl[.gz]; got {names}"
                    )
                with zf.open(target) as raw:
                    if target.endswith(".gz"):
                        with gzip.open(raw, "rt", encoding="utf-8") as f:
                            return _read_jsonl_stream(f)
                    return _read_jsonl_stream(io_text_wrap(raw))
        except zipfile.BadZipFile as exc:
            raise SystemExit(f"Bad zip: {path} ({exc})") from exc

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return _read_jsonl_stream(f)


def io_text_wrap(raw):
    """Wrap a binary stream as a UTF-8 text iterator without slurping."""
    import io

    return io.TextIOWrapper(raw, encoding="utf-8")


def strip_report_wrappers(text: str) -> str:
    match = REPORT_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def parse_report(report: str) -> ParsedReport:
    report = strip_report_wrappers(report)
    parsed = ParsedReport(report_len=len(report))

    if not report:
        parsed.issues.append("empty_report")
        return parsed

    conclusion_match = CONCLUSION_RE.search(report)
    if conclusion_match:
        parsed.conclusion = norm_label(conclusion_match.group(1))
    else:
        parsed.issues.append("missing_or_invalid_conclusion")

    risk_match = RISK_RE.search(report)
    if risk_match:
        parsed.risk_score = max(0, min(100, int(risk_match.group(1))))
    else:
        parsed.issues.append("missing_risk_score")

    for box_match in GROUNDING_RE.finditer(report):
        nums = re.findall(r"-?\d+(?:\.\d+)?", box_match.group(1))
        if len(nums) < 4:
            parsed.issues.append("malformed_grounding")
            continue
        try:
            box = [float(v) for v in nums[:4]]
        except ValueError:
            parsed.issues.append("malformed_grounding")
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            parsed.issues.append("invalid_box_order")
            continue
        parsed.boxes.append(box)

    parsed.reasons = [" ".join(m.group(1).split()) for m in REASON_RE.finditer(report)]

    summary_match = SUMMARY_RE.search(report)
    if summary_match:
        parsed.summary = " ".join(summary_match.group(1).split())
    else:
        parsed.issues.append("missing_summary")

    for issue, pattern in HEADER_FRAGMENTS:
        if not pattern.search(report):
            parsed.issues.append(issue)

    return parsed


def clamp_box(box: List[float], width: Optional[int], height: Optional[int]) -> Optional[List[int]]:
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = box
    if width and height:
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mean_best_box_iou(gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> Optional[float]:
    if not gt_boxes:
        return None
    if not pred_boxes:
        return 0.0
    return float(np.mean([max(box_iou(gt, pred) for pred in pred_boxes) for gt in gt_boxes]))


def get_image_size(row: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    for key_w, key_h in [("width", "height"), ("image_width", "image_height")]:
        if row.get(key_w) and row.get(key_h):
            return int(row[key_w]), int(row[key_h])
    image_path = row.get("image_path")
    if not image_path:
        return None, None
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return None, None


def boxes_to_mask(boxes: List[List[float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        clamped = clamp_box(box, width, height)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped
        mask[y1:y2, x1:x2] = True
    return mask


def read_binary_mask(mask_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("L")
            if im.size != target_size:
                im = im.resize(target_size, Image.NEAREST)
            return np.asarray(im) > 0
    except Exception:
        return None


def mask_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return float(inter / union) if union else 0.0


def mask_f1(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fp = np.logical_and(~gt_mask, pred_mask).sum()
    fn = np.logical_and(gt_mask, ~pred_mask).sum()
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def object_count_score(gt_label: str, gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> Optional[float]:
    if gt_label == "AUTHENTIC":
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if not gt_boxes:
        return None
    return 1.0 if len(gt_boxes) == len(pred_boxes) else 0.0


_GT_TARGET_LEN = {"FORGED": 3453, "AUTHENTIC": 1281}  # from val_1000 GT analysis


def _length_alignment(pred_len: int, target: int) -> float:
    """Score in [0, 1] that peaks when ``pred_len ≈ target``.

    * 1.0 inside [0.5×target, 1.5×target]
    * Linear falloff to 0 at 0.2×target / 3.0×target
    """
    if target <= 0 or pred_len <= 0:
        return 0.0
    ratio = pred_len / target
    if 0.5 <= ratio <= 1.5:
        return 1.0
    if ratio < 0.5:
        return max(0.0, (ratio - 0.2) / 0.3)
    return max(0.0, (3.0 - ratio) / 1.5)


def _verdict_self_consistency(pred: ParsedReport) -> float:
    """How well a report's verdict, risk_score, and box presence agree internally.

    GT-blind: the score doesn't peek at gt_label, so it can be combined with a
    GT-referenced bonus elsewhere without double-counting detection errors.
    """
    if pred.conclusion not in VALID_LABELS:
        return 0.0
    score = 0.0
    if pred.risk_score is None:
        return 0.0
    if pred.conclusion == "AUTHENTIC":
        score += 0.5 if pred.risk_score <= 10 else 0.0
        score += 0.5 if len(pred.boxes) == 0 else 0.0
    else:  # FORGED
        score += 0.5 if pred.risk_score >= 20 else 0.0
        score += 0.5 if len(pred.boxes) > 0 else 0.0
    return score


def _format_compliance(pred: ParsedReport) -> float:
    """Format checklist compatible with what an LLM judge would inspect."""
    if pred.report_len == 0:
        return 0.0
    issue_set = set(pred.issues)
    score = 0.0
    score += 0.20 if pred.conclusion in VALID_LABELS else 0.0
    score += 0.15 if pred.risk_score is not None else 0.0
    score += 0.15 if "missing_header" not in issue_set else 0.0
    score += 0.15 if "missing_detail_section" not in issue_set else 0.0
    score += 0.20 if pred.summary else 0.0
    score += 0.15 if "missing_end_marker" not in issue_set else 0.0
    return score


def _grounding_quality(pred: ParsedReport, width: Optional[int], height: Optional[int]) -> float:
    """For FORGED predictions only: anomalies should have valid boxes + reasons."""
    if not pred.boxes:
        return 0.0
    valid_boxes = sum(1 for box in pred.boxes if clamp_box(box, width, height) is not None)
    box_quality = valid_boxes / len(pred.boxes)
    reason_quality = min(1.0, len(pred.reasons) / len(pred.boxes))
    return 0.5 * box_quality + 0.5 * reason_quality


def _explanation_detail(pred: ParsedReport, gt_label: str) -> float:
    """Reward GT-style detail levels (long REASONs for FORGED, descriptive SUMMARY for AUTHENTIC)."""
    if gt_label == "AUTHENTIC":
        # AUTHENTIC GT = ~1281 chars total, dominated by SUMMARY w/ image description.
        return min(1.0, len(pred.summary) / 600.0)
    # FORGED GT averages ~250 chars per REASON and ~220 chars for SUMMARY.
    if not pred.reasons:
        return 0.5 * min(1.0, len(pred.summary) / 220.0)
    avg_reason = sum(len(r) for r in pred.reasons) / len(pred.reasons)
    return 0.5 * min(1.0, avg_reason / 200.0) + 0.5 * min(1.0, len(pred.summary) / 220.0)


def _language_match_factor(pred: ParsedReport, gt_language_code: str) -> float:
    """Multiplicative factor in [0.6, 1.0] depending on script-bucket match.

    GT REASONs/SUMMARY are written in the document's native script. If the
    prediction stays English while GT expects Chinese/Thai/Arabic, the LLM
    judge penalises hard. We damp S_Rep by up to 40% to mirror that.
    """
    expected_bucket = _LANG_TO_BUCKET.get(gt_language_code or "", "latin")
    body = " ".join(pred.reasons) + " " + pred.summary
    pred_bucket = detect_script_bucket(body) if body.strip() else "other"
    if expected_bucket == "latin":
        # Latin-script GT tolerates English-only output.
        return 1.0 if pred_bucket in {"latin", "other"} else 0.85
    if pred_bucket == expected_bucket:
        return 1.0
    if pred_bucket in {"other"}:
        return 0.75
    return 0.6


def report_quality_proxy(
    gt_label: str,
    gt_language_code: str,
    gt_anomaly_count: Optional[int],
    pred: ParsedReport,
    width: Optional[int],
    height: Optional[int],
) -> Tuple[float, Dict[str, float]]:
    """Deterministic proxy for the hidden S_Rep LLM judge.

    Returns ``(score, breakdown)`` so per-component contributions can be
    inspected. The proxy intentionally damps S_Rep when the report drifts
    from GT in language/length/verdict — failure modes that consistently
    cost real LLM-judge points in v0.1/v0.2 submissions.
    """
    if pred.report_len == 0:
        return 0.0, {"reason": "empty_report"}

    fmt = _format_compliance(pred)
    consistency = _verdict_self_consistency(pred)
    if pred.conclusion == "AUTHENTIC":
        grounding = 1.0 if len(pred.boxes) == 0 else 0.0
    else:
        grounding = _grounding_quality(pred, width, height)
    detail = _explanation_detail(pred, gt_label)
    target_len = _GT_TARGET_LEN.get(gt_label, _GT_TARGET_LEN["FORGED"])
    length_align = _length_alignment(pred.report_len, target_len)
    verdict_match = 1.0 if pred.conclusion == gt_label and gt_label in VALID_LABELS else 0.0

    # Anomaly count alignment (FORGED only). 1.0 if within 50% of GT count.
    if gt_label == "FORGED" and gt_anomaly_count and gt_anomaly_count > 0:
        diff = abs(len(pred.boxes) - gt_anomaly_count)
        count_align = max(0.0, 1.0 - diff / max(gt_anomaly_count, 1))
    elif gt_label == "AUTHENTIC":
        count_align = 1.0 if len(pred.boxes) == 0 else 0.0
    else:
        count_align = 0.5  # unknown — neutral

    weights = {
        "format": 0.15,
        "self_consistency": 0.10,
        "verdict_match": 0.20,
        "grounding": 0.15,
        "count_align": 0.10,
        "detail": 0.15,
        "length_align": 0.15,
    }
    breakdown = {
        "format": fmt,
        "self_consistency": consistency,
        "verdict_match": verdict_match,
        "grounding": grounding,
        "count_align": count_align,
        "detail": detail,
        "length_align": length_align,
    }
    score = sum(weights[k] * breakdown[k] for k in weights)
    lang_factor = _language_match_factor(pred, gt_language_code)
    score *= lang_factor
    breakdown["language_factor"] = lang_factor
    return max(0.0, min(1.0, score)), breakdown


def load_predictions(path: Path, raw_jsonl: bool) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    preds: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    empty = 0
    for row in rows:
        image_name = str(row.get("image_name") or Path(str(row.get("image_path") or "")).name)
        if not image_name:
            continue

        report = str(row.get("report") or "")
        if raw_jsonl and not report:
            report = str(
                row.get("raw_output")
                or row.get("api_raw_content")
                or row.get("raw_output_full")
                or ""
            )
        if not report:
            empty += 1
        if image_name in preds:
            duplicates.append(image_name)
        preds[image_name] = {**row, "image_name": image_name, "report": strip_report_wrappers(report)}

    if duplicates:
        sample = ", ".join(duplicates[:3])
        print(
            f"WARNING: {len(duplicates)} duplicate image_name in predictions; last entry wins. "
            f"first few: {sample}",
            file=sys.stderr,
        )
    if empty:
        print(
            f"WARNING: {empty}/{len(rows)} predictions have empty report text",
            file=sys.stderr,
        )
    return preds


def evaluate_sample(gt_row: Dict[str, Any], pred_row: Dict[str, Any]) -> SampleEval:
    image_name = str(gt_row.get("image_file") or gt_row.get("image_name") or pred_row.get("image_name"))
    sample_id = str(gt_row.get("sample_id") or "")
    language_code = str(gt_row.get("language_code") or "unknown")
    gt_report = str(gt_row.get("report_text") or gt_row.get("report") or "")
    pred_report = str(pred_row.get("report") or "")

    gt = parse_report(gt_report)
    pred = parse_report(pred_report)
    gt_label = norm_label(gt_row.get("label") or gt.conclusion)
    if gt_label == "PRISTINE":
        gt_label = "AUTHENTIC"
    pred_label = pred.conclusion

    width, height = get_image_size(gt_row)
    pred_boxes = [box for box in pred.boxes if clamp_box(box, width, height) is not None]
    gt_boxes = [box for box in gt.boxes if clamp_box(box, width, height) is not None]

    # ---- S_Loc: FORGED-only. AUTHENTIC samples are out of scope (handled in
    # S_Det / S_Rep). Mixing AUTHENTIC's 1.0/0.0 into the same average inflates
    # S_Loc by ~num_authentic/num_total and was the main local-vs-online drift.
    box_score: Optional[float] = None
    mask_score: Optional[float] = None
    f1_score: Optional[float] = None
    loc_score: Optional[float] = None
    loc_method = "skipped_authentic" if gt_label == "AUTHENTIC" else "none"

    if gt_label == "FORGED":
        box_score = mean_best_box_iou(gt_boxes, pred_boxes)
        if width and height:
            gt_mask = read_binary_mask(str(gt_row.get("mask_path") or ""), (width, height))
            if gt_mask is not None:
                pred_mask = boxes_to_mask(pred_boxes, width, height)
                mask_score = mask_iou(gt_mask, pred_mask)
                f1_score = mask_f1(gt_mask, pred_mask)
                loc_score = f1_score
                loc_method = "mask_f1"
        if loc_score is None:
            if box_score is not None:
                loc_score = box_score
                loc_method = "box_miou"
            else:
                # FORGED but no GT boxes available -> cannot score this sample.
                loc_score = None
                loc_method = "no_gt_boxes"

    count_score = object_count_score(gt_label, gt_boxes, pred_boxes)
    gt_anomaly_count = len(gt_boxes) if gt_label == "FORGED" else 0
    rep_score, rep_breakdown = report_quality_proxy(
        gt_label,
        language_code,
        gt_anomaly_count,
        pred,
        width,
        height,
    )

    issues = list(pred.issues)
    if gt_label == "AUTHENTIC" and pred_boxes:
        issues.append("authentic_predicted_with_boxes")
    if gt_label == "FORGED" and not pred_boxes:
        issues.append("forged_without_boxes")

    expected_script = _LANG_TO_BUCKET.get(language_code, "latin")
    pred_body = " ".join(pred.reasons) + " " + pred.summary
    pred_script = detect_script_bucket(pred_body) if pred_body.strip() else "other"

    return SampleEval(
        image_name=image_name,
        sample_id=sample_id,
        language_code=language_code,
        gt_label=gt_label,
        pred_label=pred_label,
        det_correct=gt_label == pred_label,
        loc_score=loc_score,
        loc_method=loc_method,
        box_miou=box_score,
        mask_iou=mask_score,
        mask_f1=f1_score,
        count_score=count_score,
        exp_score=None,
        rep_score=rep_score,
        rep_breakdown=rep_breakdown,
        pred_boxes=len(pred_boxes),
        gt_boxes=len(gt_boxes),
        pred_report_len=pred.report_len,
        gt_report_len=gt.report_len,
        pred_script=pred_script,
        expected_script=expected_script,
        issues=issues,
    )


def mean(values: Iterable[Optional[float]]) -> float:
    nums = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(nums)) if nums else 0.0


def compute_bertscore(
    gt_reports: List[str],
    pred_reports: List[str],
    model_type: str,
    batch_size: int,
) -> Tuple[float, List[float]]:
    try:
        from bert_score import BERTScorer
    except ImportError:
        print("WARNING: bert-score is not installed; S_Exp = 0.0", file=sys.stderr)
        return 0.0, []

    try:
        scorer = BERTScorer(
            model_type=model_type,
            device="cpu",
            rescale_with_baseline=True,
            num_layers=9,
        )
    except TypeError:
        scorer = BERTScorer(
            model_type=model_type,
            device="cpu",
            rescale_with_baseline=True,
            verbose=False,
            num_layers=9,
        )

    _, _, f1 = scorer.score(pred_reports, gt_reports, batch_size=batch_size)
    scores = [float(v) for v in f1.tolist()]
    return mean(scores), scores


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def token_f1(reference: str, candidate: str) -> float:
    ref_tokens = TOKEN_RE.findall(reference.lower())
    cand_tokens = TOKEN_RE.findall(candidate.lower())
    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    overlap = sum(min(ref_counts[token], cand_counts[token]) for token in ref_counts.keys() & cand_counts.keys())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def binary_f1(samples: List[SampleEval], positive: str = "FORGED") -> Tuple[float, Dict[str, int]]:
    tp = sum(1 for s in samples if s.gt_label == positive and s.pred_label == positive)
    fp = sum(1 for s in samples if s.gt_label != positive and s.pred_label == positive)
    fn = sum(1 for s in samples if s.gt_label == positive and s.pred_label != positive)
    tn = sum(1 for s in samples if s.gt_label != positive and s.pred_label != positive)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom if denom else 0.0), {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def to_dict(sample: SampleEval) -> Dict[str, Any]:
    return {
        "image_name": sample.image_name,
        "sample_id": sample.sample_id,
        "language_code": sample.language_code,
        "expected_script": sample.expected_script,
        "pred_script": sample.pred_script,
        "gt_label": sample.gt_label,
        "pred_label": sample.pred_label,
        "det_correct": sample.det_correct,
        "loc_score": sample.loc_score,
        "loc_method": sample.loc_method,
        "box_miou": sample.box_miou,
        "mask_iou": sample.mask_iou,
        "mask_f1": sample.mask_f1,
        "count_score": sample.count_score,
        "exp_score": sample.exp_score,
        "rep_score": sample.rep_score,
        "rep_breakdown": sample.rep_breakdown,
        "pred_boxes": sample.pred_boxes,
        "gt_boxes": sample.gt_boxes,
        "pred_report_len": sample.pred_report_len,
        "gt_report_len": sample.gt_report_len,
        "issues": sample.issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-jsonl", required=True, help="Split JSONL containing image_file/report_text/mask_path")
    pred_group = parser.add_mutually_exclusive_group(required=True)
    pred_group.add_argument("--pred", help="CodaBench-style .zip, prediction.jsonl.gz, or plain prediction JSONL")
    pred_group.add_argument("--raw-jsonl", help="Raw pipeline JSONL; report is read from raw_output/report")
    parser.add_argument("--output", default="outputs/eval/competition_aligned_scores.json")
    parser.add_argument("--output-csv", default="", help="Optional per-sample CSV diagnostics")
    parser.add_argument("--skip-bertscore", action="store_true", help="Use token-F1 fallback instead of BERTScore")
    parser.add_argument(
        "--bert-model",
        default="bert-base-uncased",
        help=(
            "BERT model for S_Exp. Default matches CodaBench legacy logs "
            "(bert-base-uncased + bert-score==0.3.13). Use bert-base-multilingual-cased "
            "to score zh/th/ar GT more fairly during local model selection."
        ),
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument(
        "--det-metric",
        choices=["accuracy", "f1", "macro_f1"],
        default="accuracy",
        help="Aggregation for S_Det (accuracy is the most likely official choice on the balanced test split).",
    )
    parser.add_argument("--loc-metric", choices=["f1", "iou"], default="f1")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = evaluate all common samples")
    parser.add_argument("--include-samples", action="store_true", help="Write per-sample details into output JSON")
    parser.add_argument(
        "--allow-empty-loc",
        action="store_true",
        help="If set, missing FORGED loc scores fall back to 0 instead of being dropped from the average.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Also report a calibrated S_Fin estimate. Calibration is based on the v0.2 "
            "anchor (val_1000 proxy=0.4004 -> leaderboard=0.2409) and damps S_Rep_proxy "
            "by a factor derived from that anchor. Recompute as more anchors arrive."
        ),
    )
    return parser.parse_args()


def macro_f1(samples: List[SampleEval]) -> float:
    f1_forged, _ = binary_f1(samples, positive="FORGED")
    f1_authentic, _ = binary_f1(samples, positive="AUTHENTIC")
    return (f1_forged + f1_authentic) / 2.0


def aggregate_loc(samples: List[SampleEval], allow_empty: bool) -> Tuple[float, float, int, int]:
    """Return (mF1, mIoU, n_forged_scored, n_forged_total)."""
    forged = [s for s in samples if s.gt_label == "FORGED"]
    if not forged:
        return 0.0, 0.0, 0, 0
    f1_values: List[float] = []
    iou_values: List[float] = []
    for s in forged:
        if s.mask_f1 is not None:
            f1_values.append(s.mask_f1)
        elif s.loc_score is not None and s.loc_method != "no_gt_boxes":
            f1_values.append(s.loc_score)
        elif allow_empty:
            f1_values.append(0.0)
        if s.mask_iou is not None:
            iou_values.append(s.mask_iou)
        elif s.box_miou is not None:
            iou_values.append(s.box_miou)
        elif allow_empty:
            iou_values.append(0.0)
    n_scored = len(f1_values)
    return mean(f1_values), mean(iou_values), n_scored, len(forged)


def main() -> None:
    args = parse_args()
    gt_rows = read_jsonl(Path(args.gt_jsonl))
    pred_path = Path(args.pred or args.raw_jsonl)
    pred_rows = load_predictions(pred_path, raw_jsonl=bool(args.raw_jsonl))

    gt_by_name = {str(row.get("image_file") or row.get("image_name")): row for row in gt_rows}
    common = sorted(set(gt_by_name) & set(pred_rows))
    if args.max_samples > 0:
        common = common[: args.max_samples]
    if not common:
        raise SystemExit("No common image_name values between GT and predictions.")

    coverage = len(common) / max(1, len(gt_rows))
    if coverage < 0.95:
        print(
            f"WARNING: predictions cover {coverage:.1%} of GT ({len(common)}/{len(gt_rows)}). "
            "Missing samples are silently dropped from all averages.",
            file=sys.stderr,
        )

    has_gt_reports = any(
        bool(row.get("report_text") or row.get("report")) for row in gt_rows
    )
    if not has_gt_reports:
        print(
            "WARNING: GT split has no report_text — S_Exp will fall back to 0.0 and S_Rep cannot be referenced. "
            "This evaluator is intended for splits that ship GT reports (val/debug), not the hidden test split.",
            file=sys.stderr,
        )

    samples = [evaluate_sample(gt_by_name[name], pred_rows[name]) for name in common]

    det_accuracy = mean([1.0 if sample.det_correct else 0.0 for sample in samples])
    det_f1, binary_confusion = binary_f1(samples)
    det_macro_f1 = macro_f1(samples)
    s_det_map = {"accuracy": det_accuracy, "f1": det_f1, "macro_f1": det_macro_f1}
    s_det = s_det_map[args.det_metric]

    loc_f1, loc_iou, n_loc_scored, n_forged = aggregate_loc(samples, allow_empty=args.allow_empty_loc)
    s_loc = loc_f1 if args.loc_metric == "f1" else loc_iou
    authentic_samples = [s for s in samples if s.gt_label == "AUTHENTIC"]
    authentic_no_box_rate = mean([
        1.0 if s.pred_boxes == 0 else 0.0 for s in authentic_samples
    ]) if authentic_samples else 0.0
    s_rep = mean([sample.rep_score for sample in samples])

    s_exp = 0.0
    s_exp_method = "token_f1_proxy"
    bert_scores: List[float] = []
    gt_reports = [str(gt_by_name[name].get("report_text") or gt_by_name[name].get("report") or "") for name in common]
    pred_reports = [str(pred_rows[name].get("report") or "") for name in common]
    token_scores = [token_f1(gt, pred) for gt, pred in zip(gt_reports, pred_reports)]
    if not has_gt_reports:
        bert_scores = [0.0] * len(common)
        s_exp_method = "no_gt_reports"
    elif args.skip_bertscore:
        s_exp = mean(token_scores)
        bert_scores = token_scores
    else:
        s_exp, bert_scores = compute_bertscore(
            gt_reports,
            pred_reports,
            model_type=args.bert_model,
            batch_size=args.bertscore_batch_size,
        )
        if bert_scores:
            s_exp_method = f"bertscore_f1[{args.bert_model}]"
        else:
            s_exp = mean(token_scores)
            bert_scores = token_scores

    for sample, exp_score in zip(samples, bert_scores):
        sample.exp_score = exp_score

    s_fin = (
        WEIGHTS["S_Det"] * s_det
        + WEIGHTS["S_Loc"] * s_loc
        + WEIGHTS["S_Exp"] * s_exp
        + WEIGHTS["S_Rep"] * s_rep
    )

    s_rep_calibrated: Optional[float] = None
    s_fin_calibrated: Optional[float] = None
    if args.calibrate:
        s_rep_calibrated = calibrate_s_rep(s_rep)
        s_fin_calibrated = (
            WEIGHTS["S_Det"] * s_det
            + WEIGHTS["S_Loc"] * s_loc
            + WEIGHTS["S_Exp"] * s_exp
            + WEIGHTS["S_Rep"] * s_rep_calibrated
        )

    confusion: Counter[Tuple[str, str]] = Counter((s.gt_label, s.pred_label) for s in samples)
    issues: Counter[str] = Counter(issue for sample in samples for issue in sample.issues)
    loc_methods: Counter[str] = Counter(sample.loc_method for sample in samples)
    rep_breakdown_avg: Dict[str, float] = {}
    if samples:
        keys = sorted({k for s in samples for k in s.rep_breakdown.keys()})
        for k in keys:
            vals = [s.rep_breakdown.get(k) for s in samples]
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            if vals:
                rep_breakdown_avg[k] = float(np.mean(vals))

    by_language: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[SampleEval]] = defaultdict(list)
    for sample in samples:
        grouped[sample.language_code].append(sample)
    for lang, lang_samples in sorted(grouped.items()):
        forged_lang = [s for s in lang_samples if s.gt_label == "FORGED"]
        loc_f1_lang, loc_iou_lang, _, _ = aggregate_loc(lang_samples, allow_empty=args.allow_empty_loc)
        by_language[lang] = {
            "n": len(lang_samples),
            "n_forged": len(forged_lang),
            "S_Det_accuracy": mean([1.0 if s.det_correct else 0.0 for s in lang_samples]),
            "S_Det_f1": binary_f1(lang_samples)[0],
            "S_Loc_mF1_proxy": loc_f1_lang,
            "mIoU_proxy": loc_iou_lang,
            "S_Rep_proxy": mean([s.rep_score for s in lang_samples]),
            "avg_pred_report_len": mean([float(s.pred_report_len) for s in lang_samples]),
            "avg_gt_report_len": mean([float(s.gt_report_len) for s in lang_samples]),
            "script_match_rate": mean([
                1.0 if s.pred_script == s.expected_script else 0.0 for s in lang_samples
            ]),
        }

    if bert_scores:
        for sample, bert in zip(samples, bert_scores):
            by_language[sample.language_code].setdefault("bertscore_values", []).append(bert)
        for lang_stats in by_language.values():
            values = lang_stats.pop("bertscore_values", [])
            lang_stats["S_Exp_proxy"] = mean(values)

    output = {
        "metric_note": (
            "Local proxy. S_Loc is mean over FORGED samples only "
            "(authentic_no_box_rate is reported separately). "
            "S_Rep is a deterministic report-quality proxy that decouples format/language/length "
            "from S_Det; it is not the hidden CodaBench LLM judge."
        ),
        "prediction_file": str(pred_path),
        "ground_truth_file": str(Path(args.gt_jsonl)),
        "num_gt": len(gt_rows),
        "num_pred": len(pred_rows),
        "num_common": len(common),
        "num_forged": n_forged,
        "num_forged_loc_scored": n_loc_scored,
        "weights": WEIGHTS,
        "selected_proxy_metrics": {
            "S_Det": args.det_metric,
            "S_Loc": args.loc_metric,
            "S_Exp_method": s_exp_method,
            "bert_model": args.bert_model,
            "allow_empty_loc": args.allow_empty_loc,
        },
        "scores": {
            "S_Det": s_det,
            "S_Det_accuracy": det_accuracy,
            "S_Det_f1": det_f1,
            "S_Det_macro_f1": det_macro_f1,
            "S_Loc": s_loc,
            "S_Loc_mF1_proxy": loc_f1,
            "mIoU_proxy": loc_iou,
            "authentic_no_box_rate": authentic_no_box_rate,
            "S_Exp": s_exp,
            "S_Exp_method": s_exp_method,
            "S_Rep_proxy": s_rep,
            "S_Rep_calibrated": s_rep_calibrated,
            "S_Fin_proxy": s_fin,
            "S_Fin_calibrated": s_fin_calibrated,
        },
        "calibration_anchors": CALIBRATION_ANCHORS if args.calibrate else None,
        "binary_confusion": binary_confusion,
        "confusion": {f"{gt}->{pred}": count for (gt, pred), count in sorted(confusion.items())},
        "loc_methods": dict(sorted(loc_methods.items())),
        "rep_breakdown_avg": rep_breakdown_avg,
        "top_issues": dict(issues.most_common(20)),
        "by_language": by_language,
    }
    if args.include_samples:
        output["samples"] = [to_dict(sample) for sample in samples]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "image_name",
            "sample_id",
            "language_code",
            "expected_script",
            "pred_script",
            "gt_label",
            "pred_label",
            "det_correct",
            "loc_score",
            "loc_method",
            "box_miou",
            "mask_iou",
            "mask_f1",
            "count_score",
            "exp_score",
            "rep_score",
            "rep_format",
            "rep_self_consistency",
            "rep_verdict_match",
            "rep_grounding",
            "rep_count_align",
            "rep_detail",
            "rep_length_align",
            "rep_language_factor",
            "pred_boxes",
            "gt_boxes",
            "pred_report_len",
            "gt_report_len",
            "issues",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for sample in samples:
                row = to_dict(sample)
                bd = row.pop("rep_breakdown", {}) or {}
                row["rep_format"] = bd.get("format")
                row["rep_self_consistency"] = bd.get("self_consistency")
                row["rep_verdict_match"] = bd.get("verdict_match")
                row["rep_grounding"] = bd.get("grounding")
                row["rep_count_align"] = bd.get("count_align")
                row["rep_detail"] = bd.get("detail")
                row["rep_length_align"] = bd.get("length_align")
                row["rep_language_factor"] = bd.get("language_factor")
                row["issues"] = ";".join(row.get("issues") or [])
                writer.writerow({field: row.get(field) for field in fields})

    print("=== Competition-aligned local evaluation ===")
    print(f"GT={len(gt_rows)} Pred={len(pred_rows)} Common={len(common)} Forged={n_forged} (scored {n_loc_scored})")
    print(f"S_Det       : {s_det:.4f} ({args.det_metric}; acc={det_accuracy:.4f}, f1={det_f1:.4f}, macro={det_macro_f1:.4f})")
    print(f"S_Loc       : {s_loc:.4f} ({args.loc_metric}; mF1={loc_f1:.4f}, mIoU={loc_iou:.4f}, authentic_no_box={authentic_no_box_rate:.4f})")
    print(f"S_Exp       : {s_exp:.4f} ({s_exp_method})")
    print(f"S_Rep_proxy : {s_rep:.4f}  breakdown={ {k: round(v,3) for k,v in rep_breakdown_avg.items()} }")
    print(f"S_Fin_proxy : {s_fin:.4f}")
    if args.calibrate and s_fin_calibrated is not None:
        print(f"S_Rep_calib : {s_rep_calibrated:.4f}  (fitted from v0.2 anchor)")
        print(f"S_Fin_calib : {s_fin_calibrated:.4f}  (estimated leaderboard score)")
    print(f"loc_methods : {dict(sorted(loc_methods.items()))}")
    print(f"top_issues  : {dict(issues.most_common(8))}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
