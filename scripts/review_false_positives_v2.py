#!/usr/bin/env python3
"""GT-blind false-positive reviewer for staged DocShield outputs.

The reviewer targets low-evidence forged reports where the explanation is
dominated by speculative world-knowledge, OCR/font/layout, spelling, or generic
"AI-generated" claims rather than hard localized tamper evidence.
"""

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
from run_staged_docshield_api import authentic_downgrade_report, ensure_report_structure  # noqa: E402


ANOMALY_RE = re.compile(r"###\s*ANOMALY", re.IGNORECASE)
GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[([^\[\]]+)\]", re.IGNORECASE)

WEAK_SPECULATION_TERMS = (
    "ai-generated",
    "automated generation",
    "synthetic",
    "fabricated",
    "fabrication",
    "generated content",
    "lack of human proofreading",
    "poor proofreading",
    "format word",
    "untranslated english",
    "english technical term",
    "without supporting content",
    "empty page",
    "common artifact",
    "common artefact",
    "spelling",
    "grammar",
    "capitalization",
    "line spacing",
    "font",
    "rendering",
    "pixelated",
    "formatting",
    "layout",
    "official document should",
    "not standard",
    "does not match any",
    "appears",
    "seems",
    "suggests",
    "historical",
    "common sense",
    "plausibility",
    "虛構",
    "虚构",
    "生成",
    "歷史",
    "历史",
    "常識",
    "常识",
    "缺乏",
    "不符合",
    "排版",
    "格式",
    "字体",
    "字體",
    "字形",
    "渲染",
    "錯別字",
    "错别字",
    "校對",
    "校对",
    "ตัวอักษร",
    "การจัดวาง",
    "เว้นบรรทัด",
    "ว่างเปล่า",
    "ไม่มีเนื้อหา",
    "พื้นที่ว่าง",
    "เกือบทั้งหมด",
    "ซ้ำ",
    "สะกด",
    "สร้าง",
    "มาตรฐาน",
    "คุณภาพ",
    "تنسيق",
    "مصطلح إنجليزي",
    "عبارة إنجليزية",
    "لاتينية",
    "لغوية",
    "إملائية",
    "خط",
    "تشوه",
    "غير رسمي",
    "وهمي",
    "لا يتطابق",
    "مزيف",
    "عشوائي",
)

HARD_EVIDENCE_TERMS = (
    "redaction",
    "redacted",
    "black block",
    "black box",
    "covered",
    "obscured",
    "hidden",
    "erased",
    "deleted",
    "overwritten",
    "copy-paste",
    "copy paste",
    "copy-move",
    "paste boundary",
    "splicing",
    "splice",
    "altered total",
    "amount mismatch",
    "date mismatch",
    "name mismatch",
    "calculation",
    "sum",
    "subtotal",
    "impossible date",
    "qr code",
    "barcode",
    "seal",
    "stamp",
    "signature",
    "logo conflict",
    "serial",
    "遮挡",
    "遮蔽",
    "涂黑",
    "模糊",
    "抹除",
    "刪除",
    "删除",
    "拼接",
    "邊界",
    "边界",
    "金額",
    "金额",
    "日期",
    "姓名",
    "錯誤計算",
    "错误计算",
    "เบลอ",
    "ปิดทับ",
    "ลบ",
    "จำนวนเงิน",
    "วันที่",
    "ชื่อ",
    "حجب",
    "محجوب",
    "طمس",
    "محو",
    "التاريخ",
    "المبلغ",
    "الاسم",
)

LOCALIZED_PROTECT_TERMS = (
    "compared to adjacent",
    "compared to the adjacent",
    "compared with adjacent",
    "adjacent fax",
    "adjacent text",
    "date column",
    "phone number",
    "fax number",
    "missing characters",
    "missing chars",
    "incorrect spacing",
    "garbled as",
)

LOGICAL_SINGLE_PROTECT_TERMS = (
    "financial forecast",
    "财务预测",
    "基本每股收益",
    "固定资本",
    "固定资产",
    "会计科目",
    "會計科目",
    "bibliografi",
    "daftar pustaka",
    "referensi",
    "cak nun",
    "出版日期",
    "報名截止",
    "报名截止",
    "hierarki indeks",
    "konten indeks",
    "pegawai",
)

VISUAL_ARTIFACT_DOWNGRADE_TERMS = (
    "html tag",
    "<br>",
    "white band",
    "white seam",
)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def count_hits(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered)


def wide_box_ratio(report: str, width: int, height: int) -> float:
    boxes = 0
    wide = 0
    page_area = max(1, width * height)
    for match in GROUNDING_RE.finditer(report or ""):
        nums = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        if len(nums) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in nums[:4]]
        except ValueError:
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        boxes += 1
        wide += area / page_area >= 0.35
    return wide / boxes if boxes else 0.0


def should_downgrade(
    row: dict[str, Any],
    *,
    max_risk: int,
    max_anomalies: int,
    min_weak_hits: int,
    max_hard_hits: int,
    allow_wide_exception: bool,
) -> tuple[bool, dict[str, Any]]:
    report = str(row.get("raw_output") or "")
    parsed = parse_cct_report(report)
    risk = int(parsed.get("risk_score") or 0)
    anomaly_count = len(ANOMALY_RE.findall(report))
    weak_hits = count_hits(report, WEAK_SPECULATION_TERMS)
    hard_hits = count_hits(report, HARD_EVIDENCE_TERMS)
    protect_hits = count_hits(report, LOCALIZED_PROTECT_TERMS)
    logical_protect_hits = count_hits(report, LOGICAL_SINGLE_PROTECT_TERMS)
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    wide_ratio = wide_box_ratio(report, width, height) if width and height else 0.0
    protected = anomaly_count >= 2 and hard_hits >= 1 and protect_hits >= 2
    lowered_report = report.lower()
    logical_single_downgrade = (
        parsed.get("conclusion") == "FORGED"
        and risk <= max_risk
        and anomaly_count == 1
        and "logical fraud" in lowered_report
        and "visual clumsy" not in lowered_report
        and logical_protect_hits == 0
    )
    visual_artifact_downgrade = (
        parsed.get("conclusion") == "FORGED"
        and risk <= max_risk
        and anomaly_count <= 2
        and any(term in lowered_report for term in VISUAL_ARTIFACT_DOWNGRADE_TERMS)
    )

    downgrade = (
        logical_single_downgrade
        or visual_artifact_downgrade
        or (
            parsed.get("conclusion") == "FORGED"
            and risk <= max_risk
            and anomaly_count <= max_anomalies
            and weak_hits >= min_weak_hits
            and hard_hits <= max_hard_hits
            and not protected
        )
    )
    if (
        allow_wide_exception
        and not downgrade
        and parsed.get("conclusion") == "FORGED"
        and risk <= max_risk
        and anomaly_count <= max_anomalies
        and weak_hits >= min_weak_hits + 2
        and hard_hits <= max(max_hard_hits, 1)
        and wide_ratio >= 0.5
        and not protected
    ):
        downgrade = True

    return downgrade, {
        "risk_score": risk,
        "anomaly_count": anomaly_count,
        "weak_hits": weak_hits,
        "hard_hits": hard_hits,
        "protect_hits": protect_hits,
        "logical_protect_hits": logical_protect_hits,
        "logical_single_downgrade": logical_single_downgrade,
        "visual_artifact_downgrade": visual_artifact_downgrade,
        "protected": protected,
        "wide_box_ratio": wide_ratio,
        "decision": "AUTHENTIC" if downgrade else parsed.get("conclusion", "UNKNOWN"),
        "applied": downgrade,
        "policy": (
            "Downgrade low-complexity forged reports dominated by speculative world-knowledge, "
            "font/OCR/layout/spelling, or generic AI-generation claims without hard localized tamper evidence; "
            "preserve reports with multiple concrete localized comparison cues."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-risk", type=int, default=90)
    parser.add_argument("--max-anomalies", type=int, default=2)
    parser.add_argument("--min-weak-hits", type=int, default=2)
    parser.add_argument("--max-hard-hits", type=int, default=0)
    parser.add_argument("--allow-wide-exception", action="store_true")
    args = parser.parse_args()

    in_path = repo_path(args.input_jsonl)
    out_path = repo_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    downgraded = 0
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            apply, meta = should_downgrade(
                row,
                max_risk=args.max_risk,
                max_anomalies=args.max_anomalies,
                min_weak_hits=args.min_weak_hits,
                max_hard_hits=args.max_hard_hits,
                allow_wide_exception=args.allow_wide_exception,
            )
            out = dict(row)
            stage_outputs = dict(out.get("stage_outputs") or {})
            stage_outputs["false_positive_reviewer_v2"] = meta
            if apply:
                ocr_layout = ((stage_outputs.get("ocr_layout") or {}).get("parsed") or {})
                report = ensure_report_structure(authentic_downgrade_report(ocr_layout))
                out["raw_output"] = report
                downgraded += 1
            out["parsed"] = parse_cct_report(str(out.get("raw_output") or ""))
            out["stage_outputs"] = stage_outputs
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} rows={total} downgraded={downgraded}")


if __name__ == "__main__":
    main()
