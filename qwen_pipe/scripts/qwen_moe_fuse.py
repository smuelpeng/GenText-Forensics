#!/usr/bin/env python3
"""GT-blind Qwen-pipe expert fusion.

This script keeps Qwen-pipe separate from ``debug_distribution`` while reusing
its evaluator-compatible report parser. It treats existing Qwen runs as experts:
one primary report stream plus optional donor streams for conservative rescue
or grounding reuse.

No GT labels, GT reports, masks, or evaluator-only fields are read here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"


DEFAULT_RESCUE_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("redaction",),
    ("redacted",),
    ("black block",),
    ("black box",),
    ("covered",),
    ("obscured",),
    ("hidden",),
    ("erased",),
    ("overwritten",),
    ("copy-paste",),
    ("copy paste",),
    ("splice",),
    ("splicing",),
    ("paste boundary",),
    ("localized corruption",),
    ("digital noise",),
    ("corrupted rendering",),
    ("amount mismatch",),
    ("date mismatch",),
    ("name mismatch",),
    ("calculation error",),
    ("inconsistent total",),
    ("mathematical inconsistency",),
    ("遮挡",),
    ("遮蔽",),
    ("涂黑",),
    ("抹除",),
    ("覆盖",),
    ("拼接",),
    ("边界",),
    ("邊界",),
    ("金额", "不一致"),
    ("金額", "不一致"),
    ("日期", "不一致"),
    ("计算", "错误"),
    ("計算", "錯誤"),
    ("ปิดทับ",),
    ("ลบ",),
    ("เบลอ",),
    ("จำนวนเงิน",),
    ("วันที่",),
    ("حجب",),
    ("محجوب",),
    ("طمس",),
    ("محو",),
    ("المبلغ",),
    ("التاريخ",),
)

ARTIFACT_RESCUE_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("redaction",),
    ("redacted",),
    ("object insertion",),
    ("white box",),
    ("white block",),
    ("black block",),
    ("black box",),
    ("black solid square",),
    ("blue block",),
    ("pixelated block",),
    ("localized corruption",),
    ("digital noise",),
    ("corrupted rendering",),
    ("replacement character",),
    ("tofu",),
    ("low-resolution raster",),
    ("raster image",),
    ("jagged", "pixelated", "raster"),
    ("lower resolution", "raster"),
    ("low-quality raster",),
    ("低分辨率", "光栅"),
    ("低解析度", "光柵"),
    ("黑色实心方块",),
    ("黑色實心方塊",),
    ("白色", "遮挡"),
    ("白色", "遮蔽"),
    ("กล่อง", "ทับ"),
    ("ปิดบัง",),
    ("كتلة سوداء",),
    ("حجب", "مربع"),
)

PHYSICAL_ARTIFACT_PLUS_TRIGGERS: tuple[tuple[str, ...], ...] = ARTIFACT_RESCUE_TRIGGERS + (
    ("font rendering defect",),
    ("font rendering defects",),
    ("pixelation",),
    ("jagged edges",),
    ("jagged", "misaligned"),
    ("anti-aliasing",),
    ("black rectangle",),
    ("black rectangular block",),
    ("black band",),
    ("blue", "pixelated"),
    ("pixelated", "blue"),
    ("glyph", "pixelated"),
    ("glyph", "jagged"),
    ("黑色矩形",),
    ("黑色条带",),
    ("黑色遮挡块",),
    ("渲染", "缺陷"),
    ("字形", "破坏"),
    ("残留像素",),
    ("พิกเซล",),
    ("pixelated berwarna biru",),
    ("garis hitam",),
    ("dihapus", "garis hitam"),
)

SAFE_ARTIFACT_PLUS_TRIGGERS: tuple[tuple[str, ...], ...] = ARTIFACT_RESCUE_TRIGGERS + (
    ("black rectangle",),
    ("black rectangular block",),
    ("black band",),
    ("blue", "pixelated"),
    ("pixelated", "blue"),
    ("黑色矩形",),
    ("黑色条带",),
    ("黑色遮挡块",),
    ("pixelated berwarna biru",),
    ("garis hitam",),
    ("dihapus", "garis hitam"),
)

PRECISION_ARTIFACT_PLUS_TRIGGERS: tuple[tuple[str, ...], ...] = SAFE_ARTIFACT_PLUS_TRIGGERS + (
    ("font weight",),
    ("pixelation", "jagged edges", "anti-aliasing"),
    ("巨大", "垂直空白"),
    ("行距", "不均匀"),
    ("殘留像素",),
    ("語義缺失",),
    ("字形", "破壞"),
)

LOW_RISK_RESCUE_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("pixelated berwarna biru",),
    ("blok pixelated", "biru"),
)

TRIGGER_PROFILES = {
    "broad": DEFAULT_RESCUE_TRIGGERS,
    "artifact": ARTIFACT_RESCUE_TRIGGERS,
    "artifact_plus": PHYSICAL_ARTIFACT_PLUS_TRIGGERS,
    "artifact_safe_plus": SAFE_ARTIFACT_PLUS_TRIGGERS,
    "artifact_precision_plus": PRECISION_ARTIFACT_PLUS_TRIGGERS,
}

RESCUE_BLOCKERS: tuple[tuple[str, ...], ...] = (
    # Creator/export residue is often real document provenance, not localized
    # tamper evidence. Keep this blocker rescue-only so primary forged verdicts
    # are not rewritten.
    ("raw file system path",),
    ("file system path", "template"),
    ("internal metadata", "template"),
    ("template", "exported directly"),
    ("template identifier",),
    ("unredacted internal", "path"),
)


GROUNDING_RE = re.compile(r"\[GROUNDING\]\s*:\s*\[([^\[\]]+)\]", re.IGNORECASE)


def setup_debug_import(debug_root: Path) -> None:
    docshield_dir = debug_root / "baselines" / "DocShield"
    if not docshield_dir.exists():
        raise SystemExit(f"DocShield helper directory not found: {docshield_dir}")
    sys.path.insert(0, str(docshield_dir))


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row_key(row)
            if key:
                rows[key] = row
    return rows


def row_key(row: dict[str, Any]) -> str:
    return str(
        row.get("sample_id")
        or row.get("image_name")
        or Path(str(row.get("image_path") or "")).stem
        or ""
    )


def parse_expert_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expert must use name=path")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("expert name is empty")
    return name, Path(raw_path.strip()).expanduser()


def resolve_path(path: Path, *, debug_root: Path) -> Path:
    if path.is_absolute():
        return path
    pipe_path = PIPE_ROOT / path
    if pipe_path.exists():
        return pipe_path
    return debug_root / path


def parse_report(row: dict[str, Any]) -> dict[str, Any]:
    parsed = row.get("parsed") or {}
    if parsed.get("conclusion"):
        return parsed
    from postprocess import parse_cct_report  # type: ignore

    return parse_cct_report(str(row.get("raw_output") or ""))


def conclusion(row: dict[str, Any]) -> str:
    return str(parse_report(row).get("conclusion") or "UNKNOWN").upper()


def risk_score(row: dict[str, Any]) -> int:
    try:
        return int(parse_report(row).get("risk_score") or 0)
    except Exception:
        return 0


def anomaly_count(row: dict[str, Any]) -> int:
    return len(parse_report(row).get("anomalies") or [])


def grounding_count(row: dict[str, Any]) -> int:
    raw = str(row.get("raw_output") or "")
    return len(GROUNDING_RE.findall(raw))


def trigger_match(text: str, triggers: tuple[tuple[str, ...], ...]) -> tuple[bool, tuple[str, ...]]:
    lowered = text.lower()
    for group in triggers:
        if all(term.lower() in lowered for term in group):
            return True, group
    return False, ()


def should_rescue(
    base_row: dict[str, Any],
    donor_row: dict[str, Any],
    *,
    donor_name: str,
    min_risk: int,
    min_anomalies: int,
    require_trigger: bool,
    triggers: tuple[tuple[str, ...], ...],
) -> tuple[bool, dict[str, Any]]:
    donor_raw = str(donor_row.get("raw_output") or "")
    matched, trigger = trigger_match(donor_raw, triggers)
    blocked, blocker = trigger_match(donor_raw, RESCUE_BLOCKERS)
    matched_low_risk, low_risk_trigger = trigger_match(donor_raw, LOW_RISK_RESCUE_TRIGGERS)
    donor_conclusion = conclusion(donor_row)
    donor_risk = risk_score(donor_row)
    meta = {
        "donor": donor_name,
        "donor_conclusion": donor_conclusion,
        "donor_risk_score": donor_risk,
        "donor_anomaly_count": anomaly_count(donor_row),
        "donor_grounding_count": grounding_count(donor_row),
        "matched_trigger": list(trigger),
        "matched_blocker": list(blocker),
        "matched_low_risk_trigger": list(low_risk_trigger),
        "policy": "primary-AUTHENTIC rescue only; donor must be FORGED, localized, trigger-matched, and high-risk unless a narrow low-risk trigger is matched",
    }
    risk_ok = donor_risk >= min_risk or (donor_risk >= 75 and matched_low_risk)
    apply = (
        conclusion(base_row) == "AUTHENTIC"
        and donor_conclusion == "FORGED"
        and risk_ok
        and meta["donor_anomaly_count"] >= min_anomalies
        and meta["donor_grounding_count"] > 0
        and (matched or not require_trigger)
        and not blocked
    )
    meta["applied"] = apply
    return apply, meta


def fuse_rows(
    base_rows: dict[str, dict[str, Any]],
    donors: list[tuple[str, dict[str, dict[str, Any]]]],
    *,
    min_risk: int,
    min_anomalies: int,
    require_trigger: bool,
    triggers: tuple[tuple[str, ...], ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fused: list[dict[str, Any]] = []
    stats = {
        "rows": 0,
        "rescued": 0,
        "missing_donor": 0,
        "attempted_authentic_rows": 0,
        "rescue_by_donor": {},
    }

    for key, base_row in base_rows.items():
        stats["rows"] += 1
        out = dict(base_row)
        stage_outputs = dict(out.get("stage_outputs") or {})
        moe_attempts = []

        if conclusion(base_row) == "AUTHENTIC":
            stats["attempted_authentic_rows"] += 1
            for donor_name, donor_rows in donors:
                donor_row = donor_rows.get(key)
                if donor_row is None:
                    stats["missing_donor"] += 1
                    continue
                apply, meta = should_rescue(
                    base_row,
                    donor_row,
                    donor_name=donor_name,
                    min_risk=min_risk,
                    min_anomalies=min_anomalies,
                    require_trigger=require_trigger,
                    triggers=triggers,
                )
                moe_attempts.append(meta)
                if apply:
                    out = dict(donor_row)
                    out["sample_id"] = base_row.get("sample_id") or donor_row.get("sample_id")
                    out["image_name"] = base_row.get("image_name") or donor_row.get("image_name")
                    out["image_path"] = base_row.get("image_path") or donor_row.get("image_path")
                    out["width"] = base_row.get("width") or donor_row.get("width")
                    out["height"] = base_row.get("height") or donor_row.get("height")
                    out["pipeline"] = "qwen_pipe_moe"
                    out["model"] = f"{base_row.get('model', 'primary')}+{donor_name}"
                    stage_outputs = dict(out.get("stage_outputs") or {})
                    stats["rescued"] += 1
                    stats["rescue_by_donor"][donor_name] = stats["rescue_by_donor"].get(donor_name, 0) + 1
                    break

        out["parsed"] = parse_report(out)
        stage_outputs["qwen_pipe_moe"] = {
            "primary_conclusion": conclusion(base_row),
            "primary_risk_score": risk_score(base_row),
            "attempts": moe_attempts,
            "rescued": conclusion(base_row) == "AUTHENTIC" and conclusion(out) == "FORGED",
            "policy": {
                "min_risk": min_risk,
                "min_anomalies": min_anomalies,
                "require_trigger": require_trigger,
            },
        }
        out["stage_outputs"] = stage_outputs
        fused.append(out)

    return fused, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--donor", action="append", type=parse_expert_arg, default=[])
    parser.add_argument("--rescue-min-risk", type=int, default=85)
    parser.add_argument("--rescue-min-anomalies", type=int, default=1)
    parser.add_argument("--trigger-profile", choices=sorted(TRIGGER_PROFILES), default="broad")
    parser.add_argument("--no-require-trigger", action="store_true")
    parser.add_argument("--stats-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    debug_root = Path(args.debug_root).expanduser().resolve()
    setup_debug_import(debug_root)

    base_path = resolve_path(Path(args.base_jsonl).expanduser(), debug_root=debug_root)
    output_path = Path(args.output_jsonl).expanduser()
    if not output_path.is_absolute():
        output_path = PIPE_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    donors = [
        (name, load_jsonl(resolve_path(path.expanduser(), debug_root=debug_root)))
        for name, path in args.donor
    ]
    base_rows = load_jsonl(base_path)
    fused, stats = fuse_rows(
        base_rows,
        donors,
        min_risk=args.rescue_min_risk,
        min_anomalies=args.rescue_min_anomalies,
        require_trigger=not args.no_require_trigger,
        triggers=TRIGGER_PROFILES[args.trigger_profile],
    )

    with output_path.open("w", encoding="utf-8") as fh:
        for row in fused:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats.update(
        {
            "base_jsonl": str(base_path),
            "output_jsonl": str(output_path),
            "donors": [name for name, _ in donors],
            "rescue_min_risk": args.rescue_min_risk,
            "rescue_min_anomalies": args.rescue_min_anomalies,
            "require_trigger": not args.no_require_trigger,
            "trigger_profile": args.trigger_profile,
        }
    )
    if args.stats_json:
        stats_path = Path(args.stats_json).expanduser()
        if not stats_path.is_absolute():
            stats_path = PIPE_ROOT / stats_path
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
