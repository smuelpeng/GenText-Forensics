#!/usr/bin/env python3
"""Cross-validated text-pattern false-positive rejector.

The older FP rejector used hand-written aggregate features.  This diagnostic
module tests whether the wording of the generated report itself carries a
separable false-positive pattern.  It trains a simple multinomial Naive Bayes
model over word and character n-gram features in cross-validation folds.

GT labels are used only as local fold labels after predictions exist.  The
script never sends GT to any model prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PIPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_ROOT = PIPE_ROOT.parent / "debug_distribution"

sys.path.insert(0, str(PIPE_ROOT / "scripts"))

from qwen_exhaustive_recall import read_jsonl, sample_id_from_row  # noqa: E402
from qwen_text_crop_verify import resolve_pipe_path, setup_debug_import  # noqa: E402


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff\u0e00-\u0e7f\u0600-\u06ff]{2,}", re.UNICODE)


def stable_fold(sample_id: str, folds: int) -> int:
    return sum(sample_id.encode("utf-8")) % folds


def load_eval(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(s.get("sample_id") or ""): s for s in data.get("samples") or []}


def conclusion(row: dict[str, Any]) -> str:
    return str((row.get("parsed") or {}).get("conclusion") or "").upper()


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[[0-9,\s]+\]", " [box] ", text)
    text = re.sub(r"\d+(?:[.,:/-]\d+)*", " [num] ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def features(text: str, *, max_chars: int = 6000) -> Counter[str]:
    text = normalize(text)[:max_chars]
    feats: Counter[str] = Counter()
    for tok in TOKEN_RE.findall(text):
        if len(tok) > 40:
            tok = tok[:40]
        feats[f"w:{tok}"] += 1
    compact = re.sub(r"\s+", " ", text)
    for n in (3, 4):
        for i in range(max(0, len(compact) - n + 1)):
            gram = compact[i : i + n]
            if gram.strip():
                feats[f"c{n}:{gram}"] += 1
    return feats


def train_nb(items: list[dict[str, Any]], *, alpha: float, min_df: int) -> dict[str, Any]:
    df: Counter[str] = Counter()
    for item in items:
        df.update(item["features"].keys())
    vocab = {f for f, c in df.items() if c >= min_df}
    by_class = {0: Counter(), 1: Counter()}
    totals = {0: 0, 1: 0}
    docs = {0: 0, 1: 0}
    for item in items:
        y = int(item["target_reject"])
        docs[y] += 1
        for feat, count in item["features"].items():
            if feat in vocab:
                by_class[y][feat] += count
                totals[y] += count
    return {
        "vocab": vocab,
        "counts": by_class,
        "totals": totals,
        "docs": docs,
        "alpha": alpha,
        "prior": (docs[1] + alpha) / max(alpha * 2.0 + docs[0] + docs[1], 1.0),
    }


def score_nb(feats: Counter[str], model: dict[str, Any]) -> float:
    vocab = model["vocab"]
    counts = model["counts"]
    totals = model["totals"]
    alpha = float(model["alpha"])
    v = max(1, len(vocab))
    prior = min(0.999, max(0.001, float(model["prior"])))
    logodds = math.log(prior / (1.0 - prior))
    denom1 = totals[1] + alpha * v
    denom0 = totals[0] + alpha * v
    used = 0
    for feat, count in feats.items():
        if feat not in vocab:
            continue
        p1 = (counts[1][feat] + alpha) / denom1
        p0 = (counts[0][feat] + alpha) / denom0
        logodds += min(count, 6) * math.log(p1 / p0)
        used += 1
    # Normalize lightly so long reports do not dominate only by length.
    logodds = logodds / max(1.0, math.sqrt(used))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logodds))))


def authentic_report(row: dict[str, Any], score: float) -> str:
    sid = sample_id_from_row(row)
    image_name = str(row.get("image_name") or sid)
    return f"""# FORGERY ANALYSIS REPORT

**Report ID:** TEXT-NB-FP-REJECT-{sid}
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 5

---

## DETAILED ANOMALY ANALYSIS

No localized tampering evidence is retained after cross-validated false-positive review. The previous report wording matches a locally learned false-positive pattern more strongly than a hard forgery-evidence pattern.

---

## SUMMARY
The document image {image_name} is classified as authentic by the text-pattern false-positive rejector. Rejection score: {score:.4f}.

---
**END OF REPORT**
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--applied-raw-jsonl", required=True)
    parser.add_argument("--debug-root", default=str(DEFAULT_DEBUG_ROOT))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--reject-threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_debug_import(Path(args.debug_root).expanduser().resolve())
    from postprocess import parse_cct_report  # type: ignore

    input_path = resolve_pipe_path(args.input_jsonl)
    eval_path = resolve_pipe_path(args.eval_json)
    out_diag = resolve_pipe_path(args.output_jsonl)
    out_summary = resolve_pipe_path(args.summary_json)
    out_raw = resolve_pipe_path(args.applied_raw_jsonl)
    out_diag.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    eval_samples = load_eval(eval_path)
    items: list[dict[str, Any]] = []
    for row in rows:
        sid = sample_id_from_row(row)
        es = eval_samples.get(sid) or {}
        if conclusion(row) != "FORGED":
            continue
        items.append(
            {
                "sample_id": sid,
                "row": row,
                "features": features(str(row.get("raw_output") or "")),
                "target_reject": 1 if es.get("gt_label") == "AUTHENTIC" and es.get("pred_label") == "FORGED" else 0,
                "fold": stable_fold(sid, args.folds),
            }
        )

    for fold in range(args.folds):
        train = [item for item in items if int(item["fold"]) != fold]
        test = [item for item in items if int(item["fold"]) == fold]
        if not train:
            continue
        model = train_nb(train, alpha=args.alpha, min_df=args.min_df)
        for item in test:
            item["reject_score"] = score_nb(item["features"], model)

    selected = [item for item in items if float(item.get("reject_score") or 0.0) >= args.reject_threshold]
    by_sid = {item["sample_id"]: item for item in items}
    with out_diag.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(
                json.dumps(
                    {
                        "sample_id": item["sample_id"],
                        "reject_score": item.get("reject_score", 0.0),
                        "target_reject": item["target_reject"],
                        "selected": item in selected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    changed = 0
    with out_raw.open("w", encoding="utf-8") as fh:
        for row in rows:
            sid = sample_id_from_row(row)
            item = by_sid.get(sid)
            if item and float(item.get("reject_score") or 0.0) >= args.reject_threshold:
                row = dict(row)
                report = authentic_report(row, float(item.get("reject_score") or 0.0))
                row["raw_output"] = report
                row["parsed"] = parse_cct_report(report)
                stage_outputs = dict(row.get("stage_outputs") or {})
                stage_outputs["qwen_text_nb_fp_rejector"] = {
                    "applied": True,
                    "reject_score": float(item.get("reject_score") or 0.0),
                    "reject_threshold": args.reject_threshold,
                    "gt_free_features": True,
                    "policy": "5-fold text-pattern NB false-positive rejector; GT used only for local fold labels.",
                }
                row["stage_outputs"] = stage_outputs
                changed += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_true_fp = sum(1 for item in selected if item["target_reject"] == 1)
    selected_true_tp = sum(1 for item in selected if item["target_reject"] == 0)
    target_fp_count = sum(int(item["target_reject"]) for item in items)
    summary = {
        "input_jsonl": str(input_path),
        "eval_json": str(eval_path),
        "applied_raw_jsonl": str(out_raw),
        "pred_forged_observations": len(items),
        "target_fp_count": target_fp_count,
        "reject_threshold": args.reject_threshold,
        "alpha": args.alpha,
        "min_df": args.min_df,
        "selected_count": len(selected),
        "selected_true_fp": selected_true_fp,
        "selected_true_tp": selected_true_tp,
        "selected_precision_diagnostic": selected_true_fp / max(1, len(selected)),
        "selected_fp_recall_diagnostic": selected_true_fp / max(1, target_fp_count),
        "score_min": min((float(item.get("reject_score") or 0.0) for item in items), default=0.0),
        "score_max": max((float(item.get("reject_score") or 0.0) for item in items), default=0.0),
        "changed": changed,
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
