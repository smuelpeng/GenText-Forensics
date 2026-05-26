"""Cross-Cues-aware Chain of Thought (CCT) prompt for DocShield reproduction.

Implements the 6-stage reasoning protocol from
DocShield: Towards AI Document Safety via Evidence-Grounded Agentic Reasoning
(arXiv:2604.02694, §3.2 + Algorithm 1).

The prompt is split into a system message and a user message so that base
Qwen-VL chat templates apply correctly. The model is asked to produce its
reasoning inside <think>...</think> and the final structured forensic report
inside <report>...</report>. Downstream postprocessing (postprocess.py) extracts
the <report> body and feeds it back into the GenText pipeline as ``raw_output``,
matching the schema expected by ``gentext_agent.py make-prediction``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are an expert document forensics examiner specializing in text-centric "
    "image forgery analysis. Apply Cross-Cues-aware Chain of Thought (CCT) "
    "reasoning: cross-validate visual artifacts with logical/semantic anomalies "
    "before reaching a verdict. Be conservative — most documents are AUTHENTIC; "
    "only flag a document as FORGED when at least one anomaly survives joint "
    "visual-and-logical cross-validation."
)


USER_PROMPT = """Analyze the image step by step using CCT reasoning, then synthesize the structured forensic report.

# CCT Reasoning Protocol — output inside <think>...</think>

**Stage 1 — Knowledge Preparation**: Identify the document type/scenario and the manipulation typologies most likely for this kind of document. Briefly extract the key textual content with approximate locations (no need for full OCR).

**Stage 2 — Visual Cues Extraction**: Inspect for visual artifacts: font rendering inconsistencies, edge aliasing, chromatic aberration, copy-paste boundaries, lighting/print inconsistency. List candidate visual cues with locations. If none, say so.

**Stage 3 — Logical Cues Extraction**: Check for textual contradictions: math errors, date impossibilities, sequence violations, scenario anachronisms, contextual inconsistencies. List candidate logical cues. If none, say so.

**Stage 4 — Cross-Cues Validation & Filtering**: For each candidate, judge whether it is supported by both modalities or by a single strong signal. Discard isolated low-confidence findings. Note explicit complementary checks (e.g. "logical contradiction confirmed despite clean visuals").

**Stage 5 — Spatial Grounding**: For each surviving anomaly, give pixel coordinates [x1, y1, x2, y2] using the image's native resolution. Skip this stage if the verdict is AUTHENTIC.

**Stage 6 — Verdict**: Decide FORGED if at least one cross-validated anomaly remains, otherwise AUTHENTIC. Briefly justify in 1–2 sentences.

Keep each stage concise (2–5 sentences) so that the final report fits within the token budget.

# Final Report — output inside <report>...</report>

Use exactly this Markdown skeleton (do not invent extra sections, do not wrap it in code fences):

# FORGERY ANALYSIS  REPORT

**Report ID:** FAR-xxxx-xx-xx
**Date of Examination:** xxxx-xx-xx
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** FORGED or AUTHENTIC
    **[RISK_SCORE]:** integer between 0 and 100

---

## DETAILED ANOMALY ANALYSIS

If FORGED, write one block per surviving anomaly:

### ANOMALY_001: <category> (<short location description>)
[GROUNDING]:[x1, y1, x2, y2]
[REASON]: <at least 200 characters, written in the document's language, citing both visual and logical evidence whenever possible>

### ANOMALY_002: ...
...

If AUTHENTIC, replace the anomaly blocks with exactly:
No anomalies detected. The document has been thoroughly examined and no signs of tampering, alteration, or forgery were found.

---

## SUMMARY
The examination of the document has identified <N> anomalies, resulting in a fraud risk score of <score>. <200–400 characters describing the document content in the document's language.>

---
**END OF REPORT**

# Important rules

- Anomaly categories must be one of: `Visual Clumsy`, `Logical Fraud`, `semantic_subtle`, `text tampering`.
- The REASON and SUMMARY fields must use the same language as the document text (English, Chinese, Indonesian, Thai, Arabic, Malay, etc.).
- Pixel coordinates must be integers, ordered as [x1, y1, x2, y2] with x1<x2 and y1<y2.
- Wrap your reasoning strictly in <think>...</think> and your final report strictly in <report>...</report>. Do not output anything outside these two blocks."""


def build_messages(
    image_path: str | Path,
    *,
    max_pixels: int | None = None,
    use_file_uri: bool = True,
) -> list[dict[str, Any]]:
    """Construct OpenAI-style chat messages for Qwen-VL.

    Args:
        image_path: absolute or relative path to the image on disk.
        max_pixels: optional pixel budget passed through to qwen_vl_utils.
        use_file_uri: when True (default) wrap the path as a ``file://`` URI,
            which is what ``qwen_vl_utils.process_vision_info`` expects.
    """

    path = Path(image_path)
    image_field: str
    if use_file_uri:
        image_field = path.resolve().as_uri()
    else:
        image_field = str(path)

    image_content: dict[str, Any] = {"type": "image", "image": image_field}
    if max_pixels is not None:
        image_content["max_pixels"] = max_pixels

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [image_content, {"type": "text", "text": USER_PROMPT}],
        },
    ]


def system_user_pair() -> tuple[str, str]:
    """Return the (system, user) string pair, useful for SFT data construction."""

    return SYSTEM_PROMPT, USER_PROMPT
