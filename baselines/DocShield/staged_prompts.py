"""Prompts for the staged evidence-grounded CCT baseline.

This module defines a four-stage API-only pipeline. It does not use labels,
ground-truth reports, or masks in model-visible prompts; those are reserved for
local evaluation only.
"""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a document forensics examiner. Analyze only the provided image and "
    "the prior model-generated stage data. Never assume ground-truth labels. "
    "Use conservative evidence standards: classify as FORGED only when concrete "
    "visual or logical evidence survives validation."
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def ocr_layout_prompt(image_name: str, width: int, height: int) -> str:
    return f"""Stage 1: OCR/Layout extraction for document forgery analysis.

Image: {image_name}
Native image size: width={width}, height={height}.

Return ONLY valid JSON. Do not wrap in Markdown.

Required JSON schema:
{{
  "document_language": "zh|en|id|ms|th|ar|unknown",
  "document_type": "short description",
  "global_summary": "brief summary in the document's main language",
  "text_spans": [
    {{
      "id": "s1",
      "text": "visible text exactly as read",
      "bbox": [x1, y1, x2, y2],
      "role": "title|header|body|table|number|date|signature|stamp|logo|other",
      "confidence": 0.0
    }}
  ],
  "layout_notes": ["short notes about tables, regions, stamps, portraits, logos"]
}}

Rules:
- Use native image pixel coordinates, integers, ordered as x1<x2 and y1<y2.
- Extract the most important text spans and suspicious-looking regions; do not invent text.
- Keep field names in English, but values such as global_summary should follow the document language.
- If text is unreadable, include the region with text="" and lower confidence.
"""


def evidence_prompt(ocr_layout: dict[str, Any], image_name: str, width: int, height: int) -> str:
    return f"""Stage 2: Evidence candidate extraction.

Image: {image_name}
Native image size: width={width}, height={height}.
Stage 1 OCR/Layout JSON:
{_json_dumps(ocr_layout)}

Return ONLY valid JSON. Do not wrap in Markdown.

Required JSON schema:
{{
  "visual_candidates": [
    {{
      "id": "v1",
      "category": "font_mismatch|edge_artifact|color_mismatch|copy_paste_boundary|layout_inconsistency|rendering_artifact|other",
      "span_ids": ["s1"],
      "bbox": [x1, y1, x2, y2],
      "evidence": "description in the document's main language",
      "confidence": 0.0
    }}
  ],
  "logical_candidates": [
    {{
      "id": "l1",
      "category": "math_error|date_impossible|identity_conflict|semantic_contradiction|sequence_error|context_anachronism|other",
      "span_ids": ["s1", "s2"],
      "bbox": [x1, y1, x2, y2],
      "evidence": "description in the document's main language",
      "confidence": 0.0
    }}
  ],
  "authenticity_prior": "authentic|suspicious|forged",
  "notes": ["brief notes in the document's main language"]
}}

Rules:
- Use only evidence visible in the image or derived from Stage 1 OCR/Layout.
- If there is no concrete candidate, return empty arrays and authenticity_prior="authentic".
- Every candidate must reference span_ids when text is involved.
- Use native image pixel coordinates. Coordinates should cover the visible evidence region.
"""


def validation_prompt(
    ocr_layout: dict[str, Any],
    evidence: dict[str, Any],
    image_name: str,
    width: int,
    height: int,
) -> str:
    return f"""Stage 3: Cross-cue validation and grounding.

Image: {image_name}
Native image size: width={width}, height={height}.
Stage 1 OCR/Layout JSON:
{_json_dumps(ocr_layout)}

Stage 2 Evidence JSON:
{_json_dumps(evidence)}

Return ONLY valid JSON. Do not wrap in Markdown.

Required JSON schema:
{{
  "verdict": "FORGED|AUTHENTIC",
  "risk_score": 0,
  "validated_anomalies": [
    {{
      "id": "a1",
      "category": "Visual Clumsy|Logical Fraud|semantic_subtle|text tampering",
      "source_candidate_ids": ["v1", "l1"],
      "span_ids": ["s1"],
      "bbox": [x1, y1, x2, y2],
      "visual_support": "visual evidence in the document's main language",
      "logical_support": "logical evidence in the document's main language",
      "reason": "final reason in the document's main language",
      "confidence": 0.0
    }}
  ],
  "discarded_candidates": [
    {{"id": "v1", "reason": "why discarded in the document's main language"}}
  ],
  "grounding_policy": "short statement"
}}

Rules:
- Validate candidates conservatively. Discard weak, isolated, or generic concerns.
- Grounding must come from OCR span boxes, candidate boxes, or their union. Do not invent unrelated coordinates.
- If verdict is AUTHENTIC, validated_anomalies must be an empty array and risk_score must be 0-10.
- If verdict is FORGED, every anomaly needs a bbox and reason.
- Use native image pixel coordinates, integers, ordered as x1<x2 and y1<y2.
"""


def report_prompt(
    ocr_layout: dict[str, Any],
    validated: dict[str, Any],
    image_name: str,
    width: int,
    height: int,
) -> str:
    return f"""Stage 4: Final report synthesis.

Image: {image_name}
Native image size: width={width}, height={height}.
Stage 1 OCR/Layout JSON:
{_json_dumps(ocr_layout)}

Stage 3 Validated Anomalies JSON:
{_json_dumps(validated)}

Write the final report only. Do not wrap in JSON or code fences.

Required Markdown skeleton:

# FORGERY ANALYSIS  REPORT

**Report ID:** FAR-xxxx-xx-xx
**Date of Examination:** xxxx-xx-xx
**Case Type:** Document Authentication & Fraud Analysis

**Overall Assessment:**
    **[Conclusion]:** FORGED or AUTHENTIC
    **[RISK_SCORE]:** integer between 0 and 100

---

## DETAILED ANOMALY ANALYSIS

If FORGED, write one block per validated anomaly:

### ANOMALY_001: <category> (<short location description>)
[GROUNDING]:[x1, y1, x2, y2]
[REASON]: <detailed reason in the document's main language>

If AUTHENTIC, write exactly:
No anomalies detected. The document has been thoroughly examined and no signs of tampering, alteration, or forgery were found.

---

## SUMMARY
The examination of the document has identified <N> anomalies, resulting in a fraud risk score of <score>. <Then describe the document content and evidence in the document's main language.>

---
**END OF REPORT**

Rules:
- Use the verdict, risk_score, anomaly categories, reasons, and bboxes from Stage 3 only.
- Do not add new anomalies or coordinates.
- Body text, REASON, and SUMMARY must follow the document's main language detected in Stage 1: {ocr_layout.get("document_language", "unknown")}.
- For Chinese, Thai, and Arabic documents, do not default to English prose.
- For Indonesian and Malay documents, use natural Indonesian/Malay-style Latin prose rather than an English template.
- If AUTHENTIC, include no [GROUNDING] lines.
"""
