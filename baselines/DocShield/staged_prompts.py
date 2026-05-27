"""Prompts for the staged evidence-grounded CCT baseline.

This module defines a staged API-only pipeline. It does not use labels,
ground-truth reports, or masks in model-visible prompts; those are reserved for
local evaluation only.
"""

from __future__ import annotations

import json
from typing import Any

from forgery_taxonomy import (
    STAGE1_FOCUS,
    STAGE2_FOCUS,
    STAGE3_FOCUS,
    STAGE4_FOCUS,
    STAGE5_FOCUS,
    TAXONOMY_BRIEF,
)


SYSTEM_PROMPT = (
    "You are a document forensics examiner. Analyze only the provided image and "
    "the prior model-generated stage data. Never assume ground-truth labels. "
    "Use evidence standards for text-centric document safety: classify as FORGED "
    "when concrete visual artifacts, text rendering defects, layout corruption, "
    "or logical contradictions undermine document authenticity. Do not require "
    "proof of attacker intent."
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _taxonomy_block(stage_focus: str, enabled: bool) -> str:
    if not enabled:
        return ""
    return f"\n{TAXONOMY_BRIEF}\n\n{stage_focus}\n"


def _ocr_transcript_block(ocr_transcript: str | None) -> str:
    text = (ocr_transcript or "").strip()
    if not text:
        return ""
    return f"""
Auxiliary OCR transcript from a dedicated OCR model:
{text}

Use this transcript only as a text-reading aid. It may contain OCR errors,
missing line breaks, or merged table cells. Spatial boxes must still come from
the image itself, and OCR-only oddities are not evidence of forgery by
themselves.
"""


def ocr_transcript_prompt(image_name: str, width: int, height: int) -> str:
    return f"""OCR transcript extraction.

Image: {image_name}
Native image size: width={width}, height={height}.

Return the visible document text only. Preserve the document's original
language and reading order as much as possible. Do not add authenticity
analysis, explanations, Markdown headings, or guesses about whether the document
is forged.
"""


def ocr_layout_prompt(
    image_name: str,
    width: int,
    height: int,
    taxonomy_enabled: bool = False,
    ocr_transcript: str | None = None,
) -> str:
    critical_schema = ""
    critical_rule = ""
    if taxonomy_enabled:
        critical_schema = """,
  "critical_fields": [
    {
      "id": "c1",
      "span_ids": ["s1"],
      "field_type": "name|date|amount|total|id|score|row_label|logo|signature|other",
      "why_critical": "short reason in the document's main language"
    }
  ]"""
        critical_rule = (
            "\n- For tables/lists, preserve row labels, numeric columns, totals, dates, "
            "IDs, and names as separate spans when feasible."
        )
    return f"""Stage 1: OCR/Layout extraction for document forgery analysis.

Image: {image_name}
Native image size: width={width}, height={height}.
{_ocr_transcript_block(ocr_transcript)}
{_taxonomy_block(STAGE1_FOCUS, taxonomy_enabled)}

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
  ]{critical_schema},
  "layout_notes": ["short notes about tables, regions, stamps, portraits, logos"]
}}

Rules:
- Use native image pixel coordinates, integers, ordered as x1<x2 and y1<y2.
- Extract the most important text spans and suspicious-looking regions; do not invent text.
- Keep field names in English, but values such as global_summary should follow the document language.
- If text is unreadable, include the region with text="" and lower confidence.
{critical_rule}
"""


def evidence_prompt(
    ocr_layout: dict[str, Any],
    image_name: str,
    width: int,
    height: int,
    taxonomy_enabled: bool = False,
    ocr_transcript: str | None = None,
) -> str:
    candidate_extra_schema = ""
    candidate_rule = ""
    if taxonomy_enabled:
        candidate_extra_schema = """,
      "criticality": "critical_field|supporting_field|decorative_or_background",
      "benign_alternative": "possible benign explanation, or empty string\""""
        candidate_rule = (
            "\n- Do not create candidates for harmless whole-document scan quality or minor "
            "typos unless they affect critical fields or pair with strong tampering cues."
        )
    return f"""Stage 2: Evidence candidate extraction.

Image: {image_name}
Native image size: width={width}, height={height}.
Stage 1 OCR/Layout JSON:
{_json_dumps(ocr_layout)}
{_ocr_transcript_block(ocr_transcript)}
{_taxonomy_block(STAGE2_FOCUS, taxonomy_enabled)}

Return ONLY valid JSON. Do not wrap in Markdown.

Required JSON schema:
{{
  "visual_candidates": [
    {{
      "id": "v1",
      "category": "font_mismatch|edge_artifact|color_mismatch|copy_paste_boundary|layout_inconsistency|rendering_artifact|other",
      "span_ids": ["s1"],
      "bbox": [x1, y1, x2, y2],
      "evidence": "description in the document's main language"{candidate_extra_schema},
      "confidence": 0.0
    }}
  ],
  "logical_candidates": [
    {{
      "id": "l1",
      "category": "math_error|date_impossible|identity_conflict|semantic_contradiction|sequence_error|context_anachronism|other",
      "span_ids": ["s1", "s2"],
      "bbox": [x1, y1, x2, y2],
      "evidence": "description in the document's main language"{candidate_extra_schema},
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
{candidate_rule}
"""


def validation_prompt(
    ocr_layout: dict[str, Any],
    evidence: dict[str, Any],
    image_name: str,
    width: int,
    height: int,
    taxonomy_enabled: bool = False,
) -> str:
    benign_review_schema = ""
    if taxonomy_enabled:
        benign_review_schema = """
      "benign_alternative_review": "why benign alternatives are insufficient, in the document's main language","""
    validation_rules = (
        "- Validate candidates as document-authenticity evidence. Do not discard an anomaly merely because it could be caused by generation, OCR, proofreading, formatting, or template errors; those visible/logical defects can still indicate a forged or manipulated document in this benchmark.\n"
        "- Discard only candidates that are genuinely benign, unreadable, unsupported by the image/OCR, or too vague to locate."
    )
    if taxonomy_enabled:
        validation_rules = (
            "- Validate candidates as document-authenticity evidence, not as generic image defects.\n"
            "- Discard candidates that are genuinely benign, unreadable, unsupported by the image/OCR, too vague to locate, or merely production-quality issues without critical-field impact."
        )
    risk_rule = ""
    if taxonomy_enabled:
        risk_rule = (
            "\n- Risk score guidance: 90-100 for multiple strong cues or direct critical-field tampering; "
            "70-89 for one strong localized cue; 20-69 for weak suspicion; 0-10 for benign/unsupported."
        )
    return f"""Stage 3: Cross-cue validation and filtering.

Image: {image_name}
Native image size: width={width}, height={height}.
Stage 1 OCR/Layout JSON:
{_json_dumps(ocr_layout)}

Stage 2 Evidence JSON:
{_json_dumps(evidence)}
{_taxonomy_block(STAGE3_FOCUS, taxonomy_enabled)}

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
      "visual_support": "visual evidence in the document's main language",
      "logical_support": "logical evidence in the document's main language",{benign_review_schema}
      "reason": "final reason in the document's main language",
      "confidence": 0.0
    }}
  ],
  "discarded_candidates": [
    {{"id": "v1", "reason": "why discarded in the document's main language"}}
  ],
  "validation_policy": "short statement"
}}

Rules:
{validation_rules}
- Logical contradictions that change document meaning, impossible dates, broken numbering, malformed names, garbled critical text, or inconsistent totals should usually survive validation even when visual artifacts are subtle.
- Visual artifacts such as font/rendering mismatch, copy-paste boundaries, localized blur, color/edge inconsistency, or layout corruption should survive when they are localized.
- If verdict is AUTHENTIC, validated_anomalies must be an empty array and risk_score must be 0-10.
- If verdict is FORGED, every anomaly must keep source_candidate_ids or span_ids so the next grounding stage can map it to coordinates.
{risk_rule}
"""


def grounding_prompt(
    ocr_layout: dict[str, Any],
    evidence: dict[str, Any],
    validated: dict[str, Any],
    image_name: str,
    width: int,
    height: int,
    taxonomy_enabled: bool = False,
    ocr_transcript: str | None = None,
) -> str:
    patch_rule = ""
    if taxonomy_enabled:
        patch_rule = "\n- If an anomaly involves occlusion/redaction/blur/erasure, ground the visible patch itself rather than the whole OCR line when possible."
    return f"""Stage 4: Spatial grounding for validated anomalies.

Image: {image_name}
Native image size: width={width}, height={height}.
Stage 1 OCR/Layout JSON:
{_json_dumps(ocr_layout)}
{_ocr_transcript_block(ocr_transcript)}

Stage 2 Evidence JSON:
{_json_dumps(evidence)}

Stage 3 Validated Anomalies JSON:
{_json_dumps(validated)}
{_taxonomy_block(STAGE4_FOCUS, taxonomy_enabled)}

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
  "discarded_candidates": [],
  "grounding_policy": "short statement"
}}

Rules:
- Ground every Stage 3 anomaly by matching its span_ids/source_candidate_ids to Stage 1 OCR boxes and Stage 2 candidate boxes.
- Use the auxiliary OCR transcript only to identify the exact text span that should be grounded; do not introduce new anomalies, verdicts, or risk changes from OCR text alone.
- Grounding must come from OCR span boxes, candidate boxes, or their tight union. Do not invent unrelated coordinates.
- If an anomaly is logical but references text, ground the exact text span(s) that carry the contradiction.
- If an anomaly has source_candidate_ids but no span_ids, use the candidate bbox.
- If verdict is FORGED, every anomaly must have a valid bbox in native pixel coordinates.
- If verdict is AUTHENTIC, return no anomalies and no boxes.
{patch_rule}
"""


def report_prompt(
    ocr_layout: dict[str, Any],
    validated: dict[str, Any],
    image_name: str,
    width: int,
    height: int,
    taxonomy_enabled: bool = False,
) -> str:
    return f"""Stage 5: Final report synthesis.

Image: {image_name}
Native image size: width={width}, height={height}.
Stage 1 OCR/Layout JSON:
{_json_dumps(ocr_layout)}

Stage 3 Validated Anomalies JSON:
{_json_dumps(validated)}
{_taxonomy_block(STAGE5_FOCUS, taxonomy_enabled)}

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
