"""GT-blind forgery taxonomy used by staged DocShield prompts.

The categories are distilled from aggregate validation diagnostics and generic
document-forensics knowledge. Do not add sample IDs, labels, masks, GT reports,
or split-specific answers here.
"""

from __future__ import annotations


TAXONOMY_BRIEF = """Known text-centric document forgery taxonomy (generic, not sample-specific):

Strong visual tampering cues:
- Redaction or occlusion: black/gray blocks, covered text, masked fields, hidden names, blocked numbers.
- Blur, smudge, erasure, or missing fragments: localized unreadable text, erased strokes, smeared fields.
- Copy-paste or splice boundary: pasted patches, visible seams, compression mismatch, abrupt background changes.
- Local font/rendering mismatch: inconsistent glyph shape, stroke width, color, edge aliasing, or baseline only when localized and not a whole-document scan artifact.

Strong logical/semantic tampering cues:
- Numeric contradictions: totals, subtotals, percentages, scores, table values, IDs, or amounts that cannot be reconciled.
- Date/timeline contradictions: impossible years, age/time conflicts, anachronistic logos, copyright or QR-code conflicts.
- Entity/identity conflicts: wrong names, organizations, logos, signatures, IDs, or inconsistent person/company references.
- Table/list/order corruption: duplicate rows, missing required row/column, broken sequence, hierarchy errors.
- Semantic substitution: a word, name, number, or field that changes document meaning while the surrounding layout remains plausible.

Common benign-production cues that should not alone prove forgery:
- OCR/scan/digitization artifacts, low-resolution reproduction, compression noise, watermark bleed-through.
- Minor typos, spelling, punctuation, grammar, whole-document font rendering, template oddities, line spacing, or alignment issues.
- Ordinary comparative dates, standard table formatting, and generic layout irregularities that do not affect critical fields.
"""


STAGE1_FOCUS = """Stage focus:
- Identify document type, language, reading order, tables, stamps/logos/signatures, and critical fields such as names, dates, amounts, totals, IDs, scores, row labels, and official marks.
- Preserve spatial boxes for critical fields and suspicious regions even if OCR text is uncertain.
- Note whether a defect is localized to a field or appears uniformly across the scan/template.
"""


STAGE2_FOCUS = """Stage focus:
- Generate candidates from both strong tampering cues and subtle semantic substitutions.
- For each candidate, separate concrete evidence from a benign alternative explanation.
- Prefer candidates that affect critical fields or are supported by multiple visual/logical cues.
- Do not over-escalate isolated OCR, scan, watermark, typography, spelling, or formatting issues unless they change a critical field or coincide with strong tampering evidence.
"""


STAGE3_FOCUS = """Stage focus:
- Retain candidates with strong tampering evidence, critical-field impact, or cross-cue support.
- Downgrade benign-production-only candidates when they are isolated, low-complexity, and do not alter document meaning.
- Treat redaction/occlusion, localized blur/erasure, copy-paste boundaries, impossible totals/dates, identity conflicts, and semantic substitutions as high-risk.
- A single typo or rendering defect is usually insufficient unless it corrupts a key field such as name, amount, ID, date, score, table label, logo, or signature.
"""


STAGE4_FOCUS = """Stage focus:
- Ground the exact critical field or visible tampered region. For redactions/blur/smudges, ground the affected patch. For logical errors, ground the text spans carrying the contradiction.
- If multiple fields jointly prove one contradiction, return one box per manipulated field when possible, or a tight union if they are adjacent.
- Avoid full-page boxes; avoid unrelated decorative regions.
"""


STAGE5_FOCUS = """Stage focus:
- Report only validated anomalies. Do not introduce new anomalies in synthesis.
- Explain whether each anomaly is visual, logical, or semantic, and why benign-production alternatives were insufficient.
- For authentic decisions, describe the document and explicitly state that observed OCR/scan/layout imperfections do not establish tampering.
"""
