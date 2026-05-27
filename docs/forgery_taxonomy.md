# Forgery Taxonomy And Prompt-Loop Notes

This note summarizes aggregate diagnostics from the local 300-sample validation set. It is for algorithm tuning only. Do not copy sample-level GT labels, masks, reports, or answers into model-visible prompts.

## Aggregate GT Shape

- Split size: 300 samples, 152 forged and 148 authentic.
- Languages: zh, en, th, ar, ms, id.
- Forged GT reports average about 6.4 grounded regions per image; the median is 5.
- Forged GT reasons average about 300 characters each, so the target explanation style is multi-evidence and region-by-region rather than a single global sentence.
- Dominant GT anomaly categories are `logical_fraud`, `semantic_subtle`, and `visual_clumsy`.

## Forgery Tactics To Anticipate

- Occlusion/redaction: black or gray blocks, covered text, hidden names, masked numbers. This is highly discriminative for forged samples.
- Blur, smudge, erasure, or missing fragments: localized unreadable fields, erased strokes, smeared text, fragmentary characters.
- Copy-paste/splice boundary: pasted patches, visible seams, background/compression mismatch, abrupt local texture changes.
- Font/rendering inconsistency: local glyph shape, stroke width, color, baseline, edge aliasing, or typography mismatch. This is common in both forged and authentic samples, so it must be tied to locality and critical-field impact.
- Numeric/math contradictions: totals, subtotals, percentages, scores, dates, IDs, amounts, or table values that cannot be reconciled.
- Date/timeline impossibility: impossible years, age/time conflicts, copyright/logo/QR anachronisms.
- Entity/identity conflict: wrong name, company, logo, ID, signature, person/entity relation, or organization reference.
- Table/list/order corruption: duplicate rows, broken row order, missing required row/column, hierarchy errors.
- Semantic substitution: word/name/number/field changes meaning while the surrounding layout remains plausible.

## Benign Patterns That Cause False Positives

These are frequent in authentic reports and current model false positives:

- OCR, scanning, digitization, encoding, or PDF conversion artifacts.
- Minor typo, spelling, punctuation, grammar, or proofreading issues.
- Whole-document low quality, compression, watermark bleed-through, font rendering, line spacing, or alignment issues.
- Template oddities and generic layout irregularities that do not alter a critical field.
- Standard comparative dates, ordinary financial-table structure, and benign row/column layout differences.

## Module-Level Prompt Strategy

- Stage 1 OCR/Layout should identify critical fields explicitly: names, dates, amounts, totals, IDs, scores, row labels, logos, signatures, and official marks.
- Stage 2 Evidence should generate candidates with a benign alternative review, not just a suspicion statement.
- Stage 3 Validation should retain candidates with strong tampering cues, critical-field impact, or cross-cue support; it should discard isolated benign-production issues.
- Stage 4 Grounding should map validated anomalies back to exact critical fields or visible tampered patches; broad full-page boxes are not useful.
- Stage 5 Report should only synthesize validated anomalies and should explain why benign alternatives were insufficient.

## Current Flow Shortcomings Versus DocShield

- OCR remains VLM-based unless an external OCR model is explicitly enabled; Qwen-OCR access is not currently available with the configured key.
- Visual cue extraction is prompt-only and lacks crop/zoom, edge/color/compression detectors, or textline segmentation.
- Logical cue extraction is not fully independent from visual evidence; it is still bundled in the evidence prompt.
- Cross-cue validation has a deterministic benign reviewer but no trained reward, no self-consistency loop, and no PR2-style reviewer refinement.
- Grounding is bbox-based with post-hoc expansion, not true mask/textline grounding.
- Report synthesis is Markdown-oriented for the local evaluator, not the paper's final structured JSON report.
