# FakeShield + OCR Follow-up Plan

## Baseline Lock

Current kept Qwen-pipe baseline:

| Run | S_Fin_proxy | S_Det | S_Loc | S_Exp | S_Rep_proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen_pipe_v77_evidence_multibox_300` | 0.6438 | 0.8700 | 0.1144 | 0.3756 | 0.8674 |

Residual full-300 issues:

- `authentic_predicted_with_boxes=37`
- `forged_without_boxes=2`
- v71 changed boxes on 24 improved forged samples and 22 worsened forged samples. The main regression pattern was over-shrinking a broad table/body anomaly to a single OCR span.
- v72 confirmed the opposite failure mode: ungated OCR row/table cluster expansion improves low-`S_Loc` diagnostic rows but regresses full-300 by over-expanding already-good local boxes.
- v74 keeps only strict OCR cluster replacements: 13 rows changed, 13 boxes replaced, `S_Loc` improves from v71 `0.1045` to `0.1073` with unchanged `S_Det`.
- v75b adds hard discarded-candidate rescue and keeps three forged samples that validation previously downgraded to authentic. Broad rescue introduced FP, so benign logo/placeholder/dash cases are explicitly excluded.
- v77 appends high-confidence evidence-level boxes to forged reports. It increases average predicted boxes on correctly detected forged rows from 1.99 to 2.55, reduces `loc=0` rows from 31 to 26, and raises `S_Loc` to `0.1144`.

## FakeShield Findings

FakeShield is not a light dependency for this repo: full use needs the DTE-FDM model, MFLM model, DTG weights, SAM ViT-H weights, CUDA, and older transformer/MMCV stacks. Direct integration is therefore not a good next step.

Useful ideas to borrow:

1. **Domain tag before detection**
   - FakeShield uses a ResNet-50 DomainTagGenerator with three tags: AIGC inpainting, DeepFake, Photoshop.
   - The predicted tag is prepended to the DTE-FDM prompt, so the detector reads the image with a manipulation-specific prior.
   - Qwen-pipe analogue: replace generic anomaly routing with document-forensics tags such as `table_numeric`, `text_render`, `layout_table`, `asset_style`, `redaction_block`, and use these tags to select OCR candidate generation and reviewer rules.

2. **Explanation first, localization second**
   - FakeShield runs DTE-FDM to produce a textual tamper description, then MFLM turns the description into a mask.
   - Qwen-pipe already approximates this in v71 by using the final report reason to choose OCR/textline candidates.
   - Missing piece: v71 still treats many explanations as "pick the mentioned token". For table/logical/body-smearing cases, the explanation describes a relationship over multiple OCR spans, so localization should select a row/column/paragraph cluster.

3. **Mask-oriented grounding**
   - FakeShield MFLM generates `[SEG]` tokens, projects text embeddings into a SAM-style prompt encoder, and optimizes BCE + Dice mask loss.
   - Qwen-pipe cannot train this, but can approximate the mask prior by converting OCR line/cell clusters and local visual components into multiple compact grounding boxes rather than one rectangle.

4. **Region-aware inputs**
   - MFLM supports region/bbox-conditioned reasoning via `<bbox>` tokens and region features.
   - Qwen-pipe analogue: when verifying a crop, provide the crop plus OCR text for the target span, neighbor lines, row/column context, and candidate type. This is safer than asking the model to relocalize the whole page.

## OCR Priors Worth Adding

1. **Textline and paragraph graph**
   - Build OCR spans into rows by y-overlap, paragraphs by vertical gap, and table cells by row/column alignment.
   - For `render_pixel_blur` with words like smearing, ghosting, body text, paragraph, lower section, choose the row/paragraph cluster instead of a single keyword span.

2. **Table-aware numeric grounding**
   - For `logical_numeric`, extract all numbers/dates/amounts from the reason.
   - Match cited values to OCR spans, then expand to their row, header, total cell, or operand cells depending on terms like total, sum, missing item, duplicated value.
   - This should fix cases where single-value localization hurts because GT covers the inconsistent table region.

3. **Span style anomaly scores**
   - For each OCR span, compute local image features: darkness, HSV color, background highlight, edge density, Laplacian sharpness, connected-component density.
   - Compare each span to same-line and neighboring-line spans. Use z-scores to propose candidates for `style_color` and `render_pixel_blur`.
   - Gate this by anomaly reason so it does not become another broad false-positive detector.

4. **Domain-tagged FP/FN reviewer**
   - For the 37 authentic false positives, many reports describe generic style/asset inconsistencies. A reviewer should ask: is the anomaly a normal worksheet/design asset or a forensic inconsistency?
   - For the 2 remaining false negatives, use OCR digest + image crop review under document tags, not a generic authenticity prompt.

## Next Experiments

### v72/v74: OCR Cluster Grounding

Scope: no API, full local postprocess on top of v71.

Change:
- Add OCR row/paragraph/table-cell graph.
- For `logical_numeric` and broad `render_pixel_blur`, replace single-span candidates with row/cluster candidates.
- Do not touch already-good small boxes unless cluster confidence is high.

Result:
- `qwen_pipe_v72_ocr_cluster_300` rejected: full-300 `S_Loc=0.0893`, `S_Fin_proxy=0.6333`.
- `qwen_pipe_v74_ocr_cluster_gated_300` kept: strict candidate-area and expansion gates produce full-300 `S_Loc=0.1073`, `S_Fin_proxy=0.6370`.
- Key lesson: OCR clusters are useful only as a verifier for degenerate/thin or clearly under-scoped boxes; broad row/table expansion must be tightly gated.

### v75b: Hard Discarded-Candidate Rescue

Scope: no API, full local postprocess on top of v74.

Change:
- Recover AUTHENTIC predictions when validation discarded multiple hard localized visual candidates.
- Hard candidates include localized cover/obscure/unreadable/floating/gray-block/abnormal-spacing evidence across English, Chinese, Thai, and Arabic.
- Exclude benign logo, placeholder, dashes, missing-data, and normal typography cases to avoid adding FP.

Result:
- `qwen_pipe_v75_hard_discard_rescue_300` was too broad: it rescued 5 rows but added 2 FP.
- `qwen_pipe_v75b_hard_discard_rescue_300` is kept: `S_Fin_proxy=0.6401`, `S_Det=0.8700`, `forged_without_boxes=2`, `S_Loc=0.1073`.

### v76: World-only Downgrade

Scope: no API, issue-level downgrade on top of v75b idea.

Result:
- Rejected/refine. It reduces FP from 37 to 36 but increases FN from 2 to 3, with `S_Fin_proxy=0.6396`, below v75b.
- Lesson: weak typo/template/world-knowledge downgrades need a better domain tag or API verifier; deterministic broad rules are not safe enough.

### v77: Evidence-level Multi-box Grounding

Scope: no API, full local postprocess on top of v75b.

Change:
- Append extra local [GROUNDING] boxes from high-confidence evidence candidates to forged reports.
- Keep final verdict and existing anomalies unchanged.
- Gate by local visual categories, hard visual terms, benign exclude terms, max area, total-box budget, and duplicate IoU.

Result:
- Kept/current best: `qwen_pipe_v77_evidence_multibox_300` reaches `S_Fin_proxy=0.6438`, `S_Det=0.8700`, `S_Loc=0.1144`.
- Net localization: 16 forged samples improve, 9 regress, with positive total delta.

### v78 candidate: Span Style Prior

Scope: local visual scoring only.

Change:
- Compute per-OCR-span style/sharpness/color features and neighbor z-scores.
- Use these only for `style_color` and `render_pixel_blur` rows whose current box is broad or zero-overlap.

Decision test:
- Ablate by type: style-only, render-only, combined.
- Reject if it repeats v59/v69 behavior by shrinking correct broad boxes to glyph noise.

### v79 candidate: Domain-tagged Issue Reviewer

Scope: small API batch, no GT in prompt.

Change:
- Give Qwen the image/crop, OCR digest, current report, and document-forensics domain tag.
- Ask for a strict decision: keep forged, downgrade to authentic, or request localization refinement.

Decision test:
- Run only on current 37 FP + 2 FN issue rows.
- Keep only if `S_Det` improves with no meaningful `S_Loc` regression on forged rows.

## Stop Rules

- Do not directly integrate FakeShield weights unless the user explicitly wants a CUDA/weight-heavy branch.
- Do not add broad color/patch replacement without an OCR reason gate; v59 already showed that broad patch replacement hurts.
- Do not use GT masks or reports in prompts. GT is only for post-run diagnosis and evaluation.
