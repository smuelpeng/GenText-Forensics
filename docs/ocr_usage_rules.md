# OCR Model Usage Rules

These rules are project configuration for using Qwen-OCR in the staged
DocShield/CCT baseline.

## API Key

- Use `/Users/penpen/Desktop/api-key.txt` for all online calls, including
  Qwen-OCR, because the primary key now has OCR access.
- Current verified OCR-capable models with the primary key: `qwen-vl-ocr` and
  `qwen3.6-plus`.
- `qwen-vl-ocr-latest` and dated OCR model IDs may still return
  `Model.AccessDenied`; do not use them in default commands unless access is
  re-verified.
- For Qwen-OCR coordinates, use the DashScope native API with
  `ocr_options={"task": "advanced_recognition"}`. This returns official
  `ocr_result.words_info` entries with `text`, quadrilateral `location`, and
  `rotate_rect` fields.
- For difficult multilingual documents, use the DashScope native API with
  `ocr_options={"task": "multi_lan"}` as a supplemental transcript channel.
  This task returns plain recognized text, not reliable forensic judgement and
  not coordinate boxes.
- Do not use the prompt-based OpenAI-compatible path as the default for
  `qwen-vl-ocr` coordinates. Qwen-OCR does not support custom system messages,
  and prompt-forced JSON can miss the model's built-in OCR task.
- `qwen3.6-plus` can be used for an ablation when we want a general VLM to
  produce prompt-shaped OCR JSON, but it is not the default OCR-coordinate path.
- `OCR_TRANSCRIPT_API_KEY_FILE` exists only as an override for controlled
  ablations. Leave it unset in normal runs.

## Model Role

- Qwen-OCR is an auxiliary perception tool.
- Its outputs may provide visible text, reading order, table text, and OCR/text
  coordinates when available.
- It must not produce authenticity verdicts, anomaly validation, risk scores, or
  final report reasoning.
- OCR-only oddities are not forgery evidence by themselves; later stages must
  verify them against image-visible cues, layout/cross-cue context, or logical
  contradictions.
- By default, native Qwen-OCR layout boxes are used during grounding and
  deterministic box normalization as `auxiliary_ocr_spans`, but they do not enter
  Stage 1/2/3 judgement prompts. Use `OCR_LAYOUT_TO_STAGE1=1` only as an
  explicit ablation. Current 60-sample evidence shows full Stage-1 OCR injection
  increases false positives.
- The separate full OCR transcript path is disabled by default. Stage 2 evidence
  extraction and Stage 4 grounding do not receive raw transcript text unless an
  explicit ablation flag is enabled.
- Do not pass raw OCR text directly into evidence extraction unless running an
  explicit ablation, because OCR-only text noise can over-trigger forged
  decisions.
- Multilingual transcript cache may be passed to Stage 1 only to improve
  document-language detection, OCR recovery, and reading order for Arabic,
  Malay/Indonesian, Thai, Chinese, and mixed-language samples. Do not gate this
  behavior by GT `language` or `language_code`; enable it explicitly for an
  experiment or cache all selected samples uniformly.

## Cache And Cost

- OCR transcript results are deterministic enough for this fixed validation set,
  so cache them under `outputs/cache/ocr_transcripts/` and reuse them across
  experiments.
- Native multilingual transcript results should be cached separately under
  `outputs/cache/ocr_transcripts_multilan/<model>/` with
  `--api-mode dashscope-native --ocr-task multi_lan`.
- OCR layout/text-box results should be cached under
  `outputs/cache/ocr_layouts/<model>/` and compared by model before using them
  for grounding. The default coordinate cache command should include
  `--api-mode dashscope-native --ocr-task advanced_recognition`.
- The staged CCT runner reads this cache only; it should not call OCR online
  during main CCT inference. Use `REQUIRE_OCR_LAYOUT_CACHE=1` when running
  experiments that require every selected sample to have cached OCR boxes.
- Use the online OCR call path only for smoke tests, small ablations, and cache
  misses.
- For rebuilding all 300 OCR results, prefer a cheaper batch/offline Qwen-OCR
  job if available in the current Aliyun account. Import or convert those batch
  results into the same cache layout before running the staged pipeline.

## Crop OCR

- If a crop/zoom module needs text refinement for a suspected region, it may
  call Qwen-OCR on the crop.
- Crop OCR should refine text and bbox evidence only. The crop result must still
  be passed to the VLM validation/grounding stages for forensic judgement.
