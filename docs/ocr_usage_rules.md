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
- By default, the full OCR transcript is injected only into Stage 1 OCR/Layout.
  Stage 2 evidence extraction and Stage 4 grounding do not receive raw OCR text
  unless an explicit ablation flag is enabled.
- Do not pass raw OCR text directly into evidence extraction unless running an
  explicit ablation, because OCR-only text noise can over-trigger forged
  decisions.

## Cache And Cost

- OCR transcript results are deterministic enough for this fixed validation set,
  so cache them under `outputs/cache/ocr_transcripts/` and reuse them across
  experiments.
- OCR layout/text-box results should be cached under
  `outputs/cache/ocr_layouts/<model>/` and compared by model before using them
  for grounding. The default coordinate cache command should include
  `--api-mode dashscope-native --ocr-task advanced_recognition`.
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
