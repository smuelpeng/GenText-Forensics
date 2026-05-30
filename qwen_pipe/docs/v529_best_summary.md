# Qwen-pipe v529 Best Summary

This branch promotes the Qwen-pipe code path as an isolated module under
`qwen_pipe/`. It does not modify `baselines/DocShield/` or the existing
`debug_distribution` runners.

## Best Run

Best full-300 local proxy result:

| Run | S_Fin_proxy | S_Det | S_Loc | S_Exp | S_Rep_proxy | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen_pipe_v529_knn_append_precision_lowloc_t010_on_v528_300` | 0.705163 | 0.963333 | 0.247136 | 0.364459 | 0.891620 | Best kept run |

Reference chain:

| Run | S_Fin_proxy | S_Det | S_Loc | S_Exp | S_Rep_proxy | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300` | 0.705130 | 0.963333 | 0.246980 | 0.364457 | 0.891615 | kept baseline |
| `qwen_pipe_v526_knn_context_pageguard_precision_lowloc_t010_on_v507_300` | 0.705142 | 0.963333 | 0.247033 | 0.364464 | 0.891615 | kept |
| `qwen_pipe_v528_knn_context_headernumguard_precision_lowloc_t010_on_v507_300` | 0.705146 | 0.963333 | 0.247059 | 0.364457 | 0.891615 | kept |
| `qwen_pipe_v529_knn_append_precision_lowloc_t010_on_v528_300` | 0.705163 | 0.963333 | 0.247136 | 0.364459 | 0.891620 | kept |

## Reproduce Final Rescue Stage

The branch excludes local `outputs/` artifacts. To reproduce v529, first provide
the cached Qwen-pipe artifacts listed below under `qwen_pipe/outputs/`:

- `outputs/raw/qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.jsonl`
- `outputs/eval/qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.json`
- `outputs/cache/pair_observations/v518_v507_lowloc_t010_context_cl32.json`
- `outputs/diagnostics/qwen_pipe_v521_empty_selected_context_diag.jsonl`

Then run:

```bash
cd debug_distribution/qwen_pipe
scripts/run_best_v529_rescue.sh
```

The script is GT-blind during prediction. GT is only used by
`eval_competition_aligned.py` after outputs are produced.

## Kept Mechanisms

1. **MOE donor rescue with strict trigger routing**:
   donor reports help only when routed through audited visual/artifact triggers
   and template/path blockers. Broad semantic donor fusion increased false
   positives and was rejected.

2. **Box normalization and OCR-count-matched grounding**:
   OCR/span-normalized boxes and forced normalized 0-1000 projection fixed many
   coordinate-space failures. This moved `S_Loc` much more than report prose
   tuning.

3. **Typed local grounding**:
   redaction/block, render/blur, style/color, layout/table, and logical/numeric
   cases need different localization policies. A generic crop/text replacement
   over-shrinks boxes; typed gates are necessary.

4. **Patch detection only for true block/redaction cases**:
   dark patch detection helps black-block edits, but broad patch replacement
   hurts text-render and color cases. The useful guard was a minimum width ratio
   for real block-like components.

5. **OCR row/table cluster grounding with strict gates**:
   FakeShield-style OCR row/table priors are useful, but only when non-degenerate
   box protection and area/expansion gates prevent over-expansion.

6. **Evidence-level multi-box append**:
   appending high-confidence local evidence boxes improves recall without
   changing verdict/report text. Full local append regressed; targeted append
   was kept.

7. **Multi-grid OCR search plus semantic ranking**:
   multi-scale OCR windows improve candidate coverage. `text-embedding-v4`
   ranking was safer than rerank for selecting OCR windows before crop
   verification.

8. **FP guarded rejector for detection**:
   a cross-validated false-positive guard produced the large late improvement in
   `S_Det`, reaching 0.963333 while preserving localization.

9. **Hard-positive KNN rescue**:
   exhaustive candidate diagnostics showed candidate coverage was high, but
   learned rankers selected harmful actions. Fold-local KNN over hard positives
   became useful after typed OCR noise guards.

10. **Numeric header/footer OCR guards**:
    OCR linegrid candidates often hallucinate page/header/bibliography numbers
    as anomaly boxes. v528 adds a narrow short-numeric-header guard while
    preserving mid-page IDs and currency/value spans.

11. **Append-mode KNN rescue**:
    remaining low-localization samples often have positive append candidates,
    not only replacement candidates. v529 adds `--action-mode append` and keeps
    only one high-precision OCR-row append.

## Rejected Lessons

- Broad physical/artifact trigger expansion increases false positives.
- Ungated local OCR/textline replacement often shrinks useful broad boxes into
  low-overlap token fragments.
- Full multi-grid local append has high candidate recall but too much mask
  over-expansion.
- Linear candidate-delta and groupwise rankers were unstable in the lowloc tail:
  sparse positives caused them to select harmful row/evidence replacements.
- `qwen3-rerank` did not outperform embedding similarity for OCR-window
  selection in this dataset.
- `qwen3-vl-plus` and thinking mode were stricter crop verifiers, but stricter
  rejection alone did not recover dispersed forged regions.

## Remaining Failure Surface

The current bottleneck is still localization, especially Thai and multi-region
document edits:

- lowloc true-positive samples below 0.10 remain across all languages, with Thai
  the largest group.
- candidate coverage is usually not the limiting factor; selection and action
  routing are.
- many remaining wins are append actions, but broad append policies hurt masks.

Recommended next experiments:

1. Train a separate append-risk model for lowloc samples using source/family
   quotas and page-position guards.
2. Split append candidates into OCR span, row, token, evidence, and linegrid
   routers rather than one global KNN threshold.
3. Add a Thai-specific OCR row-window selector; Thai is the lowest `S_Loc`
   language group in v529.
