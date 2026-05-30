# Qwen-pipe

## Promoted Branch Result

This branch promotes the isolated Qwen-pipe code path into
`debug_distribution/qwen_pipe/`. It keeps Qwen-pipe separate from
`baselines/DocShield/` and from existing debug-distribution runners.

Current best full-300 run:

| Run | S_Fin_proxy | S_Det | S_Loc | S_Exp | S_Rep_proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen_pipe_v529_knn_append_precision_lowloc_t010_on_v528_300` | 0.705163 | 0.963333 | 0.247136 | 0.364459 | 0.891620 |

The final promoted stage is reproducible from local cached Qwen-pipe artifacts:

```bash
cd debug_distribution/qwen_pipe
scripts/run_best_v529_rescue.sh
```

See `docs/v529_best_summary.md` for the kept mechanisms, rejected lessons, and
remaining failure surface.

For cluster validation, use `scripts/cluster_validate_v529.sh` with
`docs/cluster_validation.md`. For algorithm-reproduction lessons, see
`docs/algorithm_reproduction_experience.md`.

`qwen_pipe/` is an isolated local branch for Qwen-based document-forensics tuning. In this promoted branch it lives inside `debug_distribution/`; in the earlier local workspace it lived as a sibling `pipe/` directory. It references the surrounding `debug_distribution` checkout for the existing dataset, OCR cache conventions, and evaluator, but keeps Qwen-pipe scripts, notes, outputs, and ledgers under this folder.

This separation is intentional: do not put Qwen-pipe prompts, postprocessors, or experiment ledgers into `debug_distribution/baselines/DocShield/` unless a change is later promoted back explicitly.

## Layout

```text
qwen_pipe/
├── README.md
├── configs/
│   └── qwen_pipe.env.example
├── docs/
│   ├── algorithm_reproduction_experience.md
│   ├── cluster_validation.md
│   ├── tuning_loop.md
│   └── v529_best_summary.md
├── scripts/
│   ├── cluster_validate_v529.sh
│   ├── qwen_pipe_loop.sh
│   └── run_best_v529_rescue.sh
└── outputs/                  # ignored local run artifacts
```

## Quick Start

Seed the Qwen-pipe ledger with the current `debug_distribution` v28 baseline:

```bash
cd /Users/penpen/Documents/gentext-forensics/pipe
scripts/qwen_pipe_loop.sh seed-baseline
scripts/qwen_pipe_loop.sh compare
```

Run a 3-sample smoke test:

```bash
EXP_NAME=qwen_pipe_smoke_001 \
CHANGES="Smoke-test isolated Qwen-pipe loop wiring" \
NOTES="No algorithm change." \
scripts/qwen_pipe_loop.sh smoke
```

Run a 60-sample decision experiment:

```bash
EXP_NAME=qwen_pipe_v29_candidate \
MAX_SAMPLES=60 \
NUM_WORKERS=8 \
CHANGES="One scoped algorithm change" \
NOTES="Hypothesis and expected metric movement." \
scripts/qwen_pipe_loop.sh all
```

List representative failures for the current run:

```bash
EXP_NAME=qwen_pipe_v29_candidate MAX_SAMPLES=60 \
scripts/qwen_pipe_loop.sh errors forged_without_boxes
```

## Boundary Rules

- Qwen-pipe outputs go to `qwen_pipe/outputs/`, not `debug_distribution/outputs/`.
- Qwen-pipe experiment records go to `qwen_pipe/outputs/tuning/experiments.jsonl`.
- `debug_distribution` is called as a dependency; do not edit its baseline files from this branch.
- Never put GT labels, GT reports, masks, or evaluator-only fields into model-visible prompts.
- Compare runs within the same split size. A 60-sample score and a 300-sample score are not directly comparable.

## Current Baseline

The seed action records these existing `debug_distribution` evals as reference baselines in the Qwen-pipe ledger:

| Run | Split | S_Fin_proxy | S_Det | S_Loc | S_Exp | S_Rep_proxy | Top issues |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `staged_cct_v28_visual_artifact_reviewer` | 60 | 0.6532 | 0.9500 | 0.0434 | 0.3876 | 0.8610 | `forged_without_boxes=3` |
| `staged_cct_v28_visual_artifact_reviewer` | 300 | 0.6012 | 0.8167 | 0.0303 | 0.3736 | 0.8403 | `authentic_predicted_with_boxes=37`, `forged_without_boxes=18` |

Treat v28 full-300 as the current kept baseline until Qwen-pipe has a better full-300 result or a clearly targeted metric improvement.

## Current Best

Current best full-300 Qwen-pipe run:

| Run | Split | S_Fin_proxy | S_Det | S_Loc | S_Exp | S_Rep_proxy | Top issues |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen_pipe_v78_multiscale_api_evcov01_k8` | 300 | 0.6440 | 0.8700 | 0.1145 | 0.3753 | 0.8681 | `authentic_predicted_with_boxes=37`, `forged_without_boxes=2` |
| `qwen_pipe_v80_vlplus_api_k3` | 300 | 0.6438 | 0.8700 | 0.1144 | 0.3756 | 0.8675 | `qwen3-vl-plus` direct verifier replacement; rejected |
| `qwen_pipe_v81_vlplus_thinking_k3` | 300 | 0.6440 | 0.8700 | 0.1144 | 0.3754 | 0.8681 | `qwen3-vl-plus` thinking verifier; rejected/refine |
| `qwen_pipe_v82_embedding_v4_rank_k8` | 300 | 0.6441 | 0.8700 | 0.1150 | 0.3753 | 0.8681 | `text-embedding-v4` OCR-window ranker; current best |
| `qwen_pipe_v82b_embedding_v4_rank_k4` | 300 | 0.6441 | 0.8700 | 0.1150 | 0.3754 | 0.8679 | lower-cost embedding ranker diagnostic |
| `qwen_pipe_v83_rerank_k8` | 300 | 0.6440 | 0.8700 | 0.1143 | 0.3753 | 0.8682 | `qwen3-rerank`; rejected |
| `qwen_pipe_v84_embedding_v4_basekey_k8` | 300 | 0.6438 | 0.8700 | 0.1147 | 0.3750 | 0.8674 | base key for verifier + embedding; diagnostic |
| `qwen_pipe_v85_embedding_v4_basekey_stablecache_k8` | 300 | 0.6441 | 0.8700 | 0.1150 | 0.3753 | 0.8681 | base key + stable verifier cache; current recommended |

Reproduce the offline backbone through v77 without new API calls:

```bash
cd /Users/penpen/Documents/gentext-forensics/pipe
scripts/run_current_best_offline.sh
```

Apply the v78 verifier-gated search on top of that backbone:

```bash
../debug_distribution/.venv/bin/python scripts/qwen_multiscale_ocr_search.py \
  --input-jsonl outputs/raw/qwen_pipe_v77_evidence_multibox_300.jsonl \
  --output-jsonl outputs/raw/qwen_pipe_v78_multiscale_api_evcov01_k8_300.jsonl \
  --mode api \
  --api-ranking score \
  --max-existing-evidence-coverage 0.1 \
  --max-api-candidates-per-sample 8
```

Apply the current recommended embedding-ranked search. The base key supports
`text-embedding-v4`; keep both verifier/VLM and embedding ranking on
`/Users/penpen/Desktop/api-key.txt`, and reuse the stable verifier cache under
`outputs/cache/multiscale_ocr_search`.

```bash
../debug_distribution/.venv/bin/python scripts/qwen_multiscale_ocr_search.py \
  --input-jsonl outputs/raw/qwen_pipe_v77_evidence_multibox_300.jsonl \
  --output-jsonl outputs/raw/qwen_pipe_v85_embedding_v4_basekey_stablecache_k8_300.jsonl \
  --mode api \
  --model qwen3.6-35b-a3b \
  --api-key-file /Users/penpen/Desktop/api-key.txt \
  --api-ranking embedding \
  --semantic-api-key-file /Users/penpen/Desktop/api-key.txt \
  --embedding-model text-embedding-v4 \
  --cache-dir outputs/cache/multiscale_ocr_search \
  --max-existing-evidence-coverage 0.1 \
  --max-api-candidates-per-sample 8
```

Probe Bailian visual reasoning as the verifier:

```bash
../debug_distribution/.venv/bin/python scripts/qwen_multiscale_ocr_search.py \
  --input-jsonl outputs/raw/qwen_pipe_v77_evidence_multibox_300.jsonl \
  --output-jsonl outputs/raw/qwen_pipe_v81_vlplus_thinking_k3_300.jsonl \
  --mode api \
  --model qwen3-vl-plus \
  --enable-thinking \
  --cache-dir outputs/cache/multiscale_vlplus_thinking \
  --max-existing-evidence-coverage 0.1 \
  --max-api-candidates-per-sample 3
```

This applies GT-blind steps over existing `debug_distribution` outputs:

1. MOE artifact rescue: v28 primary plus ensemble/v15/v19 donors, `risk>=85`, `artifact_precision_plus` trigger profile, a rescue-only template/path blocker, and a narrow low-risk whitelist for Malay blue pixelated blocks.
2. Grounding normalization: second-pass report grounding rewrite with `scale_x=1.0`, `scale_y=1.0`.
3. OCR box refinement: replace final report grounding boxes with Stage-4 OCR/span-normalized boxes when anomaly counts match; verdicts and reasons are unchanged.
4. Box-space correction: force Stage-4 Qwen boxes through normalized 0-1000 projection for current FORGED rows to fix images whose dimensions make automatic coordinate detection ambiguous.
5. Crop/patch refinement: for block/redaction-style anomaly reasons only, replace a grounding with a high-confidence dark visual patch when the patch width is at least 4% of the page width; verdicts, reasons, and anomaly count are unchanged.
6. Typed text-crop localization: route anomaly reasons into redaction, render/blur, style/color, layout/table, or logical/numeric types; use OCR/textline candidates and a strict local gate to replace only broad text boxes or strong redaction patches. The Qwen crop verifier is implemented for diagnostics, but the promoted v71 run uses the GT-blind local policy because it improved full-300 `S_Loc` without new API calls.
7. OCR row/table cluster grounding: expand selected logical/render/layout anomalies to OCR row or table clusters only behind strict gates.
8. Hard discarded-candidate rescue: retain multiple hard visual candidates that validation had discarded, excluding benign logo/placeholder/dash cases.
9. Evidence-level multibox grounding: append high-confidence local evidence boxes to forged reports.
10. Multi-grid/multi-scale OCR search: trigger only when existing report boxes do not cover Stage-2 evidence candidates (`evidence_coverage<=0.1`), then run Qwen crop verifier over OCR row-window candidates and append only API-accepted local boxes. Full local append regressed and is not promoted.
