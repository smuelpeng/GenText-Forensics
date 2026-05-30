# Cluster Validation Runbook

Use this runbook to validate the promoted Qwen-pipe v529 result on a cluster.
The goal is to separate algorithm reproducibility from API/model variance.

## 1. Checkout

```bash
git clone git@github.com:smuelpeng/GenText-Forensics.git
cd GenText-Forensics
git checkout qwen-pipe-best-v529-20260530
```

The Qwen-pipe code is under:

```text
qwen_pipe/
```

It is intentionally separate from:

```text
baselines/DocShield/
```

## 2. Environment

Use the repo environment if available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or set `PYTHON` to an existing environment:

```bash
export PYTHON=/path/to/python
```

Required runtime libraries include `numpy`, `Pillow`, and the packages already
used by `scripts/eval_competition_aligned.py`.

## 3. Data Layout

The evaluator expects the validation JSON and masks/images in the normal
`debug_distribution` layout:

```text
data/val_300.jsonl
data/...
outputs/...
qwen_pipe/
```

Run the evaluator from the repository root or use the provided scripts. Do not
run evaluation from `qwen_pipe/` directly, because mask paths are cwd-sensitive.

## 4. Required Artifacts

The final-stage v529 reproduction needs these generated artifacts:

```text
raw/qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.jsonl
eval/qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.json
cache/pair_observations/v518_v507_lowloc_t010_context_cl32.json
diagnostics/qwen_pipe_v521_empty_selected_context_diag.jsonl
```

Recommended cluster storage layout:

```text
$QWEN_PIPE_ARTIFACT_ROOT/
├── raw/
├── eval/
├── cache/
│   └── pair_observations/
└── diagnostics/
```

These artifacts are not committed because they are generated outputs/caches.

## 5. Run Final-Stage Validation

From the repository root:

```bash
export QWEN_PIPE_ARTIFACT_ROOT=/path/to/qwen_pipe_artifacts
export QWEN_PIPE_OUTPUT_ROOT=/path/to/cluster_outputs/qwen_pipe_v529
bash qwen_pipe/scripts/cluster_validate_v529.sh
```

The script runs:

1. v528 replace-mode hard-positive KNN rescue;
2. v528 evaluation;
3. v529 append-mode hard-positive KNN rescue;
4. v529 evaluation.

Expected final metrics:

| Metric | Expected |
| --- | ---: |
| `S_Fin_proxy` | 0.705163 |
| `S_Det` | 0.963333 |
| `S_Loc` | 0.247136 |
| `S_Exp` | 0.364459 |
| `S_Rep_proxy` | 0.891620 |

The expected selected actions are:

- v528: replace two linegrid boxes,
  `GenText_Forensic_00009185` and `GenText_Forensic_00018153`.
- v529: append one OCR-row box,
  `GenText_Forensic_00011907`.

## 6. Output Files

By default, outputs are written to:

```text
$QWEN_PIPE_OUTPUT_ROOT/raw/
$QWEN_PIPE_OUTPUT_ROOT/eval/
$QWEN_PIPE_OUTPUT_ROOT/diagnostics/
```

The script prints the final eval JSON path and a compact score summary.

## 7. Troubleshooting

### `S_Loc` is much lower and `loc_methods` shows `box_miou`

The evaluator did not find GT masks. Run evaluation from the repo root or use
the provided scripts. Confirm `data/val_300.jsonl` and mask/image paths are
present.

### v529 selects zero append actions

Check that v528 was evaluated with mask-based `S_Loc` and that
`BASE_EVAL`/`V528_EVAL` include `samples`. The append risk gate is trained from
the per-sample eval JSON.

### Metrics differ slightly

First check artifact versions. The final rescue stages are deterministic given
the same v507 raw/eval and v518 observation cache.

### API-related variance appears

The final v528/v529 rescue stage does not call APIs. If the cluster run also
rebuilds earlier Qwen/VLM stages, isolate that as a new experiment name and keep
the final-stage validation separate.

## 8. Cluster Hygiene

- Do not commit `qwen_pipe/outputs/`.
- Do not write API keys into `.env` files that may be committed.
- Keep run outputs under a job-specific output root.
- Save stdout/stderr logs with the run ID, because the KNN scripts print the
  selected sample IDs and thresholds.
- If using SLURM, pin one validation job to one artifact version; do not let
  multiple jobs write to the same output paths.
