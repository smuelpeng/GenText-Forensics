# Algorithm Reproduction Experience

This note captures the practical lessons that mattered for reproducing and
improving Qwen-pipe. It is meant for future cluster validation and for avoiding
the failure modes that made many local experiments look promising but fail on
full-300 evaluation.

## Reproduction Contract

Every material experiment should preserve this contract:

1. **Prediction is GT-blind**:
   prompts, model-visible inputs, routing inputs, and candidate generation must
   not include GT labels, GT reports, GT masks, evaluator fields, or per-sample
   scores.

2. **GT is only post-hoc supervision**:
   GT can be used for offline diagnostics, fold-local candidate-delta training,
   ablations, and final evaluation after the prediction file already exists.

3. **Keep the split fixed**:
   compare 300-sample runs to 300-sample runs. Smoke and 30-80 sample runs are
   decision aids only, not final evidence.

4. **Record every material run**:
   keep command, input run, output run, metrics, decision, and mechanism lesson
   in `outputs/tuning/experiments.jsonl`.

5. **Do not mix Qwen-pipe with DocShield code paths**:
   Qwen-pipe lives under `qwen_pipe/`. Existing `baselines/DocShield/` files are
   dependencies or comparison points, not the place to put Qwen-pipe tuning code.

## What Actually Improved the Algorithm

### 1. Coordinate-space correction mattered more than prompt polishing

Many early localization failures came from box-space mismatch rather than weak
visual reasoning. The useful sequence was:

- count-matched OCR/span box replacement;
- explicit normalized 0-1000 projection when Qwen Stage-4 boxes behaved like
  normalized coordinates;
- conservative box replacement only when anomaly count and stage consistency
  made the mapping trustworthy.

Reproduction warning: run evaluation from the `debug_distribution` working
directory or use scripts that do so. The evaluator resolves masks relative to
its cwd; running it from another cwd can silently fall back to box mIoU and
produce wrong `S_Loc`.

### 2. Candidate coverage was not the main bottleneck

Exhaustive diagnostics showed the candidate pool often covered GT regions, but
global learned rankers picked bad actions. This changed the optimization target:

- stop adding broad candidate generators without selection gates;
- inspect which candidate families have positive deltas;
- build typed routers and high-precision selectors.

For v529, the final gain came from hard-positive KNN selection, not from adding
new OCR candidates.

### 3. Replacement and append are different algorithms

Replacement is useful when an existing box is broad, misplaced, or in the wrong
coordinate space. Append is useful when the report detects the forged image but
under-reports dispersed edited regions.

The final kept path separates them:

- v528: replace-only KNN rescue with OCR header/footer guards.
- v529: append-only KNN rescue on top of v528.

Do not merge replacement and append into one global action model unless there is
a per-action risk model. Broad append improves recall locally but often hurts
mask F1 through over-expansion.

### 4. OCR priors are useful only with noise guards

OCR rows, spans, token subspans, and linegrid windows are strong localization
priors. They also generate systematic false boxes:

- page numbers and page footers;
- bibliography or section-number headers;
- title/header text near the top of the page;
- tiny numeric fragments that match report numbers but not tampered regions;
- long plain prose rows with weak query evidence.

Kept guards include:

- page-footer regex for `Page N`, `Page N of M`, and linegrid footer variants;
- short numeric top-header regex for linegrid/row candidates;
- long plain low-query OCR text rejection;
- value-span preservation for dates, amounts, IDs, percentages, URLs, and
  currency-like text.

### 5. Typed localization beats one global threshold

Forgery types need different localization behavior:

- `redaction/block`: local dark connected-component or patch detection helps.
- `render_pixel_blur`: OCR span/row windows should be verified as visual render
  artifacts, not logical mismatches.
- `style_color/highlight`: compare target line against neighboring line style.
- `layout_table/spacing`: row/table clusters are useful but need area gates.
- `logical_numeric/date/amount`: localize mentioned values or cells, not the
  whole paragraph.

The main lesson is to route first, then rank candidates inside each route.

### 6. Multi-model and API calls need cache stability

Bailian/VLM calls had measurable variance in accepted boxes. Keep stable caches
for verifier calls when comparing algorithm changes. Otherwise the experiment
mixes verifier drift with the intended mechanism change.

Useful API findings:

- `text-embedding-v4` helped rank OCR windows before crop verification.
- `qwen3-rerank` did not outperform embedding similarity here.
- `qwen3-vl-plus` and thinking mode were stricter, but stricter rejection did
  not recover dispersed GT regions.

Cluster validation should replay cached outputs when validating final stages,
then separately run API-refresh experiments with a new run name.

### 7. False-positive rejection must be guarded

Large `S_Det` gains came from guarded false-positive rejection, but broad
downgrades traded false positives for false negatives. Good rejectors should:

- be cross-validated or fold-local;
- preserve high-risk visual evidence;
- avoid downgrading single hard visual cues solely because text is weak;
- report exact changed sample IDs and confusion changes.

### 8. Report text was stable; localization was the bottleneck

Most gains came from grounding and detection routing. Do not spend long
iterations polishing report wording unless `S_Exp` or `S_Rep_proxy` becomes the
explicit target metric. The current weak metric is still `S_Loc`.

## Cluster Reproduction Checklist

Before running on a cluster:

1. Check out the best branch.
2. Create a Python environment with `requirements.txt` or reuse the same venv.
3. Place `data/val_300.jsonl` and image/mask assets under `debug_distribution`.
4. Place required Qwen-pipe cached artifacts under a shared artifact directory
   or under `qwen_pipe/outputs/`.
5. Run prediction stages without GT inputs.
6. Run evaluation from `debug_distribution` cwd.
7. Verify metrics against v529:
   `S_Fin_proxy=0.705163`, `S_Det=0.963333`, `S_Loc=0.247136`,
   `S_Exp=0.364459`, `S_Rep_proxy=0.891620`.

If metrics differ, first check cwd and mask resolution, then check artifact
versions, then check whether API caches were refreshed.

## Minimum Artifacts for Final-Stage Reproduction

The final v528/v529 rescue stage needs:

- `qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.jsonl`
- `qwen_pipe_v507_fp_cv_guarded_arsoft_ar0313_ms0325_t0535_on_v502_300.json`
- `v518_v507_lowloc_t010_context_cl32.json`
- `qwen_pipe_v521_empty_selected_context_diag.jsonl`

These are local/generated artifacts and should not be committed to git. Store
them in cluster storage and pass their parent directory through
`QWEN_PIPE_ARTIFACT_ROOT` when using the cluster validation script.
