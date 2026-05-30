# Qwen-pipe Tuning Loop

This loop follows the `paper-guided-algorithm-tuning` skill: every iteration must connect a paper/detail prior, local failure evidence, a scoped change, a small-batch test, and a recorded decision.

## Component Matrix

| Paper component | Reused from `debug_distribution` | Qwen-pipe gap | Failure evidence | Next experiment |
| --- | --- | --- | --- | --- |
| OCR / layout preparation | `qwen-vl-ocr` cache and Stage 1 layout extraction | OCR is still an anchor, not a true textline module | Low `S_Loc`; `forged_without_boxes` remains | Keep box-space correction; test safer textline refinement only on low-confidence rows |
| Visual cues extraction | Stage 2 visual candidates and reviewer rules plus crop/patch redaction detector | Local detector only helps strong block/redaction patches | Full-300 false positives remain; broad color/render patching regresses | Mine FP patterns before adding broader detectors |
| Logical cues extraction | Stage 2 logical candidates and Stage 3 validation | Logical and visual signals are still coupled | Single-cue logical claims can overfire | Narrow reviewer/adjudicator rules |
| Cross-cue validation | Stage 3 plus deterministic reviewers, hard discarded-candidate rescue | FP remains high; broad downgrade rules still hurt true forged samples | v75b reduces `forged_without_boxes` from 5 to 2, but FP remains 37 | Add a stricter domain-tagged FP reviewer; avoid broad typo/template downgrades |
| Grounding | Stage 4, normalization, explicit 0-1000 box-space projection, conservative crop/patch redaction refinement, typed OCR/textline localization gate, strict OCR row/table cluster gate, evidence-level multi-box grounding, verifier-gated multi-grid OCR search | Boxes still miss dispersed multi-region edits and semantic-only tampering; full local row-window append over-expands masks | v78 API-gated search gives small positive movement, while full local append regresses | Improve candidate ordering and per-type crop-verifier prompts before increasing API volume |
| Report synthesis | Stage 5 Markdown report | Report proxy is stable enough | `S_Rep_proxy=0.8403` but low loc | Avoid prose-only tuning |

## Loop Commands

Use smoke first:

```bash
EXP_NAME=qwen_pipe_smoke_001 scripts/qwen_pipe_loop.sh smoke
```

Then run 60 samples for a decision:

```bash
EXP_NAME=qwen_pipe_v29_candidate MAX_SAMPLES=60 NUM_WORKERS=8 scripts/qwen_pipe_loop.sh all
```

Only promote to full 300 after the 60-sample run improves the target metric:

```bash
EXP_NAME=qwen_pipe_v29_full MAX_SAMPLES=0 NUM_WORKERS=8 DECISION=keep scripts/qwen_pipe_loop.sh all
```

Inspect representative errors:

```bash
EXP_NAME=qwen_pipe_v29_candidate MAX_SAMPLES=60 scripts/qwen_pipe_loop.sh errors forged_without_boxes
EXP_NAME=qwen_pipe_v29_candidate MAX_SAMPLES=60 scripts/qwen_pipe_loop.sh errors authentic_predicted_with_boxes
```

## Stop Conditions

- Stop after two consecutive regressions on the same metric without new diagnostic evidence.
- Stop if the next step requires unavailable training data, unavailable models, or an unconfigured paid API.
- Stop immediately if GT leakage appears in a prompt or stage input.

## Completed Qwen-pipe Experiments

Completed:

1. `qwen_pipe_v29_moe_rescue`: broad multi-model donor rescue. Rejected because semantic/date/copy-paste triggers increased false positives.
2. `qwen_pipe_v30_moe_rescue_artifact_r85`: kept. Artifact-only donor rescue improved full-300 `S_Fin_proxy` from 0.6012 to 0.6078.
3. `qwen_pipe_v34_boxscale_1p1_1p1`: kept. A small second-pass box expansion improved full-300 `S_Fin_proxy` to 0.6086 and `S_Loc` to 0.0345.
4. `qwen_pipe_v38_moe_e15v19_blocker_boxscale_1p1_1p1`: kept. Added v15/v19 donors plus a rescue-only template/path blocker; improved full-300 `S_Fin_proxy` to 0.6103.
5. `qwen_pipe_v40_moe_artifact_plus_r85`: rejected. Broad physical terms such as generic font/pixelation cues reduced FN but increased FP to 51.
6. `qwen_pipe_v42_moe_artifact_safe_plus_boxscale_1p1_1p1`: kept. Narrowed safe physical triggers, improved full-300 `S_Fin_proxy` to 0.6133.
7. `qwen_pipe_v48_moe_artifact_precision_plus_boxnorm_1p0_1p0`: kept. Adds audited precision triggers (`font weight`, `pixelation+jagged edges+anti-aliasing`) and uses grounding normalization without expansion; full-300 `S_Fin_proxy=0.6183`, `S_Det=0.8500`, `S_Loc=0.0365`.
8. `qwen_pipe_v50_moe_artifact_precision_plus_lowriskblue_boxnorm_1p0_1p0`: kept. Adds audited Chinese/Traditional glyph-space triggers and a narrow low-risk Malay blue-block whitelist; full-300 `S_Fin_proxy=0.6231`, `S_Det=0.8600`, `S_Loc=0.0375`.
9. `qwen_pipe_v52_ocr_box_refine_countmatch`: kept. Replaces final report grounding boxes with Stage-4 OCR/span-normalized boxes when counts match; full-300 `S_Fin_proxy=0.6283`, `S_Det=0.8600`, `S_Loc=0.0639`.
10. `qwen_pipe_v58_boxspace_force1000`: kept. Crop/GT-mask diagnosis showed Qwen Stage-4 boxes were often 0-1000 coordinates misread as pixels on images near 1000 px wide. Forcing normalized projection improved full-300 `S_Fin_proxy=0.6337`, `S_Loc=0.0912`.
11. `qwen_pipe_v59_crop_patch_refine`: rejected. Broad dark/red/yellow patch replacement over-corrected rendering and color cases; `S_Loc` dropped to 0.0637.
12. `qwen_pipe_v60_crop_patch_blockonly`: rejected as a standalone replacement policy. Restricting to dark block/redaction patches helped some rows, but still selected small glyph-like components and reduced `S_Loc` to 0.0865.
13. `qwen_pipe_v63_crop_patch_blockonly_w04`: kept. Adds a 4%-of-page minimum patch width to keep only true redaction/block-like components; full-300 `S_Fin_proxy=0.6342`, `S_Det=0.8600`, `S_Loc=0.0937`.
14. `qwen_pipe_v68_text_crop_api_30`: diagnostic. Implements the typed crop verifier interface and runs Qwen on 30 lowest-`S_Loc` samples. It improves that hardest subset from near-zero `S_Loc` to `0.0088`, but the gain is too weak to promote.
15. `qwen_pipe_v69_text_crop_local_300`: rejected. Ungated local OCR/textline replacements over-shrank boxes and reduced full-300 `S_Loc` to `0.0872`.
16. `qwen_pipe_v70_text_crop_local_strict_300`: rejected. Removing generic block routing helped classification but still replaced too many useful existing boxes; full-300 `S_Loc=0.0876`.
17. `qwen_pipe_v71_text_crop_local_gated_300`: kept. Type-routed local policy only accepts strong redaction patches and OCR text matches when the current box is broad. Full-300 `S_Fin_proxy=0.6364`, `S_Det=0.8600`, `S_Loc=0.1045`, `S_Exp=0.3758`, `S_Rep_proxy=0.8603`.
18. `qwen_pipe_v72_ocr_cluster_300`: rejected. Ungated FakeShield-inspired OCR row/table cluster expansion helped the low-`S_Loc` diagnostic subset but over-expanded correct local boxes on full 300; full-300 `S_Fin_proxy=0.6333`, `S_Loc=0.0893`.
19. `qwen_pipe_v73_ocr_cluster_gated_300`: kept as an intermediate gate. It protects non-degenerate boxes with candidate-area and expansion gates; full-300 `S_Fin_proxy=0.6369`, `S_Loc=0.1071`.
20. `qwen_pipe_v74_ocr_cluster_gated_300`: kept. Tightens the OCR cluster gate after a local grid sweep (`max_candidate_ratio=0.08`, `degenerate_height=20`, `max_nondegenerate_expand=15`); full-300 `S_Fin_proxy=0.6370`, `S_Det=0.8600`, `S_Loc=0.1073`, `S_Exp=0.3758`, `S_Rep_proxy=0.8603`.
21. `qwen_pipe_v75_hard_discard_rescue_300`: diagnostic. Rescuing all hard-looking discarded candidates improved FN but introduced 2 FP; full-300 `S_Fin_proxy=0.6373`, `S_Det=0.8633`.
22. `qwen_pipe_v75b_hard_discard_rescue_300`: kept. Adds DocShield/FakeShield-inspired rescue for multiple hard visual candidates discarded by validation, excluding benign logo/placeholder/dash cases. Full-300 `S_Fin_proxy=0.6401`, `S_Det=0.8700`, `S_Loc=0.1073`, `S_Exp=0.3757`, `S_Rep_proxy=0.8608`; `forged_without_boxes` drops from 5 to 2.
23. `qwen_pipe_v76_issue_rescue_world_downgrade_300`: rejected/refine. A narrow world/template-only downgrade reduces FP from 37 to 36 but increases FN from 2 to 3 versus v75b; full-300 `S_Fin_proxy=0.6396`.
24. `qwen_pipe_v77_evidence_multibox_300`: kept. Appends high-confidence local evidence boxes to forged reports, with `max_total_boxes=6`, `max_extra_boxes=4`, `duplicate_iou=0.10`. Full-300 `S_Fin_proxy=0.6438`, `S_Det=0.8700`, `S_Loc=0.1144`, `S_Exp=0.3756`, `S_Rep_proxy=0.8674`.
25. `qwen_pipe_v78_multiscale_local_300`: rejected/refine. Multi-grid OCR row-window append proves the mechanism can generate dense candidates, but full local append is too broad: 151 rows changed, 561 boxes added, `S_Loc` drops to `0.0675`.
26. `qwen_pipe_v78_multiscale_local_lowloc20_300`: diagnostic. Selecting 20 low-`S_Loc` rows offline improves `S_Loc` on 12 samples with no localization losses, confirming the search mechanism helps under-localized rows but needs a GT-blind trigger.
27. `qwen_pipe_v78_multiscale_api_evcov01_k8_300`: current best. Adds a GT-blind trigger: run multi-grid search only when existing report boxes do not cover Stage-2 evidence candidates (`evidence_coverage<=0.1`), then append only Qwen crop-verifier accepted boxes. Full-300 `S_Fin_proxy=0.6440`, `S_Det=0.8700`, `S_Loc=0.1145`, `S_Exp=0.3753`, `S_Rep_proxy=0.8681`; API calls estimate 12 and boxes added 5.
28. `qwen_pipe_v79_multiscale_api_evcov01_stagefirst_k8_300`: rejected/refine. Verifying uncovered Stage-2 evidence candidates before generic OCR row windows accepted more boxes, but `S_Fin_proxy=0.6439`, below v78; useful lesson is that candidate priority alone does not solve verifier target mismatch.
29. `qwen_pipe_v80_vlplus_api_k3_300`: rejected. Tested `qwen3-vl-plus` as a direct crop-verifier replacement with the same `evidence_coverage<=0.1` trigger and `k=3`. It accepted only one extra box and slightly underperformed v78: `S_Fin_proxy=0.6438`, `S_Loc=0.1144`. The model is stricter, but stricter rejection alone does not recover dispersed GT boxes.
30. `qwen_pipe_v81_vlplus_thinking_k3_300`: rejected/refine. Added `--enable-thinking` to the crop verifier and tested `qwen3-vl-plus` thinking mode on full 300. It accepted three boxes and nearly tied v78 (`S_Fin_proxy=0.6440`) but still lower on `S_Loc=0.1144` and much slower. Keep it as an offline hard-case reviewer, not the main localization path.
31. `qwen_pipe_v82_embedding_v4_rank_k8_300`: kept/current best. Uses the original OCR key (`/Users/penpen/Desktop/api-key_OCR.txt`) to call `text-embedding-v4` and semantically rank multi-scale OCR candidates before crop verification. Full-300 `S_Fin_proxy=0.6441`, `S_Det=0.8700`, `S_Loc=0.1150`, `S_Exp=0.3753`, `S_Rep_proxy=0.8681`. Main gain is `GenText_Forensic_00009514`, whose `loc_score` rises from `0.0209` to `0.1004`.
32. `qwen_pipe_v82b_embedding_v4_rank_k4_300`: diagnostic. Same embedding ranker with `k=4`; best tested `S_Loc=0.1150` but slightly lower `S_Fin_proxy=0.6441` than v82 because representation score shifts. Useful lower-cost setting.
33. `qwen_pipe_v83_rerank_k8_300`: rejected. `qwen3-rerank` API works with the OCR key, but candidate ordering is worse for localization: `S_Loc=0.1143`, below v78 and v82. For this OCR-window selection task, direct embedding similarity is safer than rerank.
34. `qwen_pipe_v84_embedding_v4_basekey_k8_300`: diagnostic. The base key `/Users/penpen/Desktop/api-key.txt` now supports `text-embedding-v4`, so this run used it for both verifier and embedding ranking with a fresh cache. Full-300 `S_Fin_proxy=0.6438`, `S_Det=0.8700`, `S_Loc=0.1147`, `S_Exp=0.3750`, `S_Rep_proxy=0.8674`. It is below v82 because fresh verifier calls accepted more low-yield boxes; `qwen3-rerank` still returns `Model.AccessDenied` on the base key.
35. `qwen_pipe_v85_embedding_v4_basekey_stablecache_k8_300`: kept/current recommended. Uses the base key for both `qwen3.6-35b-a3b` verifier calls and `text-embedding-v4` semantic ranking, but reuses the stable base-key verifier response cache under `outputs/cache/multiscale_ocr_search`. It exactly matches v82 without the OCR-key dependency: `S_Fin_proxy=0.6441`, `S_Det=0.8700`, `S_Loc=0.1150`, `S_Exp=0.3753`, `S_Rep_proxy=0.8681`; verifier API calls estimate is `0`.

### Bailian model probes

- The stable path is now base-key only for verifier/VLM calls and `text-embedding-v4` ranking. Reuse the existing verifier cache to avoid fresh crop-verifier variance. The base key still denies `qwen3-rerank`, so rerank remains an isolated experiment only.
- `text-embedding-v4` is useful as an OCR-window semantic ranker: it replaces brittle keyword row scoring with semantic similarity between report/evidence text and OCR row/window text, then passes only top-K windows to the crop verifier. This is now validated by v82.
- `qwen3-rerank` is callable but did not help this task; it selected worse OCR windows than embedding-v4 in v83.
- `qwen3-vl-plus` and `qwen3-vl-plus --enable-thinking` were callable as visual crop verifiers. Single-crop output was well-formed and grounded, but full-set results showed no improvement over v78 because the accepted boxes were often tiny logical/OCR fragments rather than GT-overlapping forged regions.

Next:

1. Add per-type crop-verifier prompts and negative examples; current verifier accepts some numeric-looking windows that do not improve GT overlap.
2. Improve embedding-ranked multi-grid candidate generation with window-level negative cues. v82 shows semantic retrieval helps; v83 shows rerank is not automatically better.
3. Build a stricter structured precision adjudicator for the remaining 37 `authentic_predicted_with_boxes` cases. Avoid broad weak-term downgrades; v76 showed they trade FP for FN.
4. Use `docs/fakeshield_ocr_followup.md` for the next OCR-prior experiments: domain-tagged routing, OCR row/paragraph/table clusters, span style scores, and issue-only reviewer batches inspired by FakeShield.
