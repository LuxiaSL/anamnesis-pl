# Liveness Audit — `pipeline/anamnesis/`

Produced: 2026-04-16. Captures the state of the codebase at a moment when the
project is paused and fresh eyes (human or agent) need to parse it without
replaying the full history. The goal is **reduce parsing tax for future work**,
not tidy code for its own sake.

## TL;DR

- ~17k lines of Python across 49 files. The core path is coherent; the edges
  have accumulated duplication and one-off investigations.
- **No file is unambiguously dead.** Some files are "standalone investigations
  whose outputs are committed" — archive candidates, but the call is
  maintainer-discretion (these are hypothesis probes, not load-bearing code).
- The real friction is concentrated in a few specific places:
  - Layer presets duplicated in 3 locations.
  - `PROCESSING_MODES` in `config.py` is a verbatim copy of `RUN4_MODES` in `modes/`.
  - `verification_runner.py` reimplements ~400 lines already present in `unified_runner/`.
  - README documents `feature_pipeline` CLI flags that don't exist; 5+ scripts are undocumented.
  - Two parallel "how you extract features" contracts (tier-function tuples vs `FeatureFamilyResult`).

## Canonical paths — "if you want to X, start here"

| Task | Entry point | Then read |
|---|---|---|
| Calibrate positional means for a model | `scripts/run_8b_calibration.py` | `config.py`, `extraction/model_loader.py` |
| Generate baseline 8B data (5 canonical modes) | `scripts/run_8b_experiment.py` | `extraction/generation_runner.py:run_experiment`, `config.PROCESSING_MODES` |
| Generate under any mode set / save raw tensors | `scripts/run_extraction.py` | `extraction/generation_runner.py`, `modes/` |
| Generate R2-equivalent (non-format-controlled process modes) | `scripts/run_8b_r2_experiment.py` | `modes/run3_original_modes.py` |
| Recompute features from saved raw tensors | `extraction/feature_pipeline.py` (has `main()`) | `extraction/state_extractor.py`, `extraction/feature_families/` |
| Train contrastive projection | `scripts/train_contrastive_projection.py` | `extraction/feature_families/contrastive_projection.py` |
| Run full analysis gauntlet | `scripts/run_unified_analysis.py` | `analysis/unified_runner/__init__.py:run_full_analysis` |
| Binary prompt-swap confound test | `scripts/run_binary_prompt_swap.py` | `analysis/geometric_trio/data_loader.py` |
| Sub-family decomposition | `scripts/run_subfamily_decomp.py` | — |
| Complementarity / resolution / inversion | `scripts/analyze_complementarity.py` | reads `outputs/analysis/*/results.json` |
| Cross-run functional transfer (R2↔R3) | `scripts/run_cross_run_transfer.py` | — |
| LLM judge scoring of generations | `scripts/run_judge_scoring.py` | — |

## Package-level liveness

Tag key:
- **CANONICAL** — on the documented critical path.
- **SUPPORTING** — imported by canonical code; not invoked directly.
- **ANALYSIS-ONLY** — standalone analyses/CLIs with committed outputs.
- **ARCHIVE?** — one-off benchmarks/validations whose purpose was served; keep
  or delete is maintainer's call (often hypothesis probes).
- **TOOL** — developer utility, not an experiment.

### `anamnesis/` top-level

| File | Tag | Notes |
|---|---|---|
| `__init__.py` | SUPPORTING | Empty (1 line). No re-exports — readers always need deep paths. |
| `config.py` | CANONICAL | Model / generation / extraction / feature-pipeline / calibration configs. **Duplicates `RUN4_MODES`** as `PROCESSING_MODES` (lines 277-311). |

### `anamnesis/extraction/`

All CANONICAL or SUPPORTING. No dead code suspected at file level.

| File | Tag | Notes |
|---|---|---|
| `model_loader.py` | CANONICAL | Pre-RoPE k_proj hooks. Dependency for anything GPU-touching. |
| `streaming_generate.py` | CANONICAL | Custom generation loop. |
| `generation_runner.py` | CANONICAL | Orchestrates generate → collect → extract → save. Imports `PROCESSING_MODES` from `config.py` (not from `modes/`). |
| `state_extractor.py` | CANONICAL | 1040 lines. Pure-numpy tier 1/2/2.5/3 extraction. Returns bespoke tuple, **not** `FeatureFamilyResult`. |
| `raw_saver.py` | CANONICAL | Tensor serialization for offline reprocessing. |
| `feature_pipeline.py` | CANONICAL | Has two paths: `compute_features_from_raw` (v1, baseline only) and `compute_features_v2` (v1 + families). v1 is effectively a flavor of v2 with no families enabled. Also contains a **third copy** of layer presets (`_MODEL_LAYER_PRESETS` lines 463-480). |
| `feature_families/__init__.py` | CANONICAL | Defines `FeatureFamilyResult` contract. |
| `feature_families/_helpers.py` | SUPPORTING | Shared cosine / entropy helpers. |
| `feature_families/operators.py` | SUPPORTING | Windowed stats, STFT, slope operators. |
| `feature_families/{residual_stream,attention_flow,gate_features,temporal_dynamics,contrastive_projection}.py` | CANONICAL | Each a pluggable family. |

### `anamnesis/analysis/`

| File | Tag | Notes |
|---|---|---|
| `__init__.py` | SUPPORTING | 1 line, empty. |
| ~~`t3_investigation.py`~~ | ANALYSIS-ONLY | **[DELETED in Phase 1]** Standalone main(). Outputs preserved at `outputs/analysis/t3_investigation/`. Was a *model-comparison probe* ("why is T3 alive at 8B but dead at 3B?") — if rerun on future models, start from fresh code informed by current feature pipeline rather than reviving this file. |

#### `analysis/unified_runner/`

All CANONICAL. Each file exposes a `run_<section>(data)` function called from `__init__.py:run_full_analysis`.

| File | Tag | Notes |
|---|---|---|
| `__init__.py` | CANONICAL | Section orchestrator (10 sections, hardcoded if/elif). Section 6b (`manifold_geometry`) silently shares section 6's skip guard — latent bug. |
| `data_loading.py` | CANONICAL | `AnalysisData` wraps `Run4Data` + text fields for semantic analysis. |
| `utils.py` | CANONICAL | `standardize`, `remove_constant`, `clean_for_json`, `get_available_tiers`. Small but load-bearing. |
| `integrity.py`, `classification.py`, `tier_ablation.py`, `geometry.py`, `clustering.py`, `contrastive.py`, `semantic.py`, `scorecard.py` | CANONICAL | One per section. |

#### `analysis/geometric_trio/`

| File | Tag | Notes |
|---|---|---|
| `__init__.py` | SUPPORTING | 1 line. |
| `data_loader.py` | CANONICAL | `Run4Data`, `SampleMeta`, `TIER_KEYS`, `TIER_GROUPS`, `BASELINE_TIERS`, `ENGINEERED_TIERS`, `load_run4`. **Shared dependency** — `unified_runner/` imports from it for tier constants. |
| ~~`intrinsic_dimension.py`~~ | ANALYSIS-ONLY | **[DELETED in Phase 1]** Standalone CLI; outputs preserved at `results/intrinsic_dimension_results.json`. Faithful equivalent: `unified_runner/geometry.py:run_intrinsic_dimension`. |
| ~~`ccgp.py`~~ | ANALYSIS-ONLY | **[DELETED in Phase 1]** Standalone CLI; outputs preserved at `results/ccgp_results.json`. Faithful equivalent: `unified_runner/geometry.py:run_ccgp` / `_ccgp_variant`. |
| ~~`delta_hyperbolicity.py`~~ | ANALYSIS-ONLY | **[DELETED in Phase 1]** Standalone CLI; outputs preserved under `results/`. Faithful equivalent: `unified_runner/geometry.py:run_topology` (`gromov_delta_euclidean` block). |
| ~~`verification_runner.py`~~ | ANALYSIS-ONLY + DUPLICATION | **[DELETED in Phase 1]** Verification run complete; output preserved at `results/verification_run.json`. Reimplemented ~400 lines of `unified_runner/utils.py` + `geometry.py` verbatim — duplication was the whole liability. |

### `anamnesis/modes/`

All live. Each has a current consumer.

| File | Tag | Notes |
|---|---|---|
| `__init__.py` | SUPPORTING | Re-exports all four mode sets. |
| `run4_modes.py` | CANONICAL | The 5 format-controlled modes. Docstring: *"config.py imports from here or duplicates them"* — it's the latter. |
| `extended_modes.py` | CANONICAL | 8 modes (run4 + associative, compressed, structured with format control). |
| `run3_original_modes.py` | CANONICAL | Verbatim Run-2-paper / Run-3-local process-mode prompts (non-format-controlled). Used by `run_8b_r2_experiment.py`. |
| `prompt_swap.py` | CANONICAL | `PROMPT_SWAP_PAIRS` for confound testing. |

### `anamnesis/scripts/`

README-documented, clearly CANONICAL:

| File | Tag | Notes |
|---|---|---|
| `run_8b_experiment.py` | CANONICAL | Produces `outputs/runs/run_8b_baseline/`. |
| `run_8b_calibration.py` | CANONICAL | Produces `outputs/calibration/llama31_8b/`. |
| `run_extraction.py` | CANONICAL | Parameterized version of the above. Produces `8b_fat_01`, `3b_fat_01`, `smoke_test`. Has own `MODEL_CONFIGS` — second copy of layer presets. |
| `run_unified_analysis.py` | CANONICAL | Main analysis CLI. `KNOWN_RUNS` / `KNOWN_ADDONS` hardcoded here — canonical metadata in a script. |
| `run_binary_prompt_swap.py` | CANONICAL | Prompt-swap confound analysis. |
| `run_subfamily_decomp.py` | CANONICAL | Sub-family decomposition. |
| `analyze_complementarity.py` | CANONICAL | Cross-tier complementarity (reads `results.json` outputs). |
| `train_contrastive_projection.py` | CANONICAL | Contrastive MLP training. |
| `run_judge_scoring.py` | CANONICAL | LLM-judge eval of generated text. |

Live but NOT in README (documentation drift, not liveness issue):

| File | Tag | Notes |
|---|---|---|
| `run_8b_r2_experiment.py` | CANONICAL | Produces `outputs/runs/run_8b_r2_equivalent/`. Required by `run_cross_run_transfer.py`. |
| `run_cross_run_transfer.py` | CANONICAL | Cross-run functional transfer R2↔R3 at 8B. |

Archive candidates — one-off A/B tests or profiling tools (all deleted in Phase 1):

| File | Tag | Notes |
|---|---|---|
| ~~`test_fast_postprocess.py`~~ | ARCHIVE? | **[DELETED in Phase 1]** A/B benchmark; served its purpose when the postprocess optimization landed. Not a regression test. |
| ~~`compare_streaming_vs_hf.py`~~ | ARCHIVE? | **[DELETED in Phase 1]** One-shot validation; `streaming_generate.py` confirmed to match HF `generate()`. |
| ~~`profile_generate_overhead.py`~~ | TOOL | **[DELETED in Phase 1]** One-shot profiler; a future optimization pass should build a profiler tailored to its target rather than reviving this one. |

## Duplication & drift (the real entanglement)

### 1. Layer presets duplicated three times

- `config.py:100-147` — `ExtractionConfig` defaults (8B only, single model).
- `scripts/run_extraction.py:54-87` — `MODEL_CONFIGS` (8B + 3B, full model identity).
- `extraction/feature_pipeline.py:463-480` — `_MODEL_LAYER_PRESETS` (8B + 3B, subset of above).

**Consequence:** changing a layer list or adding a new model means editing three files with different shapes. Fix: single pydantic `ModelPreset` registry in `config.py`, consumed by all three.

### 2. `PROCESSING_MODES` vs `RUN4_MODES`

- `config.py:277-311` — `PROCESSING_MODES` dict.
- `modes/run4_modes.py:14-60` — `RUN4_MODES` dict, verbatim identical.

`run4_modes.py`'s own module docstring admits the duplication. The split
consumers: `run_8b_experiment.py → generation_runner → config.PROCESSING_MODES`;
`run_extraction.py → modes.run4_modes.RUN4_MODES`.

Fix: `config.PROCESSING_MODES = modes.run4_modes.RUN4_MODES` (single-line
re-export) or route all consumers through `modes/`.

### 3. `verification_runner.py` reimplements `unified_runner/` utilities

In `analysis/geometric_trio/verification_runner.py`:

| Local copy | Canonical location |
|---|---|
| `_standardize` (line 82) | `unified_runner/utils.py:13` `standardize` |
| `_remove_constant` (line 89) | `unified_runner/utils.py:21` `remove_constant` |
| `_clean_for_json` (line 655) | `unified_runner/utils.py:28` `clean_for_json` |
| `_generate_topic_folds` (line 252) | `unified_runner/geometry.py:188` |
| `_run_ccgp_variant` (line 269) | `unified_runner/geometry.py:203` (`_ccgp_variant`) |
| `_compute_centroids` (line 446) | `unified_runner/geometry.py:376` |
| `_hierarchical_clustering` (line 474) | `unified_runner/geometry.py:400` |
| `_to_newick` (nested, line 495) | `unified_runner/geometry.py:415` |

Two options:
- **Delete `verification_runner.py`.** Its verification run is complete (output
  at `results/verification_run.json`). Archive the file outside the package.
- **Refactor to import shared helpers.** Keeps verifier runnable; removes drift
  risk. Shared helpers may need minor API adjustments.

### 4. Two feature-extraction contracts

- `state_extractor.extract_tier1/2/2.5/3(...)` returns bespoke tuples, composed
  manually in `extract_all_features`.
- `feature_families/*.extract_X(...)` returns `FeatureFamilyResult(features, feature_names, family_name)`.

`feature_pipeline.compute_features_v2` then concatenates both worlds with ~12
lines of boilerplate per family. A single contract (every extractor returns a
`FeatureFamilyResult`-like object) would let the concatenation loop replace all
the manual slice-tracking.

**Risk note:** `state_extractor.py` is 1040 lines of tested logic producing
paper results. Retrofit is high-value but high-blast-radius. Defer.

### 5. `v1` vs `v2` feature computation paths

`feature_pipeline.compute_features_from_raw` (v1) and `compute_features_v2`
coexist. v1 is reached only when `--v2` is not passed to the CLI. Practically,
all current production data is v2. v1 is effectively "v2 with
`include_baseline_tiers=True` and all families disabled" — collapsible.

### 6. `KNOWN_RUNS` / `KNOWN_ADDONS` in a script

`scripts/run_unified_analysis.py:24-42` hardcodes run locations and addon
directories. This is canonical project metadata, not CLI logic. Belongs in
`config.py` as a `RunRegistry`.

## README drift

- **Quick start §3** (lines 173-178) documents a `feature_pipeline` CLI with
  `--run` and `--families` flags. These **do not exist**. Actual flags:
  `--raw-dir`, `--output-dir`, `--v2`, `--no-trajectory`, `--no-attention-flow`,
  `--no-gate`, `--no-temporal-dynamics`, `--no-stft`, `--no-baseline`,
  `--contrastive-model`, `--pca-model`, `--model {8b|3b}`, `--workers`,
  `--metadata-dir`, `--no-tier3`.
- **Quick start is missing a step** between extraction and feature
  recomputation: `train_contrastive_projection.py` must have been run before
  `--contrastive-model` can be passed. Readers following the Quick Start won't
  know that contrastive projection requires separately-trained weights.
- Scripts listed in README (§Structure): 9. Scripts present: 14. Missing from
  README: `run_8b_r2_experiment`, `run_cross_run_transfer`,
  `test_fast_postprocess`, `compare_streaming_vs_hf`, `profile_generate_overhead`.
- `modes/` listed in README: 3 files. Actually 4 (missing `run3_original_modes.py`).
- `geometric_trio/` listed in README: 3 files. Actually 6 (missing `data_loader.py`,
  `verification_runner.py`, `__init__.py`). `data_loader.py` is load-bearing
  (shared dep for unified_runner) so its omission is material.
- `analysis/` §Structure omits `t3_investigation.py`.

## CLAUDE.md coverage gap

- `CLAUDE.md:81-90` "How to run" section covers `run_unified_analysis`,
  `run_binary_prompt_swap`, `analyze_complementarity`. It **omits
  `run_8b_r2_experiment` and `run_cross_run_transfer`** — a reader following
  CLAUDE.md would not know how to reproduce cross-run transfer results.
- `MEMORY.md:84-90` inherits the same gap; `run_cross_run_transfer` appears as
  a result referenced elsewhere but the prerequisite `run_8b_r2_experiment`
  (which produces the R2-equivalent data) is undocumented.

## Latent bugs + minor cleanup spotted during audit

- `unified_runner/__init__.py:250-258` — "Section 6b" (manifold geometry) only
  runs when section 6 is not skipped, and does not emit a "SKIPPED" log when
  section 6 is skipped. If a user passes `--skip 6`, manifold_geometry silently
  does not run.
- `unified_runner/semantic.py:25` — `StratifiedKFold` is imported but never
  used in the file (only `GroupKFold` is used for topic-heldout splits).
  One-line delete.

## Proposed cleanup order (deferred — for discussion)

Ranked by parseability payoff per unit of risk:

1. **Fix README drift.** Zero-risk documentation fix. Highest clarity gain per
   minute of work.
2. **Collapse layer preset duplication.** Single `ModelPreset` registry in
   `config.py` consumed by `run_extraction.py` and `feature_pipeline.py`.
3. **Collapse `PROCESSING_MODES` → `RUN4_MODES`.** One-line re-export.
4. **Decide fate of `verification_runner.py`.** Either delete (output
   preserved) or refactor to import. No middle ground.
5. **Move `KNOWN_RUNS` / `KNOWN_ADDONS` to `config.py` as `RunRegistry`.**
6. ~~**Archive the A/B benchmarks and profilers.**~~ **DONE (Phase 1)** — all three files deleted rather than archived: `test_fast_postprocess.py`, `compare_streaming_vs_hf.py`, `profile_generate_overhead.py`. Git history preserves them if needed. Future optimization work would want a fresh profiler tailored to its target, not this one.
7. **Add schemas for analysis result dicts.** Pydantic result types for each
   `run_<section>()` would make the 400+ `clf.get("tier", {}).get("rf_5way", ...)` accesses
   in `__init__.py:_print_summary` and `analyze_complementarity.py` type-safe.
8. **Section registry for `run_full_analysis`.** Collapses 10 hardcoded if/else
   blocks, fixes the 6b skip bug. Small but clarifying.
9. ~~**Fate of `t3_investigation.py` and `geometric_trio/{intrinsic_dimension, ccgp, delta_hyperbolicity, verification_runner}.py` standalone CLIs.**~~ **DONE (Phase 1)** — all 5 deleted. `geometric_trio/` now contains only `data_loader.py` (load-bearing shared dep) + `results/` (historical outputs). Analysis functionality lives in `unified_runner/geometry.py` (`run_intrinsic_dimension`, `run_ccgp`, `run_topology`) and is the living code path.
10. **Retrofit `state_extractor.extract_tier*` to `FeatureFamilyResult` contract.**
    High value, high blast radius — last.

## Relationship to `phase_0/`

`CLAUDE.md:46` tags `phase_0/` as a frozen archive. Two pipeline scripts
parallel Phase 0 originals:

- `pipeline/.../scripts/run_cross_run_transfer.py` (204 lines, 8B) ←→ `phase_0/scripts/run_cross_run_transfer.py` (383 lines, 3B)
- `pipeline/.../scripts/run_judge_scoring.py` ←→ `phase_0/scripts/run_judge_scoring.py`

These are **not literal ports**: the pipeline versions adapt Phase-0.5
methodology to 8B data and new infrastructure (e.g., `phase_0`'s
`run_judge_scoring` imports `config.PROCESSING_MODES`; the pipeline version
re-hardcodes `VALID_MODES`). So they should not be unified. But readers
reaching either version should know the other exists — currently no
cross-reference links them. Adding a line in each docstring ("counterpart at
phase_0/scripts/...") would close that seam.

## Open questions for the maintainer

Items where "is this live?" depends on user intent, not code inspection:

- `test_fast_postprocess.py`, `compare_streaming_vs_hf.py`: intended as
  permanent regression tests, or one-shot validation that served its purpose?
- `profile_generate_overhead.py`: keep as an ongoing tool (future optimization)
  or archive with the one-offs?
- `geometric_trio/{ccgp, intrinsic_dimension, delta_hyperbolicity, verification_runner}.py` standalone CLIs: superseded by `unified_runner/` (verified — ID,
  CCGP, and delta-hyperbolicity all have equivalents in
  `unified_runner/geometry.py`), or kept deliberately as versioned snapshots?
- `t3_investigation.py`: is the "T3 alive at 8B, dead at 3B" question closed,
  or still an open probe you'd rerun on future models?
- `run_8b_r2_experiment.py` + `run_cross_run_transfer.py`: should be
  README-documented as part of the canonical set, or are they
  "one-experiment-deep" and fine as-is?
- Any other confound / hypothesis probes in the wider project that relate to
  scripts here but weren't caught?
