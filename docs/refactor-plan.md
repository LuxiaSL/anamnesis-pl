# Pipeline Refactor Plan

Companion to `pipeline/docs/liveness-audit.md`. Produced 2026-04-16.
Project is paused; this plan executes cleanups staged by risk.

## Purpose

Reduce parsing tax for future work. Each phase is an atomic, reversible
unit. Phases are designed so a fresh agent with no session context can
execute one by reading this doc plus the audit.

## Prerequisites for any agent executing a phase

1. **Read both docs first** — `pipeline/docs/liveness-audit.md` and this
   file. The audit explains the state; this file explains the moves.
2. **Work in a git worktree** on the `pipeline/` repo
   (`github.com/LuxiaSL/anamnesis-pl`). Do not edit on the main branch
   shared with other sessions.
3. **Verify working tree is clean** in the worktree before starting.
4. **Stay in the phase's scope.** Don't opportunistically fix other things
   flagged in the audit — cross-cutting changes belong to their assigned
   phase for clean rollback.
5. **One commit per logical change** per phase. Never squash across
   subtasks unless the phase brief explicitly says so.
6. **Commit message style** — imperative, "why" over "what", reference
   `liveness-audit.md` for provenance.
7. **Strong typing + error handling** — Luxia prefers pydantic types at
   boundaries and explicit error paths, not silent fallthrough.

## Phase sequencing

```
Phase 1 (Deletions)
    ↓
Phase 2 (Mechanical cleanups)  ← start after Phase 1 lands
    ↓
Phase 3 (Medium refactors)     ← independent from P2; can be done after
    ↓
Phase 4 (Deferred)             ← not in scope now
```

Phase 1 should land before Phase 2's README update so the updated scripts
listing reflects post-deletion state. Phase 2 and Phase 3 touch different
concerns (config/docs vs analysis types), so Phase 3 can proceed as soon
as Phase 2 lands — no hard dependency on subtask ordering within Phase 3.

---

## Phase 1 — Deletions

**Goal:** Remove 8 files the audit identified as one-offs whose purpose
is served (outputs preserved in `outputs/` or `geometric_trio/results/`)
or whose functionality is faithfully reproduced in `unified_runner/`.

**Scope:** Deletions only. No edits to remaining files beyond what's
required to keep imports resolving.

**Preconditions:**
- Phase 1 working tree clean.
- Verify `run_8b_r2_experiment.py` and `run_cross_run_transfer.py` mtimes
  are older than the last push by the user. If Luxia is actively editing
  them, pause and confirm before proceeding.

### Files to delete

1. `anamnesis/scripts/test_fast_postprocess.py` — A/B benchmark, served.
2. `anamnesis/scripts/compare_streaming_vs_hf.py` — one-shot validation, served.
3. `anamnesis/scripts/profile_generate_overhead.py` — one-shot profiler; future optimization will want a fresh profiler tailored to the target.
4. `anamnesis/analysis/t3_investigation.py` — closed investigation, outputs at `outputs/analysis/t3_investigation/`.
5. `anamnesis/analysis/geometric_trio/verification_runner.py` — ~400 lines of verbatim duplicates of `unified_runner/utils.py` + `geometry.py`; output at `geometric_trio/results/verification_run.json`.
6. `anamnesis/analysis/geometric_trio/intrinsic_dimension.py` — faithful in `unified_runner/geometry.py:run_intrinsic_dimension`.
7. `anamnesis/analysis/geometric_trio/ccgp.py` — faithful in `unified_runner/geometry.py:run_ccgp` / `_ccgp_variant`.
8. `anamnesis/analysis/geometric_trio/delta_hyperbolicity.py` — faithful in `unified_runner/geometry.py:run_topology` (`gromov_delta_euclidean` block).

### Files to KEEP (do not delete)

- `anamnesis/analysis/geometric_trio/data_loader.py` — load-bearing shared dependency. `unified_runner/` imports `Run4Data`, `TIER_KEYS`, `BASELINE_TIERS`, `ENGINEERED_TIERS` from it.
- `anamnesis/analysis/geometric_trio/results/*.json` — historical outputs from the archived CLIs. Leave in place as provenance.
- `anamnesis/analysis/geometric_trio/__init__.py` — package marker.

### Verification steps

Run each from `pipeline/` working directory (activated venv if needed):

```bash
# Every remaining importable path still loads
python -c "from anamnesis.analysis.unified_runner import run_full_analysis; print('ok')"
python -c "from anamnesis.analysis.geometric_trio.data_loader import load_run4, Run4Data; print('ok')"
python -c "from anamnesis.analysis.unified_runner.geometry import run_intrinsic_dimension, run_ccgp, run_topology; print('ok')"

# No straggling imports
grep -rn "from anamnesis.analysis.t3_investigation\|from anamnesis.analysis.geometric_trio.verification_runner\|from anamnesis.analysis.geometric_trio.intrinsic_dimension\|from anamnesis.analysis.geometric_trio.ccgp\|from anamnesis.analysis.geometric_trio.delta_hyperbolicity" anamnesis/ || echo "no orphan imports"
```

If any grep returns a hit, investigate before deleting — there may be a
consumer the audit missed.

### Commit

Single atomic commit. Suggested message:

```
Delete superseded one-offs and standalone CLIs

- scripts/test_fast_postprocess.py, compare_streaming_vs_hf.py,
  profile_generate_overhead.py: served their purpose; not regression
  tests. Future optimization work would want a fresh profiler.
- analysis/t3_investigation.py: closed investigation (>72h inactive),
  outputs preserved at outputs/analysis/t3_investigation/.
- analysis/geometric_trio/{verification_runner,intrinsic_dimension,
  ccgp,delta_hyperbolicity}.py: verification_runner duplicates
  unified_runner helpers verbatim; the other three are faithfully
  reproduced in unified_runner/geometry.py. Results preserved at
  analysis/geometric_trio/results/.

See pipeline/docs/liveness-audit.md for provenance.
```

### Post-phase update

Edit `pipeline/docs/liveness-audit.md`:
- Strike-through or remove the deleted rows from "Archive candidates"
  section.
- Update "Proposed cleanup order" item 6 and 9 to reflect completion.

### Rollback

`git revert <commit>` brings all 8 files back. The pipeline repo's git
history is the only safety net required — no output files were touched.

---

## Phase 2 — Mechanical cleanups

**Goal:** Collapse documented duplication and doc drift with low-risk
mechanical changes. Every subtask is a single-commit move.

**Scope:** Six subtasks, one commit each. Order within the phase is not
critical except that the README update (2a) should come last so it can
reference the final post-phase state.

**Preconditions:**
- Phase 1 committed.
- Fresh worktree, clean tree.

### 2a. Fix README drift

**Touch:** `pipeline/README.md`

**Changes:**
1. **Quick Start §3 (lines 173-178)** — fix `feature_pipeline` CLI docs.
   Replace the hallucinated `--run`/`--families` example with the real
   CLI. Example to use:
   ```bash
   python -m anamnesis.extraction.feature_pipeline \
       --raw-dir outputs/runs/<run_name>/raw_tensors/ \
       --output-dir outputs/runs/<run_name>/signatures_v2/ \
       --v2 \
       --model 8b \
       --pca-model outputs/calibration/llama31_8b/pca_model.pkl \
       --contrastive-model <path_to_contrastive.npz> \
       --workers 8
   ```
   Document that `--v2` is required to engage the pluggable families
   and that each family can be disabled with `--no-<family>`.

2. **Quick Start** — insert a step between §2 (extraction) and §3
   (feature recomputation) for contrastive projection training:
   ```bash
   # 2.5. Train contrastive projection (once per model; output feeds --contrastive-model)
   python -m anamnesis.scripts.train_contrastive_projection --run <run_name>
   ```
   (Verify the actual CLI args by reading `scripts/train_contrastive_projection.py`'s argparser before writing.)

3. **§Structure `scripts/`** — add entries for scripts that exist
   post-Phase-1 but aren't documented:
   - `run_8b_r2_experiment.py` — R2-equivalent extraction (non-format-controlled process modes).
   - `run_cross_run_transfer.py` — cross-run functional transfer R2↔R3 at 8B.

4. **§Structure `modes/`** — add `run3_original_modes.py` (verbatim Phase-0 R2 process modes, required by `run_8b_r2_experiment.py`).

5. **§Structure `analysis/geometric_trio/`** — after Phase 1 deletions,
   update to show only `data_loader.py` (shared dep) and `results/`
   (historical outputs). Note that geometric trio analyses live in
   `unified_runner/geometry.py` now.

6. **§Structure `analysis/`** — remove `t3_investigation.py` from listing
   if present (it wasn't documented, but double-check).

**Verify:** render README locally or scan for broken references. Every
script/file mentioned must exist after Phase 1.

### 2b. PROCESSING_MODES re-export

**Touch:** `anamnesis/config.py:270-311` (approximately; verify current line numbers before editing)

**Change:** Replace the literal `PROCESSING_MODES` dict with a re-export:

```python
from anamnesis.modes.run4_modes import RUN4_MODES as PROCESSING_MODES
```

Keep `MODE_INDEX` and `ProcessingMode` literal in config.py — they're
downstream of `PROCESSING_MODES.keys()` but the current layout relies on
them being importable from config. Alternative: derive `MODE_INDEX` from
`RUN4_MODE_INDEX` import.

Delete `_FORMAT_CONSTRAINT` from config.py if it's only used to build
`PROCESSING_MODES` (it is, per audit). Leaves `RUN4_MODES` in `modes/` as
the single source.

**Verify:**
```bash
python -c "from anamnesis.config import PROCESSING_MODES, MODE_INDEX, ProcessingMode; print(len(PROCESSING_MODES), len(MODE_INDEX))"
# Expect: 5 5
python -c "from anamnesis.extraction.generation_runner import PROCESSING_MODES; print(list(PROCESSING_MODES.keys()))"
# Expect: same 5 mode names
```

### 2c. Remove unused `StratifiedKFold` import

**Touch:** `anamnesis/analysis/unified_runner/semantic.py:25`

Change:
```python
from sklearn.model_selection import GroupKFold, StratifiedKFold
```
to:
```python
from sklearn.model_selection import GroupKFold
```

**Verify:** `python -c "from anamnesis.analysis.unified_runner.semantic import run_semantic; print('ok')"`

### 2d. Fix Section 6b skip guard

**Touch:** `anamnesis/analysis/unified_runner/__init__.py:250-258` and the `SECTION_KEYS` / `SECTION_NAMES` dicts near the top.

**Bug:** Section 6b (manifold_geometry) runs inside `if 6 not in skip:`
and silently skips without logging when section 6 is skipped.

**Fix:** Promote manifold_geometry to its own section number (suggest
`11`) with its own guard. Add to `SECTION_KEYS` and `SECTION_NAMES`.

```python
SECTION_KEYS: dict[int, str] = {
    1: "integrity",
    ...,
    10: "scorecard",
    11: "manifold_geometry",
}
SECTION_NAMES: dict[int, str] = {
    ...,
    11: "Manifold Geometry",
}
```

Then replace the in-section-6 block with an independent section-11 block
matching the pattern of others (checkpoint save + SKIPPED else branch).

**Verify:**
```bash
python -m anamnesis.scripts.run_unified_analysis --run 8b_v2 --skip 6 --resume  # should skip topology AND manifold_geometry explicitly
python -m anamnesis.scripts.run_unified_analysis --run 8b_v2 --skip 11          # should skip only manifold_geometry
```

### 2e. Consolidate layer presets into `config.py`

**Touch:** `anamnesis/config.py`, `anamnesis/scripts/run_extraction.py:54-87`, `anamnesis/extraction/feature_pipeline.py:463-480`

**Design:** Single `ModelPreset` pydantic model + `MODEL_PRESETS` registry in `config.py`. Consumers import from config.

```python
# In config.py

class ModelPreset(BaseModel):
    model_id: str
    torch_dtype: str
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    sampled_layers: list[int]
    pca_layers: list[int]
    trajectory_layers: list[int]
    contrastive_layers: list[int]
    early_layer_cutoff: int
    late_layer_cutoff: int
    temperature: float
    eos_token_ids: list[int]
    calibration_dir: Path

MODEL_PRESETS: dict[str, ModelPreset] = {
    "8b": ModelPreset(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype="bfloat16",
        num_layers=32, hidden_dim=4096, num_attention_heads=32,
        num_kv_heads=8, head_dim=128,
        sampled_layers=[0, 8, 16, 20, 24, 28, 31],
        pca_layers=[8, 16, 20, 24, 28],
        trajectory_layers=[8, 16, 20, 24, 28],
        contrastive_layers=[8, 16, 20, 24, 28],
        early_layer_cutoff=8, late_layer_cutoff=24,
        temperature=0.6,
        eos_token_ids=[128001, 128008, 128009],
        calibration_dir=OUTPUTS_BASE / "calibration" / "llama31_8b",
    ),
    "3b": ModelPreset(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype="float16",
        num_layers=28, hidden_dim=3072, num_attention_heads=24,
        num_kv_heads=8, head_dim=128,
        sampled_layers=[0, 7, 14, 18, 21, 24, 27],
        pca_layers=[7, 14, 18, 21, 24],
        trajectory_layers=[7, 14, 18, 21, 24],
        contrastive_layers=[7, 14, 18, 21, 24],
        early_layer_cutoff=7, late_layer_cutoff=21,
        temperature=0.7,
        eos_token_ids=[128001, 128009],
        calibration_dir=LEGACY_DATA_ROOT / "outputs" / "calibration",
    ),
}
```

Update `run_extraction.py`: `from anamnesis.config import MODEL_PRESETS` and delete local `MODEL_CONFIGS`. Adapt call sites (the dict-key access `MODEL_CONFIGS["8b"]["sampled_layers"]` becomes attribute access `MODEL_PRESETS["8b"].sampled_layers`).

Update `feature_pipeline.py`: drop `_MODEL_LAYER_PRESETS`, use `MODEL_PRESETS` attribute accesses. The `--model` CLI choice still reads from `MODEL_PRESETS.keys()`.

Also review `ExtractionConfig` defaults in config.py (currently 8B-only). Consider making the defaults unconfigured or deriving from `MODEL_PRESETS["8b"]` at construction time — but this is a design call. Minimum viable: leave `ExtractionConfig` defaults alone; the refactor only targets the duplication.

**Verify:**
```bash
python -c "from anamnesis.config import MODEL_PRESETS; print(MODEL_PRESETS['8b'].sampled_layers, MODEL_PRESETS['3b'].sampled_layers)"
python -m anamnesis.scripts.run_extraction --help | head -20
python -m anamnesis.extraction.feature_pipeline --help | grep model
```

### 2f. `KNOWN_RUNS` → `RunRegistry`

**Touch:** `anamnesis/config.py`, `anamnesis/scripts/run_unified_analysis.py:24-42`

```python
# In config.py

class RunSpec(BaseModel):
    name: str
    signature_dir: Path
    addon_dirs: list[Path] = Field(default_factory=list)
    description: str = ""

RUNS: dict[str, RunSpec] = {
    "8b_baseline": RunSpec(
        name="8b_baseline",
        signature_dir=Path("outputs/runs/run_8b_baseline/signatures"),
        description="Baseline 8B extraction (5 format-controlled modes).",
    ),
    "3b_run4": RunSpec(
        name="3b_run4",
        signature_dir=LEGACY_DATA_ROOT / "outputs" / "runs" / "run4_format_controlled" / "signatures",
        description="Phase-0 3B run4 (format-controlled).",
    ),
    "8b_v2": RunSpec(
        name="8b_v2",
        signature_dir=Path("outputs/runs/8b_fat_01/signatures_v2"),
        addon_dirs=[
            Path("outputs/runs/8b_fat_01/signatures_v2_addon"),
            Path("outputs/runs/8b_fat_01/signatures_v2_contrastive"),
        ],
        description="8B v2 feature-pipeline with engineered families + contrastive.",
    ),
    "3b_v2": RunSpec(
        name="3b_v2",
        signature_dir=Path("outputs/runs/3b_fat_01/signatures_v2"),
        addon_dirs=[Path("outputs/runs/3b_fat_01/signatures_v2_contrastive")],
        description="3B v2 feature-pipeline.",
    ),
}
```

Update `run_unified_analysis.py`:
- `from anamnesis.config import RUNS`
- Replace `KNOWN_RUNS` / `KNOWN_ADDONS` accesses with `RUNS[name].signature_dir` / `RUNS[name].addon_dirs`.
- Keep `--run` / `--sig-dir` CLI flags unchanged (user-facing interface preserved).

**Verify:**
```bash
python -m anamnesis.scripts.run_unified_analysis --run 8b_v2 --skip 1 2 3 4 5 6 7 8 9 10 11  # should just print run header and exit
python -c "from anamnesis.config import RUNS; print(list(RUNS.keys()))"
```

### Post-phase update

Edit `liveness-audit.md`:
- Mark "Proposed cleanup order" items 1, 2, 3, 5 as done.
- Update "Duplication & drift" §1, §2 to note they're resolved.
- Update "README drift" section to note fixes.

### Rollback

Each subtask is its own commit. Revert individually if any breaks.

---

## Phase 3 — Medium refactors

**Goal:** Move result-dict access to typed schemas and replace the
hardcoded section dispatch with a registry. Parseability payoff is
compounding: every future analysis change becomes shorter.

**Scope:** Two independent subtasks. 3a can ship before 3b or after.

**Risk:** medium. 3a touches serialization. 3b touches orchestration.
Both need end-to-end smoke tests before merging.

### 3a. Pydantic result schemas for analysis sections

**Goal:** Replace `results["classification"]["T2+T2.5"]["rf_5way"]["accuracy"]` with `results.classification.by_tier["T2+T2.5"].rf_5way.accuracy`.

**Approach:**

1. **Per-section pass.** For each section file in `unified_runner/` (`integrity.py`, `classification.py`, `tier_ablation.py`, `geometry.py`, `clustering.py`, `contrastive.py`, `semantic.py`, `scorecard.py`), read the current code and document the exact dict shape it returns. Capture this as a pydantic model in a new file `unified_runner/results_schema.py`.

2. **Backward compatibility.** The existing `outputs/analysis/*/results.json` files should still round-trip. Use pydantic v2 `model_validate` / `model_dump` and verify:
   ```python
   import json
   from anamnesis.analysis.unified_runner.results_schema import AnalysisResults
   with open("outputs/analysis/8b_v2/results.json") as f:
       raw = json.load(f)
   parsed = AnalysisResults.model_validate(raw)
   # roundtrip check
   assert parsed.model_dump(exclude_none=True) compatible with raw
   ```

3. **Typed returns.** Update each `run_<section>()` to return its typed
   result. Update the orchestrator in `__init__.py` to accumulate into an
   `AnalysisResults` model. Keep the JSON output structurally identical.

4. **Consumer updates.** `analyze_complementarity.py`, `_print_summary` in
   `__init__.py`, and any other consumer of the result dicts — switch from
   dict access to attribute access. Expect ~50-100 call sites.

**Suggested subtask order:**
- Start with `integrity.py` (smallest, simplest dict shape).
- Move to `classification.py` (key consumer of schema — `by_tier` → `rf_5way/logreg_5way/linear_probe` etc.).
- `tier_ablation.py`, `clustering.py`, `contrastive.py` (uniform shapes).
- `geometry.py` (most complex — ID, CCGP, topology all return nested dicts).
- `semantic.py` (largest, most varied return shape — save for last).
- `scorecard.py` (consumes the others; easiest once schemas exist).

**Constraints:**
- `null` / `None` fields where sections may error out. Use `Optional[...]` consistently.
- `NaN` / `Inf` scrubbing lives in `utils.clean_for_json` — keep it.
- `numpy` arrays inside results should serialize via field validators
  (`model_validator` + explicit `.tolist()`).

**Verify after each section:**
```bash
python -m anamnesis.scripts.run_unified_analysis --run 8b_v2 --resume --skip <all-but-this-one>
diff old-results.json new-results.json  # should be empty or whitespace only
```

**Verify at the end:**
```bash
python -m anamnesis.scripts.analyze_complementarity --include-5way
# outputs should match the existing complementarity_report.json
```

**Commits:** one per section. Plus one for `results_schema.py` bootstrap.

### 3b. Section registry for `run_full_analysis`

**Goal:** Collapse the 10 hardcoded `if N not in skip: ...` blocks in
`unified_runner/__init__.py` into a loop over a `list[SectionSpec]`.

**Design:**

```python
# In unified_runner/__init__.py

@dataclass
class SectionSpec:
    number: int
    name: str
    key: str  # results dict key
    runner: Callable[[AnalysisData, ...], dict | BaseModel]
    requires_text: bool = False  # section 9 needs load_text=True

SECTIONS: list[SectionSpec] = [
    SectionSpec(1, "Data Integrity", "integrity", lazy_import_and_run("integrity", "run_integrity_checks")),
    SectionSpec(2, "Classification", "classification", ...),
    ...,
    SectionSpec(9, "Semantic Independence", "semantic", ..., requires_text=True),
    SectionSpec(10, "Prediction Scorecard", "scorecard", ...),
    SectionSpec(11, "Manifold Geometry", "manifold_geometry", ...),
]
```

Replace the 10 `if N not in skip:` blocks with a single loop. Preserve:
- Lazy imports (don't load `dadapy`, `sentence_transformers`, etc. unless
  that section runs)
- Checkpoint save after each section
- Timing tracking
- SKIPPED logging for skipped sections
- Scorecard behavior (section 10 always re-runs over current results)

**Verify:**
```bash
# Identical behavior end-to-end
python -m anamnesis.scripts.run_unified_analysis --run smoke_test
python -m anamnesis.scripts.run_unified_analysis --run smoke_test --resume  # should skip everything
python -m anamnesis.scripts.run_unified_analysis --run smoke_test --skip 2 4 5 6 7 8 9 10 11  # only integrity
```

**Dependency note:** 3a (typed schemas) and 3b (section registry) can
ship in either order, but after both land, the runner loop becomes
especially clean — each section's `runner` returns a `SectionResult`
subclass and the accumulator appends directly.

### Post-phase update

Edit `liveness-audit.md`:
- Mark "Proposed cleanup order" items 7, 8 as done.
- Update "Latent bugs" section: remove the Section 6b bug (fixed in 2d).

---

## Phase 4 — Deferred

**Item:** Retrofit `state_extractor.extract_tier1/2/2.5/3` and
`extract_all_features` to the `FeatureFamilyResult` contract used by
`feature_families/`. Unify the two extraction contracts.

**Why deferred:**
- 1040 lines of tested logic producing paper results.
- High blast radius: changing tier output structure touches every
  `results.json` in `outputs/`, every saved `signatures_v2/*.npz`, and
  every analysis that reads them.
- Parseability payoff is real but less urgent than the other phases.

**Exit criteria for taking this on:**
- Phases 1-3 complete and stable.
- Test coverage exists for `state_extractor` tier outputs (golden
  comparison against a known-good reference).
- Re-extraction + re-analysis capacity available (full run takes
  hours/days depending on data).

**Tentative sketch (when the time comes):**
1. Add a `tier_name` field to `FeatureFamilyResult` (or a thin subclass
   `TierResult`).
2. Convert `extract_tier1` → returns `TierResult(features=..., feature_names=..., family_name="T1")`.
3. Same for tiers 2, 2.5, 3.
4. `extract_all_features` becomes a simple loop over tier extractors and
   family extractors — both produce the same `TierResult`-like object.
5. `feature_pipeline.compute_features_v2` collapses to one loop, not two.
6. `compute_features_from_raw` (v1) deleted — was just v2 with all families off.

---

## Agent handoff conventions

Each phase brief above is written to be handed to a fresh agent with
this one-line prompt:

> Read `pipeline/docs/refactor-plan.md` and `pipeline/docs/liveness-audit.md`. Execute Phase **N**. Work in a git worktree on the `pipeline/` repo. Do not touch items outside the phase's scope. Report back with: files changed, commit shas, verification commands run, and anything surprising.

**What the agent should bring back:**
- Diff summary
- Commit SHAs (one per subtask in phases 2-3)
- Verification output (paste the commands + results)
- Notes on deviations from the brief (e.g., "found a third caller of
  PROCESSING_MODES, updated too")
- Updated `liveness-audit.md` showing what was completed

**What the agent should NOT do:**
- Opportunistic refactors outside the phase
- "Bundle in" fixes from other phases
- Delete outputs, calibration data, or anything in `outputs/`
- Work on `main` of any repo; always use a worktree
- Skip verification steps

**Review loop:** after each phase, Luxia reviews the worktree diff before
merging to `main`. Don't merge upstream from within the agent.

## Open items (for future planning, not these phases)

From the audit's "Open questions" that aren't yet resolved by a phase:
- Pattern-extraction from `run_8b_r2_experiment.py` +
  `run_cross_run_transfer.py` into a shared cross-run module. Wait for
  that experiment to settle, then review.
- Whether to add structured cross-references between `pipeline/` and
  `phase_0/` counterpart scripts (`run_cross_run_transfer.py`,
  `run_judge_scoring.py`).
- Whether `analysis/geometric_trio/` as a subpackage continues to make
  sense after Phase 1 empties it — maybe fold `data_loader.py` into
  `analysis/` directly or into `unified_runner/`.
