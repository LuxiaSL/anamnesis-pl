# Anamnesis Pipeline

Extraction and analysis pipeline for transformer computational state signatures.

**Core question:** Can a transformer's internal states fingerprint *how* something was processed (reasoning mode), orthogonal to *what* was processed (content)?

This repo contains the full pipeline used in the Anamnesis experiments — from hooking into model internals during generation, through feature extraction, to classification and geometric analysis.

## What this does

1. **Extraction** — During autoregressive generation, captures per-step internal states: pre-RoPE key projections, attention weights, hidden states, and logits. These are collected via forward hooks on sampled layers (7 of 32), with no modification to the model itself.

2. **Feature engineering** — Raw tensors are transformed into feature vectors across multiple tiers:
   - **T1:** Activation norms, logit statistics, token probability dynamics
   - **T2:** Attention entropy, head agreement (generalized JSD), residual deltas
   - **T2.5:** KV cache dynamics — key drift, key novelty, lookback ratio, epoch detection
   - **T3:** PCA projections of the residual stream
   - **Engineered families:** Attention flow (region decomposition, recency bias), temporal dynamics (windowed T2/T2.5), gate features (SwiGLU sparsity), contrastive projections

3. **Analysis** — Classification (Random Forest, contrastive kNN), tier ablation, geometric verification (intrinsic dimension, CCGP, delta-hyperbolicity), clustering, semantic controls.

## Key results

| Metric | Value |
|--------|-------|
| 8-way RF (chance=12.5%) | 50.6% (8B), 51.2% (3B) — engineered features only |
| 5-way RF (chance=20%) | 70% (8B), 55% (3B) — hard modes only |
| Contrastive kNN (5-way, topic-heldout) | 85% (T2+T2.5, 8B) |
| Prompt-swap control | Signal tracks execution mode, not prompt content |
| Length-only baseline | At chance — signal is not confounded by generation length |

The load-bearing signal tier is **T2.5 (KV cache dynamics)**, which measures how the model's key representations evolve during generation. This signal is execution-based, format-resistant, and concentrated in temporal dynamics rather than baseline statistics.

## Structure

```
anamnesis/
├── config.py                      # Model, generation, extraction, experiment configs
├── extraction/                    # Core pipeline — touches the model
│   ├── model_loader.py            # Hook architecture (pre-RoPE key capture)
│   ├── streaming_generate.py      # Efficient autoregressive loop with state collection
│   ├── generation_runner.py       # Orchestration: generate → collect → extract → save
│   ├── state_extractor.py         # Raw tensors → feature vectors (pure numpy)
│   ├── raw_saver.py               # Tensor serialization for offline reprocessing
│   ├── feature_pipeline.py        # v2 pluggable feature families over saved tensors
│   └── feature_families/          # Engineered feature extractors
│       ├── attention_flow.py      # Region decomposition, recency bias, head diversity
│       ├── temporal_dynamics.py   # Windowed T2/T2.5 with STFT
│       ├── gate_features.py       # SwiGLU sparsity and drift
│       ├── contrastive_projection.py  # Learned contrastive projections
│       ├── residual_stream.py     # Trajectory features (velocity, curvature)
│       └── operators.py           # Shared temporal operators (window stats, slopes)
├── analysis/                      # Downstream analysis
│   ├── unified_runner/            # Main analysis pipeline
│   │   ├── classification.py      # RF, logistic regression, cross-validation
│   │   ├── contrastive.py         # Contrastive kNN evaluation
│   │   ├── tier_ablation.py       # Leave-one-out and pairwise tier analysis
│   │   ├── clustering.py          # K-means, silhouette analysis
│   │   ├── geometry.py            # Distance matrices, hierarchical clustering
│   │   ├── semantic.py            # TF-IDF baselines, semantic disambiguation
│   │   ├── integrity.py           # Data quality checks
│   │   └── data_loading.py        # Unified data loading across runs
│   └── geometric_trio/            # Geometric verification
│       ├── intrinsic_dimension.py # MLE, TwoNN, DADApy estimators
│       ├── ccgp.py                # Cross-condition generalization performance
│       └── delta_hyperbolicity.py # Gromov hyperbolicity
├── modes/                         # Processing mode definitions (system prompts)
│   ├── extended_modes.py          # 8 modes (5 core + structured, compressed, associative)
│   ├── run4_modes.py              # Phase 0 compatible 5-way modes
│   └── prompt_swap.py             # Prompt-swap control conditions
├── scripts/                       # Experiment runners
│   ├── run_8b_experiment.py       # Main extraction experiment
│   ├── run_8b_calibration.py      # Positional decomposition calibration
│   ├── run_extraction.py          # Unified extraction (multi-model, multi-mode)
│   ├── run_unified_analysis.py    # Main analysis entry point
│   ├── run_binary_prompt_swap.py  # Prompt-swap control analysis
│   ├── analyze_complementarity.py # Cross-tier complementarity analysis
│   ├── run_subfamily_decomp.py    # Sub-family decomposition
│   ├── train_contrastive_projection.py  # Contrastive MLP training
│   └── run_judge_scoring.py       # LLM judge evaluation
├── prompts/
│   └── prompt_sets.json           # 20 topics × mode prompts
docs/
└── vllm-requirements.md           # What this pipeline needs from inference engines
```

## Requirements

- Python 3.11+
- PyTorch 2.1+
- A HuggingFace-supported model (tested with Llama 3.1 8B Instruct, Llama 3.2 3B Instruct)
- GPU with enough VRAM to run the target model with `attn_implementation="eager"`

```bash
uv sync

# For geometric analysis (intrinsic dimension, etc.):
uv sync --extra geometry
```

## Quick start

### 1. Calibration (positional mean subtraction)

```bash
python -m anamnesis.scripts.run_8b_calibration
```

### 2. Run extraction

```bash
# Full 8B experiment (20 topics × 5 modes × 2 reps = 200 generations)
python -m anamnesis.scripts.run_8b_experiment

# Or unified extraction with custom modes/model
ANAMNESIS_RUN_NAME=my_run python -m anamnesis.scripts.run_extraction
```

### 3. Feature engineering (v2 families over saved raw tensors)

```bash
python -m anamnesis.extraction.feature_pipeline \
    --run my_run \
    --families attention_flow,temporal_dynamics,gate_features \
    --workers 8
```

### 4. Analysis

```bash
# Full analysis suite
python -m anamnesis.scripts.run_unified_analysis --run my_run

# Subset of modes
python -m anamnesis.scripts.run_unified_analysis --run my_run \
    --modes linear,socratic,contrastive,dialectical,analogical

# Prompt-swap controls
python -m anamnesis.scripts.run_binary_prompt_swap --run my_run
```

## Technical notes

- **`attn_implementation="eager"` is required.** Flash/SDPA attention don't return attention weights. The experiment silently produces garbage features without this.
- **Pre-RoPE keys are captured via hooks on `k_proj` linear layers**, not from the KV cache. Post-RoPE keys have rotational position baked in, which would confound geometric features.
- **Hidden states indexing is `[t][l+1]`** — index 0 is the embedding layer output, not layer 0.
- **Head agreement uses generalized JSD** (entropy of mean − mean of entropies), not pairwise KL. O(H) not O(H^2).

See [docs/vllm-requirements.md](docs/vllm-requirements.md) for what would be needed to run this pipeline on top of vLLM or other optimized inference engines.

