# Anamnesis Pipeline

Extraction and analysis pipeline for transformer computational state signatures.

## What is this?

When a transformer generates text, its internal states — attention patterns, key-value cache evolution, activation dynamics — form a temporal trace of *how* the computation unfolded. This project investigates whether that trace carries structured information about the computational process itself, orthogonal to the semantic content produced.

The analogy is an **EKG**: a general-purpose diagnostic that measures temporal dynamics of a system's operation without requiring a mechanistic theory of what each signal component means. We don't need to know *what* the computations are doing in any meaningful sense; we need to know that they happened, how they're stored in memory, and then do comparisons.

**Anamnesis** (Greek: "recollection") — the KV cache is a computational trace archive. It records the temporal dynamics of information routing. If processing mode affects how information routes through attention and MLP layers, those differences are recorded as searchable state signatures. This pipeline extracts and analyzes those signatures.

## The hypothesis

Transformers carry a structured internal axis that tracks *how* something is processed, independent of *what* is produced. This axis is:

- **Execution-based** — it reflects what the model *does*, not what it was *told*
- **Semantically orthogonal** — semantic embeddings of the output text carry effectively zero information about computational features (median R^2 = -1.11 for text-to-compute prediction)
- **Architecturally localized** — concentrated in attention routing and KV cache dynamics, not in output statistics
- **Geometrically structured** — lives on curved manifolds requiring nonlinear access to extract

### What this is NOT about

This project does not classify "reasoning styles" or detect cognitive modes. The processing modes used in experiments are **scaffolding** — experimental variables that force the model into distinguishable computational regimes so we can test whether the internal dynamics differ. The modes are coarse bins on a continuous computational manifold; the signature contains the full computational state, of which mode information is one lossy projection.

The claim is not "we solved mode detection." It's "we demonstrated a phenomenon exists — computation creates structured, measurable internal dynamics orthogonal to content — here's one model's instantiation of it, and here's what should transfer."

## Key findings

### The tier inversion

As surface confounds are progressively stripped away, discriminative power migrates from shallow output statistics to deep computational dynamics:

| Control level | T1 (logit stats) | T2 (attention) | T2.5 (KV cache) |
|---------------|------------------|----------------|-----------------|
| No format control | dominant | mid | weak |
| Format-controlled | near chance | mid | **dominant** |

Under format control, where all modes produce visually identical paragraph prose, dropping KV cache dynamics costs 11pp but dropping logits costs 1pp. Removing 80% of features unrelated to attention routing and KV cache dynamics actually *improves* discrimination.

### Double dissociation

| Manipulation | T1 (logits) | T2.5 (KV cache) |
|--------------|-------------|-----------------|
| Temperature (0.3 vs 0.9) | 90% | ~chance |
| Processing mode (format-controlled) | ~chance | 64% |

T1 and T2.5 are **functionally independent systems**. Temperature controls token sampling (T1 sensitive); mode controls information routing (T2.5 sensitive). Neither responds to the other's manipulation.

### Execution, not instruction

Prompt-swap test: system prompt says "socratic," user message overrides to "write linearly," model complies. The pipeline classifies these as *computationally indistinguishable from pure linear* (50%, chance). It detects what the model does, not what it was told.

### Semantic independence

Ridge regression predicting 366 compute features from 384-dim semantic embeddings: **363 of 366 features below R^2 = 0.1**. Adding semantic features to the compute model provides zero additional mode information (McNemar's p = 1.000). These are orthogonal information axes.

### Headline numbers

| Metric | Value |
|--------|-------|
| 5-way RF, format-controlled (chance=20%) | 70% (8B), p < 0.001 |
| Contrastive kNN, topic-heldout (T2+T2.5) | 85% (8B) |
| T2+T2.5 super-additivity | 366 features (20%) outperform all 1837 combined |
| Length-only baseline | At chance |

## The pipeline

### 1. Extraction

During autoregressive generation, captures per-step internal states via forward hooks on sampled layers (7 of 32), with no modification to the model:

- **Pre-RoPE key projections** — output of `k_proj` linear layers *before* rotary position embeddings, so position doesn't confound geometric features
- **Attention weights** — requires `attn_implementation="eager"` (flash/SDPA don't materialize the attention matrix)
- **Hidden states** — residual stream activations at each layer
- **Logits** — full vocabulary distribution per step

### 2. Feature engineering

Raw tensors become feature vectors across tiers:

| Tier | Features | What it measures |
|------|----------|-----------------|
| T1 | Activation norms, logit stats, token dynamics | Output distribution properties |
| T2 | Attention entropy, head agreement (generalized JSD), residual deltas | Information routing patterns |
| T2.5 | Key drift, key novelty, lookback ratio, epoch detection | KV cache temporal dynamics |
| T3 | PCA projections of residual stream | Content-axis variance (dies under format control) |
| attention_flow | Region decomposition, recency bias, head diversity | Prompt-vs-generation attention structure |
| temporal_dynamics | Windowed T2/T2.5 with STFT | Time-resolved dynamics |
| gate_features | SwiGLU sparsity and drift | Activation gating patterns |
| contrastive_projection | Learned nonlinear projections | Manifold-unwrapping |

### 3. Analysis

Classification (Random Forest, contrastive kNN), tier ablation, geometric verification (intrinsic dimension, CCGP, delta-hyperbolicity), clustering, semantic controls, prompt-swap controls.

## Architecture dependence is the prediction

The specific features are expected to vary across model families. What should transfer is the *principle*: temporal dynamics of computation-relevant architecture carry processing-mode information. The testable predictions:

- **Within-family** (Llama 3.2 3B -> Llama 3.1 8B): features partially transfer, T2+T2.5 dominance holds
- **Across-family** (Llama -> Mistral -> GPT): feature families need redesign, but the *type* (temporal dynamics of attention/cache/routing) remains right
- **Across-architecture** (dense transformer -> MoE -> state-space): even the feature type may shift, but temporal dynamics of computation-relevant components should hold

## Structure

```
anamnesis/
├── config.py                      # Model, generation, extraction, experiment configs
├── extraction/                    # Core pipeline — touches the model
│   ├── model_loader.py            # Hook architecture (pre-RoPE key capture)
│   ├── streaming_generate.py      # Efficient autoregressive loop with state collection
│   ├── generation_runner.py       # Orchestration: generate -> collect -> extract -> save
│   ├── state_extractor.py         # Raw tensors -> feature vectors (pure numpy)
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
│   └── prompt_sets.json           # 20 topics x mode prompts
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
# Full 8B experiment (20 topics x 5 modes x 2 reps = 200 generations)
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
- **Head agreement uses generalized JSD** (entropy of mean - mean of entropies), not pairwise KL. O(H) not O(H^2).

See [docs/vllm-requirements.md](docs/vllm-requirements.md) for what would be needed to run this pipeline on top of vLLM or other optimized inference engines.
