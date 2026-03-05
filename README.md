# Anamnesis Pipeline

Extraction and analysis pipeline for transformer computational state signatures.

## What is this?

When a transformer generates text, its internal states — attention patterns, key-value cache evolution, activation dynamics — form a temporal trace of *how* the computation unfolded. This project investigates whether that trace carries structured information about the computational process itself, orthogonal to the semantic content produced.

The analogy is an **EKG**: a general-purpose diagnostic that measures temporal dynamics of a system's operation without requiring a mechanistic theory of what each signal component means. We don't need to know *what* the computations are doing in any meaningful sense; we need to know that they happened, how they're stored in memory, and then do comparisons.

**Anamnesis** (Greek: "recollection") — the KV cache is a computational trace archive. It records the temporal dynamics of information routing. If processing mode affects how information routes through attention and MLP layers, those differences are recorded as searchable state signatures. This pipeline extracts and analyzes those signatures.

## The hypothesis

Transformers carry a structured internal axis that tracks *how* something is processed, independent of *what* is produced. Specifically:

- **Execution-based** — the signal reflects what the model *does*, not what it was *told*. A prompt-swap test (socratic instruction, linear execution) produces signatures indistinguishable from pure linear.
- **Semantically orthogonal** — semantic embeddings of the output text carry effectively zero information about computational features. These are independent information axes: *what* was discussed vs *how* it was processed.
- **Sub-semantic** — under format control (all modes producing identical-looking paragraph prose), the signal persists. Whatever is being detected lives below the surface of the text.

### What this is NOT about

This project does not classify "reasoning styles" or detect cognitive modes. The processing modes used in experiments are **scaffolding** — experimental variables that force the model into distinguishable computational regimes so we can test whether the internal dynamics differ. The modes are coarse bins on a continuous computational manifold; the signature contains the full computational state, of which mode information is one lossy projection.

The claim is not "we solved mode detection." It's "we demonstrated a phenomenon exists — computation creates structured, measurable internal dynamics orthogonal to content — and we built the first instrument to detect it."

## What we measure and why

The pipeline captures four categories of internal state during generation, each motivated by a different hypothesis about where computational signatures might live:

**Attention weights** — the routing decisions. At each generation step, the model decides how to distribute attention across all previous tokens. If different computational strategies route information differently, this is where it shows up. Measured as entropy, head agreement (generalized JSD), region decomposition (how much attention goes to system prompt vs recent tokens vs middle context), and temporal dynamics of these quantities.

**Pre-RoPE key projections** — the cache's semantic content, stripped of positional encoding. The KV cache accumulates a record of how the model has been processing information. By capturing keys *before* rotary position embeddings are applied, we isolate representational evolution from positional artifacts. Measured as key drift (how fast representations change), key novelty (how different each new key is from the running average), lookback ratio, and epoch detection (when the cache "reorganizes").

**Hidden states** — the residual stream. The running computation at each layer. Measured as activation norms, inter-layer deltas, spectral properties, and trajectory geometry (velocity, curvature through layer-space).

**Logits** — the output distribution. What the model was about to say at each step. Measured as entropy, top-k concentration, token surprise, rank statistics. These turn out to be the *least* informative features under format control — when all modes produce similar-looking text, they produce similar output distributions by necessity.

### Feature tiers

The features are organized into tiers reflecting architectural depth:

| Tier | Source | What it captures |
|------|--------|-----------------|
| T1 | Logits, activation norms | Output distribution properties |
| T2 | Attention weights, residual deltas | Information routing patterns |
| T2.5 | Pre-RoPE keys | KV cache temporal dynamics |
| T3 | Hidden states (PCA) | Content-axis variance |

And engineered families that cut across tiers:

| Family | Source | What it captures |
|--------|--------|-----------------|
| attention_flow | Attention weights | Region decomposition, recency bias, head diversity |
| temporal_dynamics | Attention + KV cache | Time-resolved windowed statistics with STFT |
| gate_features | Hidden states (MLP) | SwiGLU activation sparsity and drift |
| contrastive_projection | Hidden states | Learned nonlinear projections for manifold-unwrapping |

## Architecture dependence is the prediction

The specific features that carry signal are expected to vary across model families — this is a prediction, not a weakness. What should transfer is the *principle*: temporal dynamics of computation-relevant architecture carry processing-mode information.

- **Within-family** (Llama 3.2 3B -> Llama 3.1 8B): features partially transfer, signal principle holds
- **Across-family** (Llama -> Mistral -> GPT): feature families likely need redesign, but the *type* of feature (temporal dynamics of attention/cache/routing) should remain informative
- **Across-architecture** (dense transformer -> MoE -> state-space): even the feature type may shift, but temporal dynamics of computation-relevant components should hold

## The pipeline

### 1. Extraction

During autoregressive generation, captures per-step internal states via forward hooks on sampled layers (7 of 32), with no modification to the model:

- **Pre-RoPE key projections** — hooks on `k_proj` linear layers, *before* rotary position embeddings
- **Attention weights** — requires `attn_implementation="eager"` (flash/SDPA don't materialize the attention matrix)
- **Hidden states** — residual stream activations at each layer
- **Logits** — full vocabulary distribution per step

### 2. Feature engineering

Raw tensors are transformed into feature vectors by `state_extractor.py` (pure numpy, no model dependency) and the pluggable `feature_families/` modules. Features are computed per-generation and saved alongside metadata.

### 3. Analysis

Classification (Random Forest, contrastive kNN), tier ablation, geometric verification (intrinsic dimension, CCGP, delta-hyperbolicity), clustering, semantic controls, prompt-swap controls.

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
│   ├── run4_modes.py              # Canonical 5-way modes
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
