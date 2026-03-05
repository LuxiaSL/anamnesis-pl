# vLLM Compatibility Requirements

## What this pipeline needs from inference

The anamnesis extraction pipeline captures internal model states during autoregressive generation. These states are used to compute features that fingerprint *how* a model processes a prompt (its computational mode), orthogonal to *what* it generates (content).

This document specifies what the pipeline requires from the inference engine, and where vLLM's architecture creates gaps.

## Required tensors per generation step

### 1. Pre-RoPE key projections (critical — T2.5 features)

**What:** The output of each layer's `k_proj` linear layer, *before* rotary position embeddings are applied.

**Shape:** `[batch, num_kv_heads, seq_len_at_step, head_dim]` — during generation, `seq_len_at_step = 1`.

**Why pre-RoPE:** Post-RoPE keys have absolute position baked in via rotation. Key drift, key novelty, and lookback ratio features measure how the model's key representations evolve over generation time. Position-encoded keys would confound this — you'd measure position change, not representational change.

**Current implementation:** Forward hooks on `k_proj` linear layers at sampled layers (7 of 32). See `anamnesis/extraction/model_loader.py:51-73`.

**vLLM gap:** vLLM fuses `q_proj`/`k_proj`/`v_proj` into a single `qkv_proj` in many model implementations, and applies RoPE immediately in the fused attention kernel. There is no clean hook point between the linear projection and RoPE application. PagedAttention operates on post-RoPE keys.

### 2. Attention weights (critical — T2 features)

**What:** Full attention weight matrices per layer per head, per generation step.

**Shape:** `[num_layers, num_query_heads, current_seq_len]` per step (the query-side distribution over all keys).

**Why:** Attention entropy, head agreement (generalized JSD), system-prompt attention mass, recency bias, and region-based attention decomposition all require the full attention distribution.

**Current implementation:** `attn_implementation="eager"` in HuggingFace, which uses standard scaled dot-product attention and returns weights. See `anamnesis/config.py:44`.

**vLLM gap:** vLLM uses FlashAttention / PagedAttention by default, neither of which materializes the full attention matrix. This is a fundamental performance optimization — materializing attention is O(n^2) in memory. There is no option to fall back to eager attention in vLLM's standard path.

### 3. Hidden states (important — T1, T3 features)

**What:** Residual stream activations at each layer, per generation step.

**Shape:** `[num_layers+1, hidden_dim]` per step (index 0 is embedding output, 1..N are transformer layer outputs).

**Why:** Activation norms, residual deltas, spectral features, and PCA projections. Also used by gate features (SwiGLU activation patterns) and residual trajectory features.

**Current implementation:** `output_hidden_states=True` in HuggingFace generate. See `anamnesis/extraction/streaming_generate.py`.

**vLLM gap:** Partial support. vLLM can return hidden states from the final layer via its API, but per-layer hidden states at every generation step require model-level hooks or modifications to the forward pass. The `SamplerOutput` / `SequenceOutput` interface doesn't expose intermediate layer states.

### 4. Logits (important — T1 features)

**What:** Full vocabulary logits at each generation step.

**Shape:** `[vocab_size]` per step.

**Why:** Top-k entropy, probability mass distribution, token surprise, rank statistics.

**Current implementation:** `output_logits=True` passed to `generate()`. Note: this is NOT a default HuggingFace config option — it must be explicitly passed. See `anamnesis/config.py:78`.

**vLLM gap:** Minimal. vLLM exposes logprobs via its API (`logprobs` parameter). Full logits may require a custom `Logits` processor or hook, but this is the most tractable gap.

## Sampled layers (not all layers needed)

The pipeline does NOT need states from every layer. It samples 7 layers at proportional depth positions:

- **8B (32 layers):** `[0, 8, 16, 20, 24, 28, 31]` — denser at 60-80% depth
- **3B (28 layers):** `[0, 7, 14, 18, 21, 24, 27]`

This is important for any implementation: the overhead of capturing states at 7 layers is much lower than capturing all 32.

## What a minimal vLLM integration would look like

The fundamental requirement is a **per-step callback** during autoregressive generation that receives:

```python
@dataclass
class StepState:
    """Minimum viable state capture per generation step."""
    step_idx: int
    pre_rope_keys: dict[int, Tensor]      # layer_idx -> [1, num_kv_heads, 1, head_dim]
    hidden_states: dict[int, Tensor]       # layer_idx -> [1, hidden_dim]
    attention_weights: dict[int, Tensor]   # layer_idx -> [num_heads, current_seq_len]
    logits: Tensor                         # [vocab_size]
```

This callback could be registered before generation and called after each forward pass, before the next token is sampled. The key constraint is that `pre_rope_keys` must be captured *before* RoPE application, which means the hook point must be inside the attention module, between the linear projection and the rotary embedding.

## Alternatives to full vLLM integration

1. **HuggingFace with streaming generation** (current approach) — works, but 10x slower than vLLM for the same model due to lack of PagedAttention, continuous batching, etc. Adequate for research-scale experiments (hundreds of generations), not for production-scale probing.

2. **vLLM fork with hook points** — add pre-RoPE capture points and optional eager attention fallback for sampled layers only. The performance cost of eager attention on 7/32 layers may be acceptable.

3. **Hybrid approach** — use vLLM for generation but replay the prompt + generated tokens through a HuggingFace model with hooks for state extraction. Doubles compute but avoids modifying vLLM. Only viable if the extraction model is smaller than the generation model.

4. **Custom CUDA kernels** — write FlashAttention variants that also output attention weights (exists in some research codebases). High engineering cost.

## Feature tier dependency on tensors

| Tier | Depends on | Features | Signal contribution |
|------|-----------|----------|-------------------|
| T1 | hidden_states, logits | Activation norms, logit stats, token dynamics | Redundant under format control |
| T2 | attention_weights | Entropy, head agreement, residual deltas | Core signal (attention flow) |
| T2.5 | pre_rope_keys | Key drift, novelty, lookback ratio, epoch detection | Load-bearing tier under format control |
| T3 | hidden_states | PCA projections of residual stream | Captures format, not mode (dies under control) |
| attention_flow | attention_weights | Region decomp, recency bias, head diversity | Resolution-agnostic, wins hardest pairs |
| temporal_dynamics | attention_weights, pre_rope_keys | Windowed T2/T2.5 metrics | Scale-dependent (better at 8B) |
| gate_features | hidden_states (MLP internals) | SwiGLU sparsity, drift | Coarse discriminator |
| contrastive_projection | hidden_states | Learned projections | Perfect separation (under investigation) |

The two most critical tensors for the core scientific finding are **pre-RoPE keys** (T2.5) and **attention weights** (T2, attention_flow). These are also the two hardest to extract from vLLM.
