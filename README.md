# Anamnesis Pipeline

Extraction, replay, steering, and analysis pipeline for transformer **computational state
signatures**.

## What is this?

When a transformer generates text, its internal states — attention patterns, key-value cache
evolution, activation dynamics, routing decisions — form a temporal trace of *how* the
computation unfolded. This project investigates whether that trace carries structured information
about the computational process itself, orthogonal to the semantic content produced.

The analogy is an **EKG**: a general-purpose diagnostic that measures temporal dynamics of a
system's operation without requiring a mechanistic theory of what each signal component means. We
don't need to know *what* the computations are doing in any meaningful sense; we need to know
that they happened, how they're stored in memory, and then do comparisons.

**Anamnesis** (Greek: "recollection") — the KV cache is a computational trace archive. It records
the temporal dynamics of information routing. If processing differences affect how information
routes through attention, MLP, and (on MoE models) expert selection, those differences are
recorded as searchable state signatures. This pipeline extracts those signatures, replays them
deterministically, intervenes on them, and analyzes them.

## The hypothesis

Transformers carry a structured internal axis that tracks *how* something is processed,
independent of *what* is produced: execution-based (reflecting what the model *does*, not what it
was *told*), semantically orthogonal (independent of content embeddings), and sub-semantic
(persisting under format control). Phase 0 established evidence for each leg (see the published
paper via the parent project); some of the paper's secondary claims (signal localization, linear
accessibility) were revised by later audits — treat the paper as the phenomenon's existence
proof, not the current state of knowledge.

### What this is NOT about

This project does not classify "reasoning styles" or detect cognitive modes. The processing modes
used in experiments are **scaffolding** — experimental variables that force the model into
distinguishable computational regimes so we can test whether the internal dynamics differ. The
modes are coarse bins on a continuous computational manifold; the signature contains the full
computational state, of which mode information is one lossy projection.

## What it captures

During generation (always with **eager attention** — this is load-bearing, see Gotchas):

- **Attention weights** — the routing decisions: entropy, head agreement (generalized JSD),
  region decomposition, recency structure, and their temporal dynamics.
- **Pre-RoPE key/query/value projections** — cache content stripped of positional encoding,
  captured via forward hooks on the projection linears (on MLA models, the compressed KV latent).
- **Hidden states** — the residual stream: norms, inter-layer deltas, spectral properties,
  trajectory geometry.
- **SwiGLU gate activations** — sparsity, drift, effective dimension.
- **Logits** — output-distribution statistics per step (raw saving banks top-k).
- **Router distributions** (MoE presets) — dense pre-top-k expert allocation per MoE layer.

All model specifics (layer counts, sampled depths, EOS ids, decode params, attention layout) live
in one place: the `ModelPreset` registry in `anamnesis/config.py`.

## Capabilities

**Extraction** — `model_loader.py` (hooks + generation) → `generation_runner.py` (orchestration)
→ `state_extractor.py` (raw tensors → features; **pure numpy, GPU-free, testable without a
model** — a deliberate design constraint). This three-layer split is intentional; don't collapse
it.

**Feature families** (`extraction/feature_families/`) — pluggable, self-contained extractors over
saved raw tensors, toggled per run; the set extends without touching the core:

| family | reads |
|---|---|
| `attention_flow` | where attention goes: region decomposition, recency bias, head diversity |
| `temporal_dynamics` | windowed temporal decomposition of attention/cache metrics, with STFT |
| `gate_features` | SwiGLU gate sparsity, drift, effective dimension |
| `residual_stream` | residual trajectory: velocity, curvature, directness |
| `per_head` | per-head heterogeneity that head-averaging destroys |
| `value_geometry` | v_proj value-vector geometry (the OV-circuit storage surface) |
| `qk_geometry` | pre-RoPE query trajectory + q·k content alignment (RoPE-invariant) |
| `key_cka` | cross-layer KV-cache CKA: basis-invariant structure agreement across depth |
| `expert_routing` | MoE router reads: allocation entropy/margin/coverage/load, shared-vs-routed mass, switch/churn/drift, cross-layer routing CKA (empty on dense models) |
| `contrastive_projection` | learned contrastive projections of raw hidden states |
| `attn_res` | cross-block routing surface for from-scratch models with native capture fields |
| `operators` | shared temporal operators used by the families |

**Feature taxonomy** — `analysis/feature_map.py` tags every feature by SOURCE (which substrate
was read) × METHOD (operator class) × DYNAMIC (static vs temporal) × DEPTH (layer band). Pure
numpy/pydantic, name-based, auditable (`unclassified() == 0` is the acceptance check). This
replaced an earlier "tier" (T1/T2/T2.5/T3) vocabulary, which is **retired**: tier names survive
only as legacy toggles for the baseline feature blocks, not as a conceptual frame.

**Replay & determinism** — a decoder forward is a deterministic function of its input tokens, so
any banked generation can be re-processed in a single teacher-forced instrumented forward that
reproduces the per-position states of the original incremental generation (`replay_extract.py`;
manifests built and *strongly validated* by `replay_manifest.py` — reconstruction failures are
flagged, never silently dropped). Determinism is enforced bit-exactly by tests
(`tests/test_multicell_bitwise.py` and friends). This means raw tensors never need to be stored
long-term: keep tokens, replay states on demand.

**Intervention (the causal half)** — the replay path also accepts modified state:
- **KV-cache surgery** (`cache_surgery.py` + `replay_cached.py`): eviction, exact RoPE
  re-rotation, recompute — with a load-time gate that refuses unknown RoPE schemes.
- **Residual-stream write hooks** (`ResidualWriteSpec` / `attach_residual_write`):
  position-windowed vector injection for activation steering, usable in generation and replay.
- **MoE router perturbation** (`MoEPerturbSpec` / `attach_moe_perturbation`): router-logit
  noise/drop/top-k modification on MoE presets.

**Analysis** — `analysis/unified_runner/` (the multi-section classification/geometry gauntlet
with checkpoint/resume) · `analysis/battery/` (a single typed analysis template applied to every
experiment arm × model: pre-registered prediction manifests, empirically estimated noise floors,
paired-delta effect sizes with floor-ruled family decomposition, map-level FDR statistics —
every emitted number carries its n, aggregation law, and floor type) · `analysis/geometric_trio/`
(intrinsic dimension, CCGP, δ-hyperbolicity) · `analysis/v3_audit/` (frozen corpus-audit suite).

**Scripts** (`anamnesis/scripts/`, large and functionally clustered): generation
(`run_extraction`, `parallel_generate`, `run_gen_tokens`, `vmb_a5_gen_multicell`) · replay
(`run_replay_extraction`, `parallel_replay`, `persistent_replay_worker`) · vector building
(`vmb_a5_build_vectors`, `vmb_a5_whiten_steer_build`, `vmb_a5_build_v4_gradient`,
`vmb_a5_covariance_screen`) · steering/surgery replays (`vmb_a5_replay_multicell`,
`vmb_a4_surgery_replay`) · judging (`run_judge_scoring`, `run_2afc_mode_hardening`,
`vmb_a5_judge_*`) · arm/battery analyses (`vmb_arm_a*_analyze`, `vmb_subperceptual_census`) ·
`ops/` (scheduler submission; all deployment specifics come from environment variables and are
never committed). Many experiment scripts cite an external research log (preregistration
addenda, arm records) that lives in a separate private repo — dangling references from this repo
are expected.

## Supported models

| preset | model | notes |
|---|---|---|
| `3b` | Llama-3.2-3B-Instruct | 28L, GQA 24q/8kv |
| `8b` | Llama-3.1-8B-Instruct | 32L, GQA 32q/8kv |
| `qwen-7b` | Qwen2.5-7B-Instruct | 28L, GQA 28q/4kv |
| `olmo2-7b` | OLMo-2-1124-7B | base model (no chat template), full MHA, q/k-norm |
| `gemma3-27b` | Gemma-3-27b-it | 62L multimodal wrapper, 5:1 local:global attention interleave |
| `dsv2-lite` | DeepSeek-V2-Lite-Chat | first MoE (2 shared + 64 routed, top-6) + MLA (keys = compressed KV latent); needs a transformers version with native `deepseek_v2` support |

**Adding a model** = one `ModelPreset` entry (architecture dims, sampled/PCA/trajectory/
contrastive layers, EOS token ids, decode params, calibration dir, and — where relevant —
`attention_layer_types`) plus a `MODEL_LAYERS` entry in `analysis/feature_map.py`. The four
non-Llama presets are worked examples of the onboarding caveats: base-vs-chat template, MHA vs
GQA, interleaved attention, MoE/MLA.

## Install & quick start

```bash
uv sync                      # python >=3.11; use --extra geometry for dadapy/scikit-dimension
```

```bash
# generation + extraction (from pipeline/)
python -m anamnesis.scripts.run_extraction --model 8b --modes linear,socratic --n-samples 20 --save-raw --run-name my_run
# engineered features over saved raw tensors
python -m anamnesis.scripts.feature_pipeline --run my_run --v2
# replay states from banked tokens (no stored tensors needed)
python -m anamnesis.scripts.run_replay_extraction --run my_run
# analysis
python -m anamnesis.scripts.run_unified_analysis --run my_run
```

## Technical gotchas (each of these silently corrupts data if missed)

1. **`attn_implementation="eager"` is required** — flash/SDPA don't return attention weights; the
   run completes and produces garbage attention features.
2. **`output_logits=True` must be passed to `generate()`** — without it, output-source features
   are silently absent.
3. **`hidden_states` indexing is `[t][l+1]`** — index 0 is the embedding output, not layer 0.
4. **Pre-RoPE keys come from hooks on the `k_proj` linears**, not the post-RoPE KV cache
   (position would be baked into the geometry).
5. **Skip index 0 for attentions/logits/chosen_ids during prefill.**
6. **GQA**: attention weights index by *query* head count; the KV cache by *KV* head count.
7. **EOS detection needs the explicit per-model token-id list** (in the preset).
8. **`metadata.json` wraps generations under a `"generations"` key**, not a flat list.
9. **Head agreement uses generalized JSD** (entropy-of-mean − mean-of-entropies), O(H).
10. **`OMP_NUM_THREADS=1`** (and MKL/OPENBLAS) for the parallel extractor — it oversubscribes
    otherwise. sklearn RF `n_jobs=-1` deadlocks under joblib here; use `n_jobs=1` + processes.

## Relation to the research program

This repo is the **instrument**. Experimental design, preregistrations, running records, and
result adjudication live in the parent research repo (private); Phase 0's frozen code and paper
live in their own repo. The pipeline is deliberately findings-free: capabilities documented here
should remain true regardless of what the experiments conclude.
