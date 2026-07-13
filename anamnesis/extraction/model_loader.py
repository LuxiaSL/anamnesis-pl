"""Model loading and pre-RoPE hook management.

Architecture-agnostic — all model specifics come from ModelConfig.

Responsibilities:
  - Load model + tokenizer with eager attention (required for attn weights)
  - Register forward hooks on k_proj linear layers to capture pre-RoPE keys
  - Provide hook lifecycle management (register, collect, clear, remove)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from anamnesis.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class HookState:
    """Mutable storage for hook captures across generation steps.

    Tensors are kept on GPU during generation to avoid per-step synchronous
    CPU transfers. Call flush_to_cpu() after generation completes and before
    accessing tensors via get_generation_keys() / get_generation_gates().
    """

    # layer_idx → list of tensors, one per generation step
    # Each tensor shape: [1, num_kv_heads, seq_len_at_step, head_dim]
    #   - During prefill: seq_len_at_step = prompt_length
    #   - During generation: seq_len_at_step = 1
    pre_rope_keys: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    # layer_idx → list of tensors, one per generation step
    # Each tensor shape: [1, seq_len_at_step, intermediate_size]
    #   - During prefill: seq_len_at_step = prompt_length
    #   - During generation: seq_len_at_step = 1
    # This is the gate_proj output BEFORE SiLU activation.
    # Apply SiLU in feature computation to get actual gate values.
    gate_activations: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    # layer_idx → list of v_proj outputs (values), reshaped [1, num_kv_heads, seq_len, head_dim].
    # Captured at the same point as keys (post-projection, GQA layout). For replay (single
    # forward) each list holds one full-sequence tensor; for generate, one per step.
    v_proj_values: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    # layer_idx → list of q_proj outputs (queries, PRE-RoPE), reshaped
    # [1, num_attention_heads, seq_len, head_dim]. Position-free query content;
    # re-apply RoPE offline (with banked positions) for post-RoPE QK geometry.
    queries: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    # layer_idx → list of o_proj outputs (attention-block output, per-token
    # contribution the attention sublayer ADDS to the residual stream),
    # shape [1, seq_len, hidden_dim]. The attention-output surface (vmb Stage A):
    # with block-boundary hidden states banked, mlp_out is derivable as
    # resid_{l+1} − resid_l − attn_out, so this one capture closes two census cells.
    attn_outputs: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    enabled: bool = True
    _on_cpu: bool = field(default=False, repr=False)

    def flush_to_cpu(self) -> None:
        """Batch-transfer all captured tensors from GPU to CPU.

        Replaces ~N*layers synchronous per-step transfers with one batch
        transfer per layer. Must be called after generation completes and
        before accessing tensors via get_generation_keys/gates.
        """
        if self._on_cpu:
            return
        for layer_idx in list(self.pre_rope_keys.keys()):
            tensors = self.pre_rope_keys[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.pre_rope_keys[layer_idx] = [t.cpu() for t in tensors]
        for layer_idx in list(self.gate_activations.keys()):
            tensors = self.gate_activations[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.gate_activations[layer_idx] = [t.cpu() for t in tensors]
        for layer_idx in list(self.v_proj_values.keys()):
            tensors = self.v_proj_values[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.v_proj_values[layer_idx] = [t.cpu() for t in tensors]
        for layer_idx in list(self.queries.keys()):
            tensors = self.queries[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.queries[layer_idx] = [t.cpu() for t in tensors]
        for layer_idx in list(self.attn_outputs.keys()):
            tensors = self.attn_outputs[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.attn_outputs[layer_idx] = [t.cpu() for t in tensors]
        self._on_cpu = True

    def clear(self) -> None:
        """Release all captured tensors."""
        for tensors in self.pre_rope_keys.values():
            del tensors[:]
        self.pre_rope_keys.clear()
        for tensors in self.gate_activations.values():
            del tensors[:]
        self.gate_activations.clear()
        for tensors in self.v_proj_values.values():
            del tensors[:]
        self.v_proj_values.clear()
        for tensors in self.queries.values():
            del tensors[:]
        self.queries.clear()
        for tensors in self.attn_outputs.values():
            del tensors[:]
        self.attn_outputs.clear()
        self._on_cpu = False

    def get_generation_keys(self, layer_idx: int) -> list[Tensor]:
        """Return captured keys for a layer (excluding prefill step 0)."""
        all_keys = self.pre_rope_keys.get(layer_idx, [])
        if len(all_keys) <= 1:
            return []
        # Skip index 0 (prefill), return generation steps only
        return all_keys[1:]

    def get_generation_gates(self, layer_idx: int) -> list[Tensor]:
        """Return captured gate activations for a layer (excluding prefill step 0)."""
        all_gates = self.gate_activations.get(layer_idx, [])
        if len(all_gates) <= 1:
            return []
        return all_gates[1:]


def decoder_layers(model):
    """Decoder layer list across architectures.

    Llama/Qwen/OLMo-2: `model.model.layers`. Gemma-3 (vmb M5) ships as
    Gemma3ForConditionalGeneration — a multimodal wrapper whose text decoder
    nests under `language_model`; hook paths must resolve through it.
    (M5 onboarding audit 2026-07-12; GPU validation pending — journal W10.)
    """
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner.layers
    lm = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if lm is not None:
        lm_inner = getattr(lm, "model", lm)
        if hasattr(lm_inner, "layers"):
            return lm_inner.layers
    raise AttributeError(
        f"cannot locate decoder layers on {type(model).__name__} — extend "
        "decoder_layers() for this architecture"
    )


def _make_k_proj_hook(
    layer_idx: int,
    hook_state: HookState,
    num_kv_heads: int,
    head_dim: int,
) -> Any:
    """Create a forward hook for a k_proj linear layer.

    k_proj output shape: [batch, seq_len, num_kv_heads * head_dim]
    We reshape to [batch, num_kv_heads, seq_len, head_dim] and keep on GPU.
    Call hook_state.flush_to_cpu() after generation to batch-transfer.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        batch, seq_len, _ = output.shape
        reshaped = output.detach().reshape(batch, seq_len, num_kv_heads, head_dim)
        reshaped = reshaped.transpose(1, 2)
        hook_state.pre_rope_keys[layer_idx].append(reshaped)

    return hook_fn


def _make_gate_proj_hook(
    layer_idx: int,
    hook_state: HookState,
) -> Any:
    """Create a forward hook for a gate_proj linear layer (SwiGLU gate).

    gate_proj output shape: [batch, seq_len, intermediate_size]
    Kept on GPU; call hook_state.flush_to_cpu() after generation.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        hook_state.gate_activations[layer_idx].append(output.detach())

    return hook_fn


def _make_v_proj_hook(
    layer_idx: int,
    hook_state: HookState,
    num_kv_heads: int,
    head_dim: int,
) -> Any:
    """Create a forward hook for a v_proj linear layer (attention values).

    v_proj output shape: [batch, seq_len, num_kv_heads * head_dim] (GQA, same
    layout as k_proj). Reshaped to [batch, num_kv_heads, seq_len, head_dim] and
    kept on GPU. Call hook_state.flush_to_cpu() after the forward.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        batch, seq_len, _ = output.shape
        reshaped = output.detach().reshape(batch, seq_len, num_kv_heads, head_dim)
        reshaped = reshaped.transpose(1, 2)
        hook_state.v_proj_values[layer_idx].append(reshaped)

    return hook_fn


def _make_q_proj_hook(
    layer_idx: int,
    hook_state: HookState,
    num_attention_heads: int,
    head_dim: int,
) -> Any:
    """Create a forward hook for a q_proj linear layer (PRE-RoPE queries).

    q_proj output shape: [batch, seq_len, num_attention_heads * head_dim].
    Reshaped to [batch, num_attention_heads, seq_len, head_dim] and kept on GPU.
    Captured pre-RoPE (mirrors the pre-RoPE key design); re-apply RoPE offline
    with banked positions for post-RoPE QK-space geometry.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        batch, seq_len, _ = output.shape
        reshaped = output.detach().reshape(batch, seq_len, num_attention_heads, head_dim)
        reshaped = reshaped.transpose(1, 2)
        hook_state.queries[layer_idx].append(reshaped)

    return hook_fn


def _make_o_proj_hook(
    layer_idx: int,
    hook_state: HookState,
) -> Any:
    """Create a forward hook for an o_proj linear layer (attention output).

    o_proj output shape: [batch, seq_len, hidden_dim] — the attention sublayer's
    additive contribution to the residual stream (pre residual-add). Kept on GPU;
    call hook_state.flush_to_cpu() after the forward.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        hook_state.attn_outputs[layer_idx].append(output.detach())

    return hook_fn


# ── Activation-WRITE path (vmb battery arm A5 / A5-inv) ─────────────────────────
#
# Read hooks above OBSERVE the forward pass; the write path PERTURBS it: a
# forward_pre_hook on a decoder layer adds alpha * unit(vector) to the residual
# stream entering that layer, at chosen sequence positions. Used for steering
# during replay of a fixed continuation (matched-token channel) and free-gen
# dose ladders. alpha=0 must reproduce the unperturbed forward exactly.


@dataclass
class ResidualWriteSpec:
    """One residual-stream injection: add `alpha * vector/||vector||` to the
    hidden states entering decoder layer `layer_idx`.

    start_pos/end_pos select ABSOLUTE sequence positions (end exclusive;
    None = unbounded). For steering over generated tokens only, set
    start_pos = prompt_length. Positional gating reads the layer's
    `cache_position` kwarg (absolute positions, present on every HF call path
    in transformers 5.x), so ONE spec is valid under single-forward replay AND
    incremental generate() (per-step seq_len=1 with KV cache) — no per-step
    toggling. If cache_position is absent the hook falls back to local-index
    slicing (correct only for full-sequence forwards) and RAISES on an
    ambiguous incremental step rather than silently mis-injecting.

    start_pos is read at every hook call, so a caller may mutate it per
    generation (prompt lengths vary) on a hook registered once.
    """

    layer_idx: int
    vector: Tensor              # [hidden_dim]; normalized at registration
    alpha: float
    start_pos: int | None = None
    end_pos: int | None = None
    normalize: bool = True


def _make_residual_write_pre_hook(
    spec: ResidualWriteSpec,
    enabled_flag: dict[str, bool],
    stats: dict[str, Any] | None = None,
) -> Any:
    """forward_pre_hook (with_kwargs) injecting spec into the layer's input hidden states.

    The decoder layer receives hidden_states as args[0] (HF Llama-class layers).
    Returns modified (args, kwargs). Positional gating uses the `cache_position`
    kwarg (absolute positions) when present — valid under prefill, incremental
    seq_len=1 steps, and single-forward replay alike. Injection tensor is
    cast/moved lazily to the input's dtype/device once, then cached on the spec
    closure (cache is keyed by alpha too, so per-cell alpha mutation is safe).
    `stats` (optional) accumulates calls/positions-injected/saw-cache-position
    for smoke assertions.
    """
    cache: dict[str, Tensor] = {}

    def pre_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not enabled_flag["enabled"] or spec.alpha == 0.0:
            return args, kwargs
        cache_position = kwargs.get("cache_position")
        if not args or not isinstance(args[0], Tensor):
            # Defensive: some architectures pass hidden_states as a kwarg.
            hs = kwargs.get("hidden_states")
            if hs is None:
                logger.warning("residual write hook: no hidden_states found; skipping injection")
                return args, kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = _inject(hs, cache_position)
            return args, kwargs
        new_args = (_inject(args[0], cache_position),) + tuple(args[1:])
        return new_args, kwargs

    def _inject(hs: Tensor, cache_position: Tensor | None) -> Tensor:
        key = f"{hs.device}_{hs.dtype}_{spec.alpha}"
        if key not in cache:
            v = spec.vector.detach().to(device=hs.device, dtype=torch.float32)
            if spec.normalize:
                v = v / v.norm().clamp_min(1e-12)
            cache[key] = (spec.alpha * v).to(dtype=hs.dtype)
        delta = cache[key]
        bounded = spec.start_pos is not None or spec.end_pos is not None
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
            stats["saw_cache_position"] = stats.get("saw_cache_position", False) or (
                cache_position is not None
            )
        if cache_position is not None:
            # Absolute-position gating: one semantics for prefill, incremental
            # decoding, and single-forward replay.
            s = spec.start_pos if spec.start_pos is not None else 0
            e = spec.end_pos if spec.end_pos is not None else int(cache_position.max().item()) + 1
            mask = (cache_position >= s) & (cache_position < e)  # [seq_len] bool
            n_inj = int(mask.sum().item())
            if stats is not None:
                stats["positions"] = stats.get("positions", 0) + n_inj
            if n_inj == 0:
                return hs
            if n_inj == hs.shape[1]:
                return hs + delta
            out = hs.clone()
            out[:, mask, :] = out[:, mask, :] + delta
            return out
        # Fallback: local-index slicing — correct ONLY for full-sequence forwards.
        if bounded and hs.shape[1] == 1:
            raise RuntimeError(
                "residual write hook: positionally-bounded spec on a seq_len=1 "
                "forward without cache_position — cannot determine the absolute "
                "position; refusing to silently mis-inject (incremental decoding "
                "requires the cache_position kwarg)"
            )
        s = spec.start_pos if spec.start_pos is not None else 0
        e = spec.end_pos if spec.end_pos is not None else hs.shape[1]
        s = max(0, min(s, hs.shape[1]))
        e = max(s, min(e, hs.shape[1]))
        if stats is not None:
            stats["positions"] = stats.get("positions", 0) + (e - s)
        if s == 0 and e == hs.shape[1]:
            return hs + delta
        out = hs.clone()
        out[:, s:e, :] = out[:, s:e, :] + delta
        return out

    return pre_hook


@dataclass
class ResidualWriteHandle:
    """Removable handle for a registered residual write; also supports toggling.

    `stats` accumulates {calls, positions, saw_cache_position} across forwards
    (reset via reset_stats) — smoke tests assert injected-position counts against
    expectation. `spec.start_pos`/`spec.alpha` may be mutated between generations
    on a live handle (read per hook call)."""

    spec: ResidualWriteSpec
    _handle: Any
    _enabled_flag: dict[str, bool]
    stats: dict[str, Any] = field(default_factory=dict)

    def disable(self) -> None:
        self._enabled_flag["enabled"] = False

    def enable(self) -> None:
        self._enabled_flag["enabled"] = True

    def remove(self) -> None:
        self._handle.remove()

    def reset_stats(self) -> None:
        self.stats.clear()


def attach_residual_write(model: Any, spec: ResidualWriteSpec) -> ResidualWriteHandle:
    """Register a residual-write injection on a BARE HF model (no LoadedModel needed).

    Validates layer_idx/hidden_dim against model.config. Used by generation-side
    scripts (run_gen_tokens) that never construct a LoadedModel; the LoadedModel
    method delegates here so both paths share one implementation.
    """
    layers = decoder_layers(model)
    n_layers = len(layers)
    hidden_dim = int(model.config.hidden_size)
    if not 0 <= spec.layer_idx < n_layers:
        raise ValueError(
            f"residual write layer_idx {spec.layer_idx} out of range (model has {n_layers} layers)"
        )
    if spec.vector.numel() != hidden_dim:
        raise ValueError(
            f"residual write vector has {spec.vector.numel()} elements, expected hidden_dim={hidden_dim}"
        )
    enabled_flag = {"enabled": True}
    stats: dict[str, Any] = {}
    handle = layers[spec.layer_idx].register_forward_pre_hook(
        _make_residual_write_pre_hook(spec, enabled_flag, stats), with_kwargs=True
    )
    logger.info(
        f"Residual write registered: layer {spec.layer_idx}, alpha={spec.alpha}, "
        f"pos=[{spec.start_pos},{spec.end_pos})"
    )
    return ResidualWriteHandle(spec=spec, _handle=handle, _enabled_flag=enabled_flag, stats=stats)


@dataclass
class LoadedModel:
    """Bundle of model, tokenizer, hooks, and their state."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    hook_state: HookState
    hook_handles: list[torch.utils.hooks.RemovableHook]
    config: ModelConfig

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.info("All hooks removed")

    def clear_hook_state(self) -> None:
        """Clear captured hook data without removing hooks."""
        self.hook_state.clear()

    def flush_hooks_to_cpu(self) -> None:
        """Batch-transfer hook tensors from GPU to CPU after generation."""
        self.hook_state.flush_to_cpu()

    def disable_hooks(self) -> None:
        """Temporarily disable hook capture (e.g. during calibration)."""
        self.hook_state.enabled = False

    def enable_hooks(self) -> None:
        """Re-enable hook capture."""
        self.hook_state.enabled = True

    def add_residual_write(self, spec: ResidualWriteSpec) -> ResidualWriteHandle:
        """Register an activation-WRITE injection (forward_pre_hook) on a decoder layer.

        Returns a handle with enable()/disable()/remove(). The caller owns the
        handle's lifecycle — writes are NOT tracked in hook_handles so read-hook
        teardown (remove_hooks) can never silently leave a perturbation armed,
        and vice versa. alpha=0 (or a disabled handle) must reproduce the
        unperturbed forward bit-for-bit (no-op path returns args unchanged).
        """
        return attach_residual_write(self.model, spec)


def load_model(
    config: ModelConfig | None = None,
    sampled_layers: list[int] | None = None,
    register_gate_hooks: bool = False,
    key_layers: list[int] | None = None,
    value_layers: list[int] | None = None,
    query_layers: list[int] | None = None,
    attn_output_layers: list[int] | None = None,
) -> LoadedModel:
    """Load model, tokenizer, and register hooks.

    Args:
        config: Model configuration. Defaults to ModelConfig().
        sampled_layers: Default layer set for gate_proj hooks and (unless
            overridden by key_layers) k_proj hooks. Defaults to
            {0, 7, 14, 18, 21, 24, 27}.
        register_gate_hooks: If True, also register hooks on gate_proj
            (SwiGLU MLP gate) at sampled layers. Captures pre-SiLU gate
            activations for gate sparsity/diversity features.
        key_layers: Layers for k_proj (pre-RoPE key) hooks. Defaults to
            sampled_layers (backward compatible). Pass all layers for the
            full v3 capture surface.
        value_layers: Layers for v_proj (value) hooks. Default None = no value
            capture. Pass all layers to bank the OV-circuit value surface.
        query_layers: Layers for q_proj (pre-RoPE query) hooks. Default None =
            no query capture. Pass sampled layers for offline QK-space geometry.
        attn_output_layers: Layers for o_proj (attention-output) hooks. Default
            None = no capture. Pass all layers for the vmb battery capture
            surface (attention-output cell; mlp_out derivable with hidden states).

    Returns:
        LoadedModel with everything wired up.
    """
    if config is None:
        config = ModelConfig()
    if sampled_layers is None:
        sampled_layers = [0, 7, 14, 18, 21, 24, 27]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    logger.info(f"Loading model: {config.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Resolve which layers get which hooks ──
    # Backward compatible: keys default to sampled_layers; values/queries are off
    # unless explicitly requested (the full v3 replay surface passes all/sampled).
    if key_layers is None:
        key_layers = list(sampled_layers)

    def _valid(layers: list[int]) -> list[int]:
        out: list[int] = []
        for i in layers:
            if 0 <= i < config.num_layers:
                out.append(i)
            else:
                logger.warning(
                    f"Skipping invalid layer index {i} (model has {config.num_layers} layers)"
                )
        return out

    key_layers = _valid(key_layers)
    value_layers = _valid(value_layers or [])
    query_layers = _valid(query_layers or [])
    attn_output_layers = _valid(attn_output_layers or [])

    hook_state = HookState()
    hook_handles: list[torch.utils.hooks.RemovableHook] = []

    # Pre-RoPE keys (k_proj)
    for layer_idx in key_layers:
        k_proj = decoder_layers(model)[layer_idx].self_attn.k_proj
        handle = k_proj.register_forward_hook(_make_k_proj_hook(
            layer_idx=layer_idx, hook_state=hook_state,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
        ))
        hook_handles.append(handle)

    # Values (v_proj) — OV-circuit surface
    for layer_idx in value_layers:
        v_proj = decoder_layers(model)[layer_idx].self_attn.v_proj
        handle = v_proj.register_forward_hook(_make_v_proj_hook(
            layer_idx=layer_idx, hook_state=hook_state,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
        ))
        hook_handles.append(handle)

    # Pre-RoPE queries (q_proj) — for offline QK-space geometry
    for layer_idx in query_layers:
        q_proj = decoder_layers(model)[layer_idx].self_attn.q_proj
        handle = q_proj.register_forward_hook(_make_q_proj_hook(
            layer_idx=layer_idx, hook_state=hook_state,
            num_attention_heads=config.num_attention_heads, head_dim=config.head_dim,
        ))
        hook_handles.append(handle)

    # Attention output (o_proj) — attention-output surface (vmb Stage A)
    for layer_idx in attn_output_layers:
        o_proj = decoder_layers(model)[layer_idx].self_attn.o_proj
        handle = o_proj.register_forward_hook(_make_o_proj_hook(
            layer_idx=layer_idx, hook_state=hook_state,
        ))
        hook_handles.append(handle)

    # Optionally register gate_proj hooks for SwiGLU gate features (sampled layers)
    gate_hook_count = 0
    if register_gate_hooks:
        for layer_idx in _valid(list(sampled_layers)):
            gate_proj = decoder_layers(model)[layer_idx].mlp.gate_proj
            handle = gate_proj.register_forward_hook(_make_gate_proj_hook(
                layer_idx=layer_idx, hook_state=hook_state,
            ))
            hook_handles.append(handle)
            gate_hook_count += 1

    logger.info(
        f"Model loaded. Hooks — k_proj×{len(key_layers)}, v_proj×{len(value_layers)}, "
        f"q_proj×{len(query_layers)}, o_proj×{len(attn_output_layers)}, "
        f"gate_proj×{gate_hook_count}"
    )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        hook_state=hook_state,
        hook_handles=hook_handles,
        config=config,
    )
