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


def load_model(
    config: ModelConfig | None = None,
    sampled_layers: list[int] | None = None,
    register_gate_hooks: bool = False,
    key_layers: list[int] | None = None,
    value_layers: list[int] | None = None,
    query_layers: list[int] | None = None,
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

    hook_state = HookState()
    hook_handles: list[torch.utils.hooks.RemovableHook] = []

    # Pre-RoPE keys (k_proj)
    for layer_idx in key_layers:
        k_proj = model.model.layers[layer_idx].self_attn.k_proj
        handle = k_proj.register_forward_hook(_make_k_proj_hook(
            layer_idx=layer_idx, hook_state=hook_state,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
        ))
        hook_handles.append(handle)

    # Values (v_proj) — OV-circuit surface
    for layer_idx in value_layers:
        v_proj = model.model.layers[layer_idx].self_attn.v_proj
        handle = v_proj.register_forward_hook(_make_v_proj_hook(
            layer_idx=layer_idx, hook_state=hook_state,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
        ))
        hook_handles.append(handle)

    # Pre-RoPE queries (q_proj) — for offline QK-space geometry
    for layer_idx in query_layers:
        q_proj = model.model.layers[layer_idx].self_attn.q_proj
        handle = q_proj.register_forward_hook(_make_q_proj_hook(
            layer_idx=layer_idx, hook_state=hook_state,
            num_attention_heads=config.num_attention_heads, head_dim=config.head_dim,
        ))
        hook_handles.append(handle)

    # Optionally register gate_proj hooks for SwiGLU gate features (sampled layers)
    gate_hook_count = 0
    if register_gate_hooks:
        for layer_idx in _valid(list(sampled_layers)):
            gate_proj = model.model.layers[layer_idx].mlp.gate_proj
            handle = gate_proj.register_forward_hook(_make_gate_proj_hook(
                layer_idx=layer_idx, hook_state=hook_state,
            ))
            hook_handles.append(handle)
            gate_hook_count += 1

    logger.info(
        f"Model loaded. Hooks — k_proj×{len(key_layers)}, v_proj×{len(value_layers)}, "
        f"q_proj×{len(query_layers)}, gate_proj×{gate_hook_count}"
    )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        hook_state=hook_state,
        hook_handles=hook_handles,
        config=config,
    )
