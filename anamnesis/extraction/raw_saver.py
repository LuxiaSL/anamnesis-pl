"""Save and load raw per-token tensors to/from disk.

Enables GPU-free feature engineering iteration: extract once (GPU),
compute features many times (CPU) with different feature configs.

Saves RawGenerationData filtered to sampled layers in compressed npz format.
Load reconstructs a RawGenerationData compatible with state_extractor.

Disk format (per generation):
    raw_tensors/gen_NNN.npz:
        hidden_states:   [T, n_layers_saved, hidden_dim]      float16
        attentions:      [T, n_layers_attn, n_heads, max_seq]  float16 (padded)
        pre_rope_keys:   [T, n_kv_layers, n_kv_heads, head_dim] float16
        logits_values:   [T, top_k]                            float32
        logits_indices:  [T, top_k]                            int32
        chosen_ids:      [T]                                   int32
        actual_lengths:  [T]                                   int32  (for attention unpadding)
        prompt_length:   scalar                                int32
        saved_layers_hs: [n_layers_saved]                      int32  (which layers for hidden_states)
        saved_layers_attn: [n_layers_attn]                     int32  (which layers for attention)
        all_layers:      [total_layers+1]                      int32  (for index mapping)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import ExtractionConfig
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def save_raw_tensors(
    raw_data: RawGenerationData,
    gen_id: int,
    output_dir: Path,
    config: ExtractionConfig,
    prompt_length: int,
    top_k_logits: int = 50,
    hidden_dtype: str = "float16",
) -> Path:
    """Save RawGenerationData to compressed npz, filtered to sampled layers.

    Parameters
    ----------
    raw_data : RawGenerationData
        The in-memory per-token tensors from a generation run.
    gen_id : int
        Generation ID for filename.
    output_dir : Path
        Directory to save into (e.g., raw_tensors/).
    config : ExtractionConfig
        Extraction config with sampled_layers and pca_layers.
    prompt_length : int
        Number of prompt tokens (stored in npz for reference).
    top_k_logits : int
        How many top logits to save per timestep (default 50).
    hidden_dtype : str
        Dtype for hidden states and attention weights (default float16).

    Returns
    -------
    Path to the saved npz file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"gen_{gen_id:03d}.npz"

    T = len(raw_data.hidden_states)
    if T == 0:
        logger.warning(f"gen_{gen_id}: no generation steps, saving empty")
        np.savez_compressed(npz_path, T=np.array(0))
        return npz_path

    save_dtype = np.dtype(hidden_dtype)
    n_total_layers_plus_one = raw_data.hidden_states[0].shape[0]  # num_layers + 1 (incl embedding)

    # ── Determine which layers to save ──
    # Hidden states: sampled_layers ∪ pca_layers (for maximum flexibility)
    hs_layer_indices = sorted(set(config.sampled_layers) | set(config.pca_layers))
    # +1 offset: hidden_states index 0 = embedding, 1..N = transformer layers
    # sampled_layers/pca_layers refer to transformer layer indices
    # So to get transformer layer L, we index hidden_states[:, L+1, :]
    # But we also want the embedding (index 0) if layer 0 is in sampled_layers
    hs_array_indices = []
    for layer_idx in hs_layer_indices:
        arr_idx = layer_idx + 1  # transformer layer L is at array index L+1
        if arr_idx < n_total_layers_plus_one:
            hs_array_indices.append(arr_idx)
    # Also include embedding layer (index 0) if anyone might need it
    # Actually, let's just include it — it's one extra slice
    if 0 not in hs_array_indices:
        hs_array_indices = [0] + hs_array_indices
        hs_layer_indices = [-1] + hs_layer_indices  # -1 = embedding layer

    # Attention: sampled_layers only (these are what T2/T2.5 features use)
    attn_layer_indices = sorted(config.sampled_layers)
    # Attention array: index L directly (no +1 offset like hidden_states)

    # ── Stack hidden states ──
    # raw_data.hidden_states[t] has shape [num_layers+1, hidden_dim]
    # Select only the layers we want
    hs_stacked = np.stack([
        h[hs_array_indices, :] for h in raw_data.hidden_states
    ]).astype(save_dtype)  # [T, n_saved, hidden_dim]

    # ── Stack attentions (with padding) ──
    # raw_data.attentions[t] has shape [num_layers, num_heads, seq_len_at_step]
    # seq_len grows by 1 each step — need padding
    n_heads = raw_data.attentions[0].shape[1] if raw_data.attentions else 0
    actual_lengths = np.array([
        raw_data.attentions[t].shape[2] for t in range(T)
    ], dtype=np.int32)
    max_seq_len = int(actual_lengths.max()) if len(actual_lengths) > 0 else 0

    n_attn_layers = len(attn_layer_indices)
    attn_stacked = np.zeros(
        (T, n_attn_layers, n_heads, max_seq_len), dtype=save_dtype,
    )
    for t in range(T):
        attn_t = raw_data.attentions[t]  # [num_layers, num_heads, seq_len_t]
        seq_len_t = attn_t.shape[2]
        for i, layer_idx in enumerate(attn_layer_indices):
            if layer_idx < attn_t.shape[0]:
                attn_stacked[t, i, :, :seq_len_t] = attn_t[layer_idx, :, :seq_len_t].astype(save_dtype)

    # ── Stack pre-RoPE keys ──
    # raw_data.pre_rope_keys: dict[layer_idx → list of T arrays [n_kv_heads, head_dim]]
    kv_layer_indices = sorted(raw_data.pre_rope_keys.keys())
    if kv_layer_indices:
        n_kv_heads = raw_data.pre_rope_keys[kv_layer_indices[0]][0].shape[0]
        head_dim = raw_data.pre_rope_keys[kv_layer_indices[0]][0].shape[1]
        keys_stacked = np.zeros(
            (T, len(kv_layer_indices), n_kv_heads, head_dim), dtype=save_dtype,
        )
        for i, layer_idx in enumerate(kv_layer_indices):
            key_list = raw_data.pre_rope_keys[layer_idx]
            for t in range(min(T, len(key_list))):
                keys_stacked[t, i] = key_list[t].astype(save_dtype)
    else:
        keys_stacked = np.array([], dtype=save_dtype)

    # ── Extract top-k logits + pre-compute entropy ──
    # raw_data.logits[t] has shape [vocab_size]
    # Entropy needs full distribution — save it pre-computed since we discard most logits
    if raw_data.logits and len(raw_data.logits) > 0:
        k = min(top_k_logits, raw_data.logits[0].shape[0])
        logits_values = np.zeros((T, k), dtype=np.float32)
        logits_indices = np.zeros((T, k), dtype=np.int32)
        logits_entropy = np.zeros(T, dtype=np.float32)
        for t in range(min(T, len(raw_data.logits))):
            logit_vec = raw_data.logits[t]
            # Top-k extraction
            top_idx = np.argpartition(logit_vec, -k)[-k:]
            top_idx_sorted = top_idx[np.argsort(logit_vec[top_idx])[::-1]]
            logits_values[t] = logit_vec[top_idx_sorted]
            logits_indices[t] = top_idx_sorted
            # Pre-compute entropy from full distribution
            probs = np.exp(logit_vec - np.max(logit_vec)).astype(np.float64)
            probs /= probs.sum()
            probs = probs[probs > 0]
            logits_entropy[t] = float(-np.sum(probs * np.log(probs)))
    else:
        logits_values = np.array([], dtype=np.float32)
        logits_indices = np.array([], dtype=np.int32)
        logits_entropy = np.array([], dtype=np.float32)

    # ── Stack gate activations (SwiGLU gate_proj, pre-SiLU) ──
    gate_layer_indices: list[int] = []
    if raw_data.gate_activations:
        gate_layer_indices = sorted(raw_data.gate_activations.keys())
    if gate_layer_indices:
        intermediate_size = raw_data.gate_activations[gate_layer_indices[0]][0].shape[0]
        gates_stacked = np.zeros(
            (T, len(gate_layer_indices), intermediate_size), dtype=save_dtype,
        )
        for i, layer_idx in enumerate(gate_layer_indices):
            gate_list = raw_data.gate_activations[layer_idx]
            for t in range(min(T, len(gate_list))):
                gates_stacked[t, i] = gate_list[t].astype(save_dtype)
    else:
        gates_stacked = np.array([], dtype=save_dtype)

    # ── Chosen token IDs ──
    chosen_ids = raw_data.chosen_token_ids.astype(np.int32)

    # ── Save ──
    save_dict = {
        "hidden_states": hs_stacked,
        "attentions": attn_stacked,
        "pre_rope_keys": keys_stacked,
        "logits_values": logits_values,
        "logits_indices": logits_indices,
        "logits_entropy": logits_entropy,
        "chosen_ids": chosen_ids,
        "actual_lengths": actual_lengths,
        "prompt_length": np.array(prompt_length, dtype=np.int32),
        "saved_layers_hs": np.array(hs_layer_indices, dtype=np.int32),
        "saved_layers_attn": np.array(attn_layer_indices, dtype=np.int32),
        "kv_layer_indices": np.array(kv_layer_indices, dtype=np.int32),
        "all_layers_count": np.array(n_total_layers_plus_one, dtype=np.int32),
        "gate_activations": gates_stacked,
        "gate_layer_indices": np.array(gate_layer_indices, dtype=np.int32),
    }

    # Include positional means if available (needed for feature extraction)
    if raw_data.positional_means is not None:
        save_dict["positional_means"] = raw_data.positional_means.astype(save_dtype)

    np.savez_compressed(npz_path, **save_dict)

    size_mb = npz_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved raw tensors: {npz_path.name} ({size_mb:.1f} MB, T={T})")

    return npz_path


def load_raw_tensors(
    gen_id: int,
    raw_dir: Path,
) -> RawGenerationData:
    """Load saved raw tensors back into RawGenerationData.

    The returned object is compatible with state_extractor.extract_all_features(),
    with the caveat that only sampled layers are populated — features that require
    all layers (like per-layer norms in Tier 1) will only have data at saved layers.

    Parameters
    ----------
    gen_id : int
        Generation ID to load.
    raw_dir : Path
        Directory containing raw tensor npz files.

    Returns
    -------
    RawGenerationData reconstructed from saved tensors.
    """
    npz_path = raw_dir / f"gen_{gen_id:03d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Raw tensor file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    T_check = data.get("T")
    if T_check is not None and int(T_check) == 0:
        return RawGenerationData(
            hidden_states=[],
            attentions=[],
            logits=[],
            chosen_token_ids=np.array([], dtype=np.float32),
            pre_rope_keys={},
            prompt_length=0,
        )

    hs_stacked = data["hidden_states"].astype(np.float32)  # [T, n_saved, hidden_dim]
    attn_stacked = data["attentions"].astype(np.float32)    # [T, n_attn, n_heads, max_seq]
    keys_stacked = data["pre_rope_keys"].astype(np.float32) if data["pre_rope_keys"].size > 0 else None
    logits_values = data["logits_values"]                   # [T, k]
    logits_indices = data["logits_indices"]                  # [T, k]
    chosen_ids = data["chosen_ids"].astype(np.float32)      # [T]
    actual_lengths = data["actual_lengths"]                  # [T]
    prompt_length = int(data["prompt_length"])
    saved_layers_hs = data["saved_layers_hs"]               # layer indices
    saved_layers_attn = data["saved_layers_attn"]           # layer indices
    kv_layer_indices = data["kv_layer_indices"] if "kv_layer_indices" in data else np.array([], dtype=np.int32)
    n_total_layers = int(data["all_layers_count"])           # num_layers + 1

    T = hs_stacked.shape[0]
    hidden_dim = hs_stacked.shape[2] if hs_stacked.ndim == 3 else 0

    # ── Reconstruct hidden_states ──
    # Need to expand back to [num_layers+1, hidden_dim] per timestep
    # Fill unsaved layers with zeros (features won't use them)
    hidden_states: list[F32] = []
    for t in range(T):
        full_h = np.zeros((n_total_layers, hidden_dim), dtype=np.float32)
        for i, layer_label in enumerate(saved_layers_hs):
            layer_label = int(layer_label)
            if layer_label == -1:
                arr_idx = 0  # embedding
            else:
                arr_idx = layer_label + 1  # transformer layer offset
            if arr_idx < n_total_layers:
                full_h[arr_idx] = hs_stacked[t, i]
        hidden_states.append(full_h)

    # ── Reconstruct attentions ──
    # Expand to [num_layers_no_embed, num_heads, seq_len_t] per timestep
    n_layers_no_embed = n_total_layers - 1  # exclude embedding
    n_heads = attn_stacked.shape[2] if attn_stacked.ndim == 4 else 0
    attentions: list[F32] = []
    for t in range(T):
        seq_len_t = int(actual_lengths[t])
        full_a = np.zeros((n_layers_no_embed, n_heads, seq_len_t), dtype=np.float32)
        for i, layer_idx in enumerate(saved_layers_attn):
            layer_idx = int(layer_idx)
            if layer_idx < n_layers_no_embed:
                full_a[layer_idx, :, :seq_len_t] = attn_stacked[t, i, :, :seq_len_t]
        attentions.append(full_a)

    # ── Reconstruct logits ──
    # We only have top-k, not full vocab. Create a partial representation.
    # Pre-computed entropy is stored separately for accurate Tier 1 features.
    logits_entropy_saved = data.get("logits_entropy")  # [T] float32 — exact entropy from full dist
    logits: list[F32] = []
    if logits_values.size > 0:
        # Determine vocab size from max index
        vocab_size = int(logits_indices.max()) + 1 if logits_indices.size > 0 else 128256
        for t in range(T):
            # Create full logit vector with very negative values for unobserved positions
            full_logits = np.full(vocab_size, -1e9, dtype=np.float32)
            valid_mask = logits_indices[t] >= 0
            if np.any(valid_mask):
                full_logits[logits_indices[t][valid_mask]] = logits_values[t][valid_mask]
            logits.append(full_logits)
    # Note: logits_entropy_saved is available via data["logits_entropy"] for callers
    # that need exact entropy without recomputing from the partial logit vector.

    # ── Reconstruct pre-RoPE keys ──
    pre_rope_keys: dict[int, list[F32]] = {}
    if keys_stacked is not None and keys_stacked.size > 0:
        for i, layer_idx in enumerate(kv_layer_indices):
            layer_idx = int(layer_idx)
            key_list: list[F32] = []
            for t in range(T):
                key_list.append(keys_stacked[t, i])
            pre_rope_keys[layer_idx] = key_list

    # ── Reconstruct gate activations ──
    gate_activations: dict[int, list[F32]] | None = None
    gate_layer_indices_arr = data.get("gate_layer_indices")
    gates_stacked_arr = data.get("gate_activations")
    if (
        gate_layer_indices_arr is not None
        and gates_stacked_arr is not None
        and gates_stacked_arr.size > 0
        and gate_layer_indices_arr.size > 0
    ):
        gates_stacked_f32 = gates_stacked_arr.astype(np.float32)
        gate_activations = {}
        for i, layer_idx in enumerate(gate_layer_indices_arr):
            layer_idx = int(layer_idx)
            gate_list: list[F32] = []
            for t in range(T):
                gate_list.append(gates_stacked_f32[t, i])
            gate_activations[layer_idx] = gate_list

    # ── Positional means ──
    positional_means: F32 | None = None
    if "positional_means" in data:
        positional_means = data["positional_means"].astype(np.float32)

    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=chosen_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=prompt_length,
        positional_means=positional_means,
        gate_activations=gate_activations,
    )


def list_raw_tensor_ids(raw_dir: Path) -> list[int]:
    """List all generation IDs that have saved raw tensors."""
    ids: list[int] = []
    for npz_path in sorted(raw_dir.glob("gen_*.npz")):
        stem = npz_path.stem  # gen_NNN
        try:
            gen_id = int(stem.split("_")[1])
            ids.append(gen_id)
        except (IndexError, ValueError):
            continue
    return ids
