"""KV-cache surgery for ARM A4 (state-surgery): eviction / re-rotation / recompute.

Vendored from kv-rotation `src/kvrot/{rope,snapshot,eviction}.py` (exp11 machinery,
bit-exact rotation verified there) with battery-specific additions:
  - inv_freq built FROM THE MODEL CONFIG with a hard gate on unknown RoPE schemes
    (addendum 12h: the RoPE-config verification gate on M3/M4 conditional inclusion —
    fires at load time, never after a confusing result);
  - middle-region keep geometry (A4 block: sink + recent protected, contiguous
    middle block evicted).

RoPE background (kvrot docstring, condensed): HF caches store keys ALREADY rotated
at their original positions. Moving a token from position p to p' left-multiplies
its key by R(p'-p); exact iff inv_freq is position-independent (standard RoPE and
static llama3 rescaling; NOT dynamic-NTK — the homomorphism assert catches that).
Values are position-free and never touched.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ── RoPE (vendored, exact) ──────────────────────────────────────────────────────

def default_inv_freq(head_dim: int, theta: float, *, device=None) -> Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (theta ** (idx / head_dim))


def llama3_inv_freq(
    head_dim: int, theta: float, *, factor: float, low_freq_factor: float,
    high_freq_factor: float, original_max_position_embeddings: int, device=None,
) -> Tensor:
    """llama3 static frequency rescaling (mirrors transformers' _compute_llama3_parameters)."""
    inv_freq = default_inv_freq(head_dim, theta, device=device)
    old_ctx = original_max_position_embeddings
    low_freq_wavelen = old_ctx / low_freq_factor
    high_freq_wavelen = old_ctx / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    return torch.where(is_medium, smoothed, inv_freq_llama)


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _cos_sin(delta_pos: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
    freqs = torch.outer(delta_pos.to(torch.float32), inv_freq.to(torch.float32))
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def reindex_keys(key: Tensor, delta_pos: Tensor, inv_freq: Tensor) -> Tensor:
    """Rotate already-rotated keys by delta_pos (per token). [..., S, head_dim]."""
    if key.shape[-2] != delta_pos.shape[0]:
        raise ValueError(f"delta_pos length {delta_pos.shape[0]} != key seq dim {key.shape[-2]}")
    cos, sin = _cos_sin(delta_pos.to(key.device), inv_freq.to(key.device))
    while cos.dim() < key.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    kf = key.float()
    return (kf * cos + rotate_half(kf) * sin).to(key.dtype)


def assert_rotation_homomorphism(inv_freq: Tensor, *, atol: float = 1e-5) -> None:
    """R(a)·R(b) == R(a+b) — what makes re-rotation exact. Fails loudly on dynamic-NTK."""
    d = inv_freq.shape[0] * 2
    k = torch.randn(1, 1, 3, d, dtype=torch.float32)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    lhs = reindex_keys(reindex_keys(k, a, inv_freq), b, inv_freq)
    rhs = reindex_keys(k, a + b, inv_freq)
    if not torch.allclose(lhs, rhs, atol=atol):
        raise AssertionError(
            "RoPE frequencies are not a rotation homomorphism — re-rotation would be "
            f"inexact. Max err={(lhs - rhs).abs().max().item():.3e}"
        )


def inv_freq_from_config(config) -> Tensor:
    """Build inv_freq from an HF model config, with the 12h RoPE gate.

    Reads EACH model's rope_theta/rope_scaling (never assumes Llama's values).
    Supported: no scaling (Qwen2.5/OLMo-2 class) and static 'llama3' rescaling.
    Anything else raises — the gate fires BEFORE any surgery, not after a
    confusing result.
    """
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    theta = float(getattr(config, "rope_theta", 10000.0))
    scaling = getattr(config, "rope_scaling", None)
    if scaling is None:
        logger.info(f"RoPE gate: standard RoPE theta={theta} head_dim={head_dim}")
        inv = default_inv_freq(head_dim, theta)
    else:
        rope_type = scaling.get("rope_type", scaling.get("type"))
        if rope_type == "llama3":
            logger.info(f"RoPE gate: llama3 static rescaling theta={theta} {scaling}")
            inv = llama3_inv_freq(
                head_dim, theta,
                factor=float(scaling["factor"]),
                low_freq_factor=float(scaling["low_freq_factor"]),
                high_freq_factor=float(scaling["high_freq_factor"]),
                original_max_position_embeddings=int(scaling["original_max_position_embeddings"]),
            )
        elif rope_type in ("default", "linear") and float(scaling.get("factor", 1.0)) == 1.0:
            inv = default_inv_freq(head_dim, theta)
        else:
            raise ValueError(
                f"RoPE gate FAILED: unsupported rope_scaling {scaling!r} — re-rotation "
                "exactness not established for this scheme (12h gate; extend deliberately)"
            )
    assert_rotation_homomorphism(inv)
    return inv


# ── KVSnapshot + surgery (vendored) ─────────────────────────────────────────────

@dataclass
class KVSnapshot:
    """Engine-agnostic cache view; surgery is pure tensor math (CPU-testable)."""

    keys: list[Tensor]       # per layer [B, kv_heads, S, head_dim]
    values: list[Tensor]     # per layer [B, kv_heads, S, head_dim]
    positions: list[Tensor]  # per layer [S] absolute positions (long)

    def __post_init__(self) -> None:
        if not (len(self.values) == len(self.positions) == len(self.keys)):
            raise ValueError("KVSnapshot: per-layer list lengths must match")
        for li in range(len(self.keys)):
            if self.keys[li].shape[-2] != self.positions[li].shape[0]:
                raise ValueError(f"layer {li}: positions length != key seq len")

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def seq_len(self, layer: int = 0) -> int:
        return int(self.keys[layer].shape[-2])

    @property
    def next_position(self) -> int:
        return int(max(int(p.max().item()) for p in self.positions)) + 1


def evict(snapshot: KVSnapshot, keep_indices: Tensor) -> KVSnapshot:
    """Drop all but keep_indices along the sequence dim (positions carried unchanged)."""
    keys, values, positions = [], [], []
    for li in range(snapshot.num_layers):
        idx = keep_indices.to(snapshot.keys[li].device)
        keys.append(snapshot.keys[li].index_select(-2, idx))
        values.append(snapshot.values[li].index_select(-2, idx))
        positions.append(snapshot.positions[li].to(idx.device).index_select(0, idx))
    return KVSnapshot(keys=keys, values=values, positions=positions)


def reindex(snapshot: KVSnapshot, new_positions: Tensor, inv_freq: Tensor) -> KVSnapshot:
    """Re-rotate keys so each token sits at its new position (exact). Values untouched."""
    keys, positions = [], []
    for li in range(snapshot.num_layers):
        new_p = new_positions.to(snapshot.positions[li].device)
        if new_p.shape[0] != snapshot.positions[li].shape[0]:
            raise ValueError(f"layer {li}: new_positions length != current seq len")
        delta = (new_p - snapshot.positions[li]).to(torch.float32)
        keys.append(reindex_keys(snapshot.keys[li], delta, inv_freq))
        positions.append(new_p.to(snapshot.positions[li].dtype))
    return KVSnapshot(keys=keys, values=[v for v in snapshot.values], positions=positions)


def from_hf_cache(past_key_values, *, positions: Tensor) -> KVSnapshot:
    keys, values = _extract_kv(past_key_values)
    return KVSnapshot(
        keys=list(keys), values=list(values),
        positions=[positions.to(keys[i].device) for i in range(len(keys))],
    )


def _extract_kv(past_key_values):
    if hasattr(past_key_values, "layers"):  # transformers >= 5
        layers = [ly for ly in past_key_values.layers if getattr(ly, "keys", None) is not None]
        return [ly.keys for ly in layers], [ly.values for ly in layers]
    if hasattr(past_key_values, "key_cache"):  # transformers < 5
        return list(past_key_values.key_cache), list(past_key_values.value_cache)
    if isinstance(past_key_values, (list, tuple)):
        return [l[0] for l in past_key_values], [l[1] for l in past_key_values]
    raise TypeError(f"unsupported cache type: {type(past_key_values)!r}")


def to_hf_dynamic_cache(snapshot: KVSnapshot):
    from transformers import DynamicCache

    cache = DynamicCache()
    for li in range(snapshot.num_layers):
        cache.update(snapshot.keys[li], snapshot.values[li], li)
    return cache


# ── A4 battery eviction geometry ────────────────────────────────────────────────

def middle_region_keep(
    context_len: int, evict_frac: float, *, num_sinks: int = 4, recent_protect: int = 32,
) -> Tensor:
    """Keep-indices for the A4 cells: evict ONE contiguous middle block of
    round(evict_frac * context_len) tokens, centered in the evictable region
    [num_sinks, context_len - recent_protect); sinks + recent tail always kept.
    """
    n_evict = int(round(evict_frac * context_len))
    lo, hi = num_sinks, context_len - recent_protect
    if n_evict <= 0:
        return torch.arange(context_len, dtype=torch.long)
    if hi - lo < n_evict:
        raise ValueError(
            f"cannot evict {n_evict} of {context_len} tokens with sinks={num_sinks}, "
            f"recent={recent_protect} protected (evictable={hi - lo})"
        )
    center = (lo + hi) // 2
    start = max(lo, min(center - n_evict // 2, hi - n_evict))
    evicted = set(range(start, start + n_evict))
    keep = [i for i in range(context_len) if i not in evicted]
    return torch.tensor(keep, dtype=torch.long)
