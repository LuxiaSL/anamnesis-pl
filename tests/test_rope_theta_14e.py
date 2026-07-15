"""14e regression test — inv_freq_from_config must read theta from the tf-5.x rope_scaling
dict when the top-level attribute is absent, and must NEVER silently default to 10000.

The bug (2026-07-15): transformers 5.3 moved `rope_theta` INSIDE `config.rope_scaling`; the
old `getattr(config, "rope_theta", 10000.0)` fired its default → every battery ROTATE row
rotated keys with theta-10000 frequencies instead of the Llama-3 500000. These tests exercise
the DEPLOYED config-reading path (not the math function with explicit args — the same-day check
that missed the bug tested the wrong thing).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from anamnesis.extraction.cache_surgery import inv_freq_from_config, llama3_inv_freq

HEAD_DIM = 128
LLAMA3 = dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0,
              high_freq_factor=4.0, original_max_position_embeddings=8192)


def _cfg(**kw):
    base = dict(head_dim=HEAD_DIM, hidden_size=4096, num_attention_heads=32)
    base.update(kw)
    return SimpleNamespace(**base)


def test_theta_read_from_rope_scaling_not_defaulted():
    """tf-5.3 layout: theta ONLY inside rope_scaling → the 500000 table, never the 10000 fallback."""
    cfg = _cfg(rope_scaling={**LLAMA3, "rope_theta": 500000.0})   # NO top-level rope_theta
    got = inv_freq_from_config(cfg)

    expect_500k = llama3_inv_freq(HEAD_DIM, 500000.0, factor=8.0, low_freq_factor=1.0,
                                  high_freq_factor=4.0, original_max_position_embeddings=8192)
    wrong_10k = llama3_inv_freq(HEAD_DIM, 10000.0, factor=8.0, low_freq_factor=1.0,
                                high_freq_factor=4.0, original_max_position_embeddings=8192)
    assert torch.allclose(got, expect_500k, rtol=1e-6), "did not build the 500000-based table"
    # the bug's signature: tail freqs ~40× too fast under theta=10000 — must NOT match
    assert not torch.allclose(got, wrong_10k, rtol=1e-3), "fell back to the theta=10000 table (the 14e bug)"


def test_top_level_theta_still_works():
    """Standard RoPE (Qwen/OLMo class): theta top-level, no scaling → unchanged behavior."""
    cfg = _cfg(rope_theta=1_000_000.0, rope_scaling=None)
    got = inv_freq_from_config(cfg)
    from anamnesis.extraction.cache_surgery import default_inv_freq
    assert torch.allclose(got, default_inv_freq(HEAD_DIM, 1_000_000.0), rtol=1e-6)


def test_raises_when_theta_findable_nowhere():
    """No top-level theta AND none inside the scaling dict → RAISE, never default."""
    cfg = _cfg(rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                             "high_freq_factor": 4.0, "original_max_position_embeddings": 8192})
    with pytest.raises(ValueError, match="rope_theta not found"):
        inv_freq_from_config(cfg)


def test_rope_parameters_layout_also_supported():
    """Some 5.x configs expose the dict as `rope_parameters` — read theta there too."""
    cfg = _cfg(rope_parameters={**LLAMA3, "rope_theta": 500000.0})
    got = inv_freq_from_config(cfg)
    expect_500k = llama3_inv_freq(HEAD_DIM, 500000.0, factor=8.0, low_freq_factor=1.0,
                                  high_freq_factor=4.0, original_max_position_embeddings=8192)
    assert torch.allclose(got, expect_500k, rtol=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
