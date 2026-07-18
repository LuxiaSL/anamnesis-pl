"""Offline regression: the full v2 feature pipeline runs on DeepSeek-V2-Lite (MLA+MoE) shapes.

Catches the class of dim-mismatch / None-surface crashes that the GPU onboard-smoke misses
(it exercises only the baseline extractor, not the v2 families). The original miss: the dense
L0 gate (intermediate_size 10944) vs MoE shared_experts gate (moe_intermediate*n_shared 2816)
broke gate_features' cross-layer cosine — so M6 capture gates the MoE shared branch only.

Pure-CPU (synthetic RawGenerationData); no GPU/model. Run: `python -m pytest tests/test_dsv2_moe_pipeline.py`
or `python tests/test_dsv2_moe_pipeline.py`.
"""
from __future__ import annotations

import numpy as np

from anamnesis.analysis.feature_map import FeatureMap
from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, MODEL_PRESETS
from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
from anamnesis.extraction.state_extractor import RawGenerationData


def _synthetic_dsv2_raw(T: int = 24) -> tuple[RawGenerationData, list[int], list[int]]:
    rng = np.random.default_rng(0)
    p = MODEL_PRESETS["dsv2-lite"]
    sl = p.sampled_layers                      # [0,5,11,15,18,22,26]
    moe = [l for l in sl if l != 0]            # gate/router on MoE layers only
    nl, hidden, heads = 27, 2048, 16
    gate_dim, n_experts, kv_lora = 1408 * 2, 64, 512   # shared gate width; routed experts; c_KV latent

    hs = [rng.standard_normal((nl + 1, hidden)).astype(np.float32) for _ in range(T)]
    att = [rng.random((nl, heads, 6 + t)).astype(np.float32) for t in range(T)]
    keys = {l: [rng.standard_normal((1, kv_lora)).astype(np.float32) for _ in range(T)] for l in sl}
    gates = {l: [rng.standard_normal(gate_dim).astype(np.float32) for _ in range(T)] for l in moe}

    def _dist():
        x = rng.normal(0, 2, n_experts)
        e = np.exp(x - x.max())
        return (e / e.sum()).astype(np.float32)

    rdist = {l: [_dist() for _ in range(T)] for l in moe}
    rnorms = {l: [np.array([abs(rng.normal(1, .2)), abs(rng.normal(2, .3))], dtype=np.float32)
                  for _ in range(T)] for l in moe}
    data = RawGenerationData(
        hidden_states=hs, attentions=att,
        logits=[rng.standard_normal(2000).astype(np.float32) for _ in range(T)],
        chosen_token_ids=np.arange(T, dtype=np.float32), pre_rope_keys=keys, prompt_length=4,
        gate_activations=gates, v_proj_values=None, queries=None,   # MLA: no v_proj/q_proj capture
        router_dist=rdist, router_branch_norms=rnorms)
    return data, sl, moe


def test_dsv2_full_v2_pipeline_runs_clean():
    data, sl, moe = _synthetic_dsv2_raw()
    p = MODEL_PRESETS["dsv2-lite"]
    ec = ExtractionConfig(sampled_layers=sl, pca_layers=p.pca_layers,
                          early_layer_cutoff=p.early_layer_cutoff,
                          late_layer_cutoff=p.late_layer_cutoff, enable_tier3=False)
    fc = FeaturePipelineConfig(
        include_baseline_tiers=True, enable_residual_trajectory=True, enable_attention_flow=True,
        enable_gate_features=True, enable_temporal_dynamics=False, enable_per_head=True,
        enable_stft=True, enable_contrastive_projection=False, enable_value_geometry=True,
        enable_qk_geometry=True, enable_kv_cka=True, enable_expert_routing=True,
        trajectory_layers=p.trajectory_layers, contrastive_layers=p.contrastive_layers)

    res = compute_features_v2_from_data(data, ec, fc, pca_components=None, pca_mean=None)

    assert len(res.features) > 0
    assert np.isfinite(res.features).all(), "non-finite features"
    # 60 xrt features = 10 per MoE layer
    xrt_n = sum(1 for n in res.feature_names if n.startswith("xrt_"))
    assert xrt_n == 10 * len(moe), f"xrt {xrt_n} != {10 * len(moe)}"
    # feature_map places everything (incl xrt)
    assert len(FeatureMap(res.feature_names, 27).unclassified()) == 0
    return res


if __name__ == "__main__":
    r = test_dsv2_full_v2_pipeline_runs_clean()
    print(f"PASS — {len(r.features)} features, "
          f"{sum(1 for n in r.feature_names if n.startswith('xrt_'))} xrt, all finite, 0 unclassified")
