"""Full real-model integration on a tiny random Llama, without downloading weights."""

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig
from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
from anamnesis.extraction.gpu_features import GpuFeatureLane
from anamnesis.extraction.gpu_schema import resolve_gpu_schema
from anamnesis.extraction.model_loader import (
    HookState,
    LoadedModel,
    _make_k_proj_hook,
    _make_v_proj_hook,
    _make_q_proj_hook,
    _make_gate_proj_hook,
)
from anamnesis.extraction.replay_cached import replay_extract_cached
from anamnesis.extraction.replay_extract import replay_extract


def tiny_loaded():
    torch.manual_seed(751)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=80,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    state = HookState()
    handles = []
    for layer, module in enumerate(model.model.layers):
        for projection, fn in (
            ("k_proj", _make_k_proj_hook),
            ("v_proj", _make_v_proj_hook),
        ):
            handles.append(
                getattr(module.self_attn, projection).register_forward_hook(
                    fn(layer, state, 2, 8)
                )
            )
        handles.append(
            module.self_attn.q_proj.register_forward_hook(
                _make_q_proj_hook(layer, state, 4, 8)
            )
        )
        handles.append(
            module.mlp.gate_proj.register_forward_hook(
                _make_gate_proj_hook(layer, state)
            )
        )
    return LoadedModel(model, None, state, handles, ModelConfig())


@pytest.mark.parametrize("continuation", [2, 3, 4, 17, 65])
@pytest.mark.parametrize("replay_path", ["cached", "full"])
@pytest.mark.parametrize("per_layer_pca", [False, True])
def test_full_lane_matches_reference_and_repeats_with_fresh_cache(
    continuation, replay_path, per_layer_pca
):
    loaded = tiny_loaded()
    prefix = 11
    rng = np.random.default_rng(492)
    tokens = rng.integers(0, 64, size=prefix + continuation).tolist()
    pm = rng.normal(0, 0.01, size=(4, prefix + continuation, 32)).astype(np.float32)
    components = rng.normal(size=(5, 32)).astype(np.float32)
    mean = rng.normal(0, 0.01, size=32).astype(np.float32)
    if per_layer_pca:
        components = {0: components, 1: components * 0.3}
        mean = {0: mean, 1: mean + 0.1}
    extraction = ExtractionConfig(
        sampled_layers=[0, 1, 2], pca_layers=[0, 1], pca_components=5
    )
    families = FeaturePipelineConfig(
        include_baseline_tiers=True,
        enable_residual_trajectory=True,
        trajectory_layers=[1],
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_per_head=True,
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
    )

    def reference_raw():
        if replay_path == "full":
            return replay_extract(loaded, tokens, prefix, pm)
        loaded.disable_hooks()
        with torch.no_grad():
            pre = loaded.model(
                torch.tensor([tokens[:prefix]]), use_cache=True, return_dict=True
            )
        return replay_extract_cached(
            loaded, pre.past_key_values, tokens[prefix:], prefix, pm
        )

    raw = reference_raw()
    ref = compute_features_v2_from_data(raw, extraction, families, components, mean)
    schema = resolve_gpu_schema(3, continuation - 1, extraction, families, components)
    assert schema.feature_names == tuple(ref.feature_names)
    assert schema.family_slices == ref.tier_slices
    original_hooks = [
        tuple(module.self_attn._forward_hooks) for module in loaded.model.model.layers
    ]
    lane = GpuFeatureLane(
        extraction,
        families,
        ref.feature_names,
        pm,
        components,
        mean,
        device="cpu",
        calibration_sha256="a" * 64,
        replay_path=replay_path,
    )
    one = lane.replay_span(loaded, tokens, prefix, len(tokens))
    two = lane.replay_span(loaded, tokens, prefix, len(tokens))
    cached_lane = GpuFeatureLane(
        extraction,
        families,
        ref.feature_names,
        pm,
        components,
        mean,
        device="cpu",
        calibration_sha256="a" * 64,
        replay_path=replay_path,
        positional_cache_range=(prefix, len(tokens)),
    )
    cached = cached_lane.replay_span(loaded, tokens, prefix, len(tokens))
    assert cached.features.tobytes() == one.features.tobytes()
    assert cached.mean_logprob == one.mean_logprob
    assert cached_lane.lane_id != lane.lane_id
    assert cached.metadata["positional_calibration_h2d_bytes"] == 0
    assert (
        cached.metadata["positional_cache_construction_bytes"] == pm[1:, prefix:].nbytes
    )
    assert one.features.tobytes() == two.features.tobytes()
    assert one.metadata["replay_id"] != two.metadata["replay_id"]
    assert one.metadata["input_tokens_sha256"] == two.metadata["input_tokens_sha256"]
    assert one.metadata["attention_layers_reduced"] == 3
    assert not one.metadata["certified"]
    assert one.metadata["lane_id"] == lane.lane_id
    assert one.metadata["replay_path"] == replay_path
    assert tuple(ref.feature_names) == one.feature_names
    failures = []
    for i, name in enumerate(ref.feature_names):
        if not np.isclose(one.features[i], ref.features[i], atol=5e-6, rtol=3e-5):
            failures.append((name, float(ref.features[i]), float(one.features[i])))
    assert not failures, failures
    np.testing.assert_array_equal(one.knnlm_baseline, ref.knnlm_baseline)
    # Hook teardown must also let the existing CPU path run afterward.
    assert not loaded.hook_state.pre_rope_keys
    assert [
        tuple(module.self_attn._forward_hooks) for module in loaded.model.model.layers
    ] == original_hooks
    raw_after = reference_raw()
    ref_after = compute_features_v2_from_data(
        raw_after, extraction, families, components, mean
    )
    assert ref_after.features.tobytes() == ref.features.tobytes()


def test_replay_path_is_part_of_lane_identity():
    families = FeaturePipelineConfig(
        include_baseline_tiers=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_per_head=True,
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
    )
    args = (
        ExtractionConfig(),
        families,
        [],
        np.zeros((4, 10, 32), np.float32),
        np.zeros((5, 32), np.float32),
        np.zeros(32, np.float32),
    )
    cached = GpuFeatureLane(
        *args, device="cpu", calibration_sha256="a" * 64, replay_path="cached"
    )
    full = GpuFeatureLane(
        *args, device="cpu", calibration_sha256="a" * 64, replay_path="full"
    )
    assert cached.lane_id != full.lane_id


def test_rejects_non_eager_before_capturing():
    loaded = tiny_loaded()
    families = FeaturePipelineConfig(
        include_baseline_tiers=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_per_head=True,
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
    )
    lane = GpuFeatureLane(
        ExtractionConfig(),
        families,
        [],
        np.zeros((4, 10, 32), np.float32),
        np.zeros((5, 32), np.float32),
        np.zeros(32, np.float32),
        device="cpu",
        calibration_sha256="a" * 64,
    )
    loaded.model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        lane.replay_span(loaded, [1, 2, 3, 4], 1, 4)
