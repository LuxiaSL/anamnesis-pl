"""Batch feature parity on real tiny Llama; not real-GPU certification."""

import numpy as np
import pytest
import torch

from test_gpu_replay import tiny_loaded
from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.batch_layout import ReplaySpan
from anamnesis.extraction.gpu_features import GpuFeatureLane
from anamnesis.extraction.gpu_schema import resolve_gpu_schema
from anamnesis.extraction.replay_cached import replay_extract_cached
from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data


@pytest.mark.parametrize(
    "prefixes,continuations,prefill_policy",
    [
        ((7, 7), (17, 17), "batched"),
        ((3, 11), (17, 13), "batched"),
        ((11, 3), (13, 17), "batched"),
        ((7, 7), (17, 13), "independent"),
    ],
)
def test_batch_feature_parity_repeat_and_cleanup(
    prefixes, continuations, prefill_policy, monkeypatch
):
    loaded = tiny_loaded()
    rng = np.random.default_rng(341)
    spans = tuple(
        ReplaySpan(tuple(rng.integers(0, 64, size=p + n).tolist()), p, p + n)
        for p, n in zip(prefixes, continuations, strict=True)
    )
    pm = rng.normal(0, 0.01, size=(4, 64, 32)).astype(np.float32)
    components = rng.normal(size=(5, 32)).astype(np.float32)
    mean = np.zeros(32, dtype=np.float32)
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
    schema = resolve_gpu_schema(3, 16, extraction, families, components)
    lane = GpuFeatureLane(
        extraction,
        families,
        list(schema.feature_names),
        pm,
        components,
        mean,
        device="cpu",
        calibration_sha256="a" * 64,
    )
    # Transformers lazily installs its own persistent output-capture hooks.
    # Establish that baseline before checking our temporary hook teardown.
    loaded.disable_hooks()
    with torch.no_grad():
        loaded.model(
            torch.tensor([spans[0].tokens]),
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
        )
    hooks = [tuple(m.self_attn._forward_hooks) for m in loaded.model.model.layers]
    forwards = []
    counter = loaded.model.register_forward_pre_hook(
        lambda _m, args: forwards.append(tuple(args[0].shape))
    )
    first = lane.replay_batch(loaded, spans, prefill_policy=prefill_policy)
    counter.remove()
    if prefill_policy == "batched":
        assert len(forwards) == 2 and all(shape[0] == 2 for shape in forwards)
    else:
        assert [shape[0] for shape in forwards] == [1, 1, 2]
    second = lane.replay_batch(loaded, spans, prefill_policy=prefill_policy)
    assert first[0].metadata["lane_id"] != lane.lane_id
    assert first[0].metadata["lane_id"] == first[1].metadata["lane_id"]
    assert (
        first[0].metadata["batch_invocation_id"]
        != second[0].metadata["batch_invocation_id"]
    )
    assert not loaded.hook_state.pre_rope_keys
    assert [
        tuple(m.self_attn._forward_hooks) for m in loaded.model.model.layers
    ] == hooks
    for i, span in enumerate(spans):
        assert first[i].features.tobytes() == second[i].features.tobytes()
        assert first[i].metadata["replay_id"] != second[i].metadata["replay_id"]
        loaded.disable_hooks()
        with torch.no_grad():
            prefix = loaded.model(
                torch.tensor([span.tokens[: span.start]]),
                use_cache=True,
                return_dict=True,
            )
        raw = replay_extract_cached(
            loaded,
            prefix.past_key_values,
            list(span.tokens[span.start : span.end]),
            span.start,
            pm,
        )
        reference = compute_features_v2_from_data(
            raw, extraction, families, components, mean
        )
        failures = [
            (name, float(actual), float(expected))
            for name, actual, expected in zip(
                schema.feature_names, first[i].features, reference.features, strict=True
            )
            if not np.isclose(actual, expected, atol=5e-6, rtol=3e-5)
        ]
        assert not failures, failures
        np.testing.assert_allclose(
            first[i].knnlm_baseline, reference.knnlm_baseline, atol=1e-6, rtol=1e-5
        )

    def fail(*args, **kwargs):
        raise RuntimeError("injected reduction failure")

    monkeypatch.setattr(lane, "_reduce_capture", fail)
    with pytest.raises(RuntimeError, match="injected"):
        lane.replay_batch(loaded, spans, prefill_policy=prefill_policy)
    assert not loaded.hook_state.pre_rope_keys
    assert [
        tuple(m.self_attn._forward_hooks) for m in loaded.model.model.layers
    ] == hooks
