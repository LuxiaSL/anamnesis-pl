"""Model onboarding load-validate smoke (vmb Stage-0 step 3; --model parameterized).

The replication-lane first-contact check for a NEW model preset, run BEFORE any
calibration/floor GPU spend. Validates the whole capture path on one GPU:
  [1] eager load via AutoModelForCausalLM (handles multimodal wrappers, e.g.
      Gemma3ForConditionalGeneration, whose text decoder nests under language_model)
  [2] decoder_layers() resolves the expected count; hook target modules present
  [3] model.generate(output_attentions=True) RETURNS attention weights with the
      preset's query-head count (the eager check flash/sdpa silently fail)
      + hidden_states == n_layers+1 + k_proj hooks fire
  [4] run_single_generation() builds a finite feature vector (baseline extractor;
      the full battery vector is replay-side)

First cleared on M5 Gemma-3-27B (2026-07-13, Opus session-1); reused for M6.

    CUDA_VISIBLE_DEVICES=0 HF_HOME=... HF_HUB_OFFLINE=1 \
        python -m anamnesis.scripts.vmb_onboard_validate --model gemma3-27b \
            --model-path google/gemma-3-27b-it
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from anamnesis.config import (MODEL_PRESETS, ModelConfig, ExperimentConfig,
                              ExtractionConfig, GenerationConfig, GenerationSpec)
from anamnesis.extraction.generation_runner import run_single_generation
from anamnesis.extraction.model_loader import decoder_layers, load_model

PROMPT = "Explain how a rainbow forms."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", default=None,
                    help="Override model_id (e.g. a local snapshot path)")
    args = ap.parse_args()
    p = MODEL_PRESETS[args.model]
    model_id = args.model_path or p.model_id

    mc = ModelConfig(model_id=model_id, torch_dtype=p.torch_dtype,
                     attn_implementation="eager", device_map="auto",
                     num_layers=p.num_layers, hidden_dim=p.hidden_dim,
                     num_attention_heads=p.num_attention_heads,
                     num_kv_heads=p.num_kv_heads, head_dim=p.head_dim)
    print(f"[1] loading {model_id} eager/{p.torch_dtype} ...", flush=True)
    loaded = load_model(mc, sampled_layers=p.sampled_layers, register_gate_hooks=True,
                        key_layers=p.sampled_layers, value_layers=p.sampled_layers,
                        query_layers=p.sampled_layers, attn_output_layers=p.sampled_layers)
    print("    loaded:", type(loaded.model).__name__)

    dl = decoder_layers(loaded.model)
    print(f"[2] decoder_layers -> {len(dl)} (expect {p.num_layers}); "
          f"type {type(dl[0]).__name__}")
    assert len(dl) == p.num_layers, f"LAYER COUNT {len(dl)} != {p.num_layers}"
    probe = sorted({p.sampled_layers[0], p.sampled_layers[len(p.sampled_layers) // 2],
                    p.sampled_layers[-1]})
    is_dsv2 = "deepseek_v2" in str(getattr(loaded.model.config, "model_type", ""))
    first_k = int(getattr(loaded.model.config, "first_k_dense_replace", 0))
    for li in probe:
        sa = dl[li].self_attn
        if is_dsv2:  # MLA + MoE: no k_proj/v_proj; MoE layers have gate/experts/shared_experts
            assert all(hasattr(sa, x) for x in ("kv_a_proj_with_mqa", "kv_b_proj", "q_proj", "o_proj")), f"L{li} MLA attn"
            if li < first_k:
                assert hasattr(dl[li].mlp, "gate_proj"), f"L{li} dense mlp gate_proj"
            else:
                assert all(hasattr(dl[li].mlp, x) for x in ("gate", "experts", "shared_experts")), f"L{li} MoE mlp"
                assert hasattr(dl[li].mlp.shared_experts, "gate_proj"), f"L{li} shared_experts gate_proj"
        else:
            assert all(hasattr(sa, x) for x in ("k_proj", "q_proj", "o_proj")), f"L{li} attn"
            assert hasattr(dl[li].mlp, "gate_proj"), f"L{li} mlp gate_proj"
    print(f"    hook target modules present on layers {probe} (dsv2/MLA+MoE={is_dsv2})")

    tok = loaded.tokenizer
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to(loaded.model.device)
    loaded.clear_hook_state()
    loaded.enable_hooks()
    with torch.no_grad():
        out = loaded.model.generate(
            ids, max_new_tokens=8, do_sample=False, eos_token_id=p.eos_token_ids,
            pad_token_id=tok.pad_token_id or p.eos_token_ids[0],
            output_attentions=True, output_hidden_states=True, output_logits=True,
            return_dict_in_generate=True)
    attn = out.attentions
    assert attn is not None and len(attn) > 1, "NO ATTENTION WEIGHTS (eager broken)"
    a0 = attn[1][0]
    print(f"[3] attn step1/layer0 shape {tuple(a0.shape)} "
          f"(expect [1, {p.num_attention_heads} q-heads, q, k])")
    assert a0.shape[1] == p.num_attention_heads, \
        f"HEAD COUNT {a0.shape[1]} != {p.num_attention_heads}"
    assert len(out.hidden_states[0]) == p.num_layers + 1, \
        f"HS layers {len(out.hidden_states[0])} != {p.num_layers + 1}"
    loaded.flush_hooks_to_cpu()
    print(f"    hidden_states={len(out.hidden_states[0])} layers; hooks fired ok")

    if is_dsv2:  # ── MoE capture smoke: router pre-hook fired + c_KV + full xrt path (HOOK-AUDIT-PLAN §6/§7) ──
        moe_layers = [l for l in p.sampled_layers if l >= first_k]
        rd = loaded.hook_state.router_dist
        fired = [l for l in moe_layers if rd.get(l)]
        assert fired, "router pre-hook fired on NO MoE layer"
        width = rd[fired[0]][-1].shape[-1]
        assert width == loaded.model.config.n_routed_experts, \
            f"router dist width {width} != {loaded.model.config.n_routed_experts}"
        assert loaded.hook_state.pre_rope_keys.get(fired[0]), "c_KV (kv_a_proj_with_mqa) hook did not fire"
        print(f"[3b] router pre-hook fired {len(fired)}/{len(moe_layers)} MoE layers; "
              f"dist width {width}; c_KV captured")
        from anamnesis.extraction.generation_runner import _router_fields_from_hooks
        from anamnesis.extraction.feature_families.expert_routing import extract_expert_routing_features
        from anamnesis.extraction.state_extractor import RawGenerationData
        from anamnesis.analysis.feature_map import FeatureMap
        r_dist, r_norms = _router_fields_from_hooks(loaded.hook_state, p.sampled_layers)
        rd_data = RawGenerationData(
            hidden_states=[np.zeros((p.num_layers + 1, p.hidden_dim), dtype=np.float32)],
            attentions=[], logits=[], chosen_token_ids=np.zeros(1), pre_rope_keys={},
            prompt_length=1, router_dist=r_dist, router_branch_norms=r_norms)
        xrt = extract_expert_routing_features(
            rd_data, sampled_layers=moe_layers, top_k=loaded.model.config.num_experts_per_tok)
        fm = FeatureMap(xrt.feature_names, p.num_layers)
        assert len(xrt.features) == 10 * len(moe_layers) and len(fm.unclassified()) == 0, \
            f"xrt smoke: n={len(xrt.features)} (expect {10 * len(moe_layers)}), unclassified={len(fm.unclassified())}"
        assert bool(np.isfinite(xrt.features).all()), "xrt features non-finite"
        print(f"[3c] xrt features: {len(xrt.features)} ({len(moe_layers)} MoE layers x10); "
              f"feature_map unclassified=0; finite=True")

    ec = ExperimentConfig(
        model=mc,
        generation=GenerationConfig(max_new_tokens=12, temperature=p.temperature,
                                    top_p=0.95, eos_token_ids=p.eos_token_ids, do_sample=True),
        extraction=ExtractionConfig(sampled_layers=p.sampled_layers))
    spec = GenerationSpec(generation_id=0, prompt_set="onboard", topic="rainbow",
                          topic_idx=0, mode="", mode_idx=0, system_prompt="",
                          user_prompt=PROMPT, seed=1234, repetition=0)
    res, meta = run_single_generation(loaded, spec, ec, positional_means=None)
    print(f"[4] extraction OK: n_features={meta['num_features']}, "
          f"gen_tokens={meta['num_generated_tokens']}, "
          f"finite={bool(np.isfinite(res.features).all())}")
    print(f"    sample text: {meta['generated_text'][:120]!r}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
