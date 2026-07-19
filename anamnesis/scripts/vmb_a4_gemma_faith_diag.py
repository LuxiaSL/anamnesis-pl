"""Gemma-A4 faithfulness DIAGNOSTIC — isolate the 41.55 divergence (2026-07-18).

The smoke's faithfulness leg diverges totally (max|Δlogit|=41.55, argmax 0.0) on Gemma3 but is
bitwise on Llama/Qwen/OLMo. kv-rotation's reconstruction is identical to ours and was only
validated on Trinity (afmoe), never Gemma3 — so the cause is unknown. This 3-way isolates it in
ONE model load:

  A. native            : model(context+cont, use_cache=False)                  [the reference]
  B. native-cache-reuse: prefill(context) -> reuse the NATIVE past_key_values -> forward(cont)
                         If B == A: forward-path + cache-passing are fine -> bug is our RECONSTRUCTION.
                         If B != A: the forward-path itself (wrapper/mask/position) is the bug.
  C. reconstructed     : prefill -> from_hf_cache -> to_hf_dynamic_cache -> forward(cont)  [smoke path]
  D. text-submodule    : same as C but forward through the inner text decoder (get_decoder /
                         .language_model / .model.language_model) instead of the multimodal wrapper.

Also reports the native cache TYPE (HybridCache vs DynamicCache) and per-path max|Δlogit| + argmax.
Read-only w.r.t. the battery; writes one JSON. GPU (1), eager.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--floor-run-dir", type=Path, required=True)
    ap.add_argument("--gen-id", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rep = {"model_path": args.model_path, "paths": {}}
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager")
    model.eval()
    dev = next(model.parameters()).device
    rep["model_class"] = type(model).__name__

    # locate the inner text decoder (multimodal wrappers nest it)
    decoder = None
    for accessor in ("get_decoder",):
        try:
            d = getattr(model, accessor)()
            if d is not None:
                decoder = d; rep["decoder_via"] = accessor; break
        except Exception:  # noqa: BLE001
            pass
    if decoder is None:
        for attr in ("language_model", "model"):
            d = getattr(model, attr, None)
            if d is not None and hasattr(d, "layers"):
                decoder = d; rep["decoder_via"] = attr; break
            d2 = getattr(d, "language_model", None) if d is not None else None
            if d2 is not None:
                decoder = d2; rep["decoder_via"] = f"{attr}.language_model"; break
    rep["decoder_type"] = type(decoder).__name__ if decoder is not None else None

    # context from a floor gen
    man = json.loads((args.floor_run_dir / "replay_manifest.json").read_text())["entries"]
    e = man[str(args.gen_id)]
    ids = list(e["input_ids"]); P = int(e["prompt_length"])
    C = P + (len(ids) - P) // 2
    context_ids, cont_ids = ids[:C], ids[C:C + args.k]
    k = len(cont_ids)
    rep["C"] = C, "k", k

    @torch.no_grad()
    def native():
        full = torch.tensor([context_ids + cont_ids], device=dev)
        return model(full, use_cache=False, return_dict=True).logits[0, C - 1: C - 1 + k].float()

    @torch.no_grad()
    def prefill_native_cache():
        out = model(torch.tensor([context_ids], device=dev), use_cache=True, return_dict=True)
        return out.past_key_values

    @torch.no_grad()
    def fwd_cached(cache, use_decoder=False):
        m = decoder if (use_decoder and decoder is not None) else model
        pos = torch.arange(C, C + k, device=dev).unsqueeze(0)
        out = m(torch.tensor([cont_ids], device=dev), past_key_values=cache,
                position_ids=pos, cache_position=torch.arange(C, C + k, device=dev),
                use_cache=True, return_dict=True)
        lg = out.logits if hasattr(out, "logits") and out.logits is not None else None
        if lg is None:  # a bare text decoder returns last_hidden_state -> apply lm_head
            h = out.last_hidden_state
            lg = model.lm_head(h) if hasattr(model, "lm_head") else model.get_output_embeddings()(h)
        return lg[0].float()

    ref = native()

    def cmp(name, logits):
        d = float((ref - logits).abs().max().item())
        a = float((ref.argmax(-1) == logits.argmax(-1)).float().mean().item())
        rep["paths"][name] = {"max_abs_logit_diff": d, "argmax_agreement": a, "match": d < 1e-2}

    base_cache = prefill_native_cache()
    rep["native_cache_type"] = type(base_cache).__name__

    # B: reuse native cache (reconstruction bypassed)
    try:
        cmp("B_native_cache_reuse", fwd_cached(prefill_native_cache()))
    except Exception as exc:  # noqa: BLE001
        rep["paths"]["B_native_cache_reuse"] = {"error": repr(exc)}

    # C: reconstructed cache (the smoke path) via our from_hf_cache/to_hf_dynamic_cache
    try:
        from anamnesis.extraction.cache_surgery import from_hf_cache, to_hf_dynamic_cache
        snap = from_hf_cache(prefill_native_cache(), positions=torch.arange(C, device=dev))
        cmp("C_reconstructed_wrapper", fwd_cached(to_hf_dynamic_cache(snap)))
    except Exception as exc:  # noqa: BLE001
        rep["paths"]["C_reconstructed_wrapper"] = {"error": repr(exc)}

    # D: reconstructed cache forwarded through the inner text decoder
    if decoder is not None:
        try:
            from anamnesis.extraction.cache_surgery import from_hf_cache, to_hf_dynamic_cache
            snap = from_hf_cache(prefill_native_cache(), positions=torch.arange(C, device=dev))
            cmp("D_reconstructed_decoder", fwd_cached(to_hf_dynamic_cache(snap), use_decoder=True))
        except Exception as exc:  # noqa: BLE001
            rep["paths"]["D_reconstructed_decoder"] = {"error": repr(exc)}

    # E: native cache forwarded through the inner text decoder
    if decoder is not None:
        try:
            cmp("E_native_cache_decoder", fwd_cached(prefill_native_cache(), use_decoder=True))
        except Exception as exc:  # noqa: BLE001
            rep["paths"]["E_native_cache_decoder"] = {"error": repr(exc)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
