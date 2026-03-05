#!/usr/bin/env python3
"""A/B test: old vs new post-processing of HF generate outputs.

Runs generate() once with all output flags, then times two approaches
to converting the outputs to numpy:
  OLD: per-tensor .cpu().float().numpy() in nested loops (current code)
  NEW: bulk GPU stack + single .cpu() transfer + numpy reshape

Usage:
    python -m anamnesis.scripts.test_fast_postprocess
    python -m anamnesis.scripts.test_fast_postprocess --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROMPT = "Explain how photosynthesis works in plants."


def old_postprocess(outputs, prompt_length: int) -> dict:
    """Original approach: per-tensor .cpu().float().numpy() in nested loops."""
    t0 = time.perf_counter()

    # Hidden states
    hidden_states_list = []
    for t in range(1, len(outputs.hidden_states)):
        layers = []
        for l_tensor in outputs.hidden_states[t]:
            layers.append(l_tensor[0, -1].cpu().float().numpy())
        hidden_states_list.append(np.stack(layers))

    t_hs = time.perf_counter()

    # Attentions
    attentions_list = []
    for t in range(1, len(outputs.attentions)):
        layers = []
        for l_tensor in outputs.attentions[t]:
            layers.append(l_tensor[0, :, -1, :].cpu().float().numpy())
        attentions_list.append(np.stack(layers))

    t_attn = time.perf_counter()

    # Logits
    logits_list = []
    for t in range(1, len(outputs.logits)):
        logits_list.append(outputs.logits[t][0].cpu().float().numpy())

    t_logits = time.perf_counter()

    # Chosen IDs
    chosen_ids = outputs.sequences[0, prompt_length + 1:].cpu().numpy().astype(np.float32)

    t_end = time.perf_counter()

    return {
        "hidden_states": hidden_states_list,
        "attentions": attentions_list,
        "logits": logits_list,
        "chosen_ids": chosen_ids,
        "time_total": t_end - t0,
        "time_hs": t_hs - t0,
        "time_attn": t_attn - t_hs,
        "time_logits": t_logits - t_attn,
        "time_ids": t_end - t_logits,
    }


def new_postprocess(outputs, prompt_length: int) -> dict:
    """New approach: bulk GPU stack + single .cpu() transfer."""
    t0 = time.perf_counter()

    n_gen_steps = len(outputs.hidden_states) - 1  # exclude prefill
    n_layers = len(outputs.hidden_states[1]) if n_gen_steps > 0 else 0

    # ── Hidden states: build flat list of tensor refs, stack on GPU, single .cpu() ──
    if n_gen_steps > 0:
        # Collect all tensor references (fast — just Python list building, no GPU ops)
        hs_refs = []
        for t in range(1, n_gen_steps + 1):
            for l in range(n_layers):
                hs_refs.append(outputs.hidden_states[t][l][0, -1])

        # Single GPU stack + single CPU transfer
        hs_stacked = torch.stack(hs_refs)  # [T*L, hidden_dim]
        hs_np = hs_stacked.cpu().float().numpy()  # one transfer
        del hs_stacked

        # Reshape and split into per-step arrays
        hs_reshaped = hs_np.reshape(n_gen_steps, n_layers, -1)  # [T, L, hidden_dim]
        hidden_states_list = [hs_reshaped[t] for t in range(n_gen_steps)]
        del hs_np, hs_reshaped
    else:
        hidden_states_list = []

    t_hs = time.perf_counter()

    # ── Attentions: stack layers per step (variable seq_len prevents full batching) ──
    attentions_list = []
    if n_gen_steps > 0:
        n_attn_layers = len(outputs.attentions[1]) if outputs.attentions else 0
        for t in range(1, n_gen_steps + 1):
            if t < len(outputs.attentions) and outputs.attentions[t] is not None:
                # Stack all layers for this step on GPU, single .cpu()
                attn_stacked = torch.stack([
                    outputs.attentions[t][l][0, :, -1, :]
                    for l in range(n_attn_layers)
                ])  # [n_layers, n_heads, seq_len_at_t]
                attentions_list.append(attn_stacked.cpu().float().numpy())
                del attn_stacked

    t_attn = time.perf_counter()

    # ── Logits: stack all steps, single .cpu() ──
    if n_gen_steps > 0 and outputs.logits:
        logit_refs = [outputs.logits[t][0] for t in range(1, len(outputs.logits))]
        if logit_refs:
            logits_stacked = torch.stack(logit_refs)  # [T, vocab_size]
            logits_np = logits_stacked.cpu().float().numpy()
            logits_list = [logits_np[t] for t in range(logits_np.shape[0])]
            del logits_stacked, logits_np
        else:
            logits_list = []
    else:
        logits_list = []

    t_logits = time.perf_counter()

    # Chosen IDs
    chosen_ids = outputs.sequences[0, prompt_length + 1:].cpu().numpy().astype(np.float32)

    t_end = time.perf_counter()

    return {
        "hidden_states": hidden_states_list,
        "attentions": attentions_list,
        "logits": logits_list,
        "chosen_ids": chosen_ids,
        "time_total": t_end - t0,
        "time_hs": t_hs - t0,
        "time_attn": t_attn - t_hs,
        "time_logits": t_logits - t_attn,
        "time_ids": t_end - t_logits,
    }


def verify_match(old: dict, new: dict) -> bool:
    """Verify old and new post-processing produce identical results."""
    ok = True

    # Hidden states
    if len(old["hidden_states"]) != len(new["hidden_states"]):
        logger.error(f"HS count mismatch: {len(old['hidden_states'])} vs {len(new['hidden_states'])}")
        ok = False
    else:
        for i in range(len(old["hidden_states"])):
            if not np.allclose(old["hidden_states"][i], new["hidden_states"][i], atol=1e-6):
                max_diff = np.max(np.abs(old["hidden_states"][i] - new["hidden_states"][i]))
                logger.error(f"HS mismatch at step {i}: max_diff={max_diff}")
                ok = False
                break

    # Logits
    if len(old["logits"]) != len(new["logits"]):
        logger.error(f"Logit count mismatch: {len(old['logits'])} vs {len(new['logits'])}")
        ok = False
    else:
        for i in range(len(old["logits"])):
            if not np.allclose(old["logits"][i], new["logits"][i], atol=1e-6):
                max_diff = np.max(np.abs(old["logits"][i] - new["logits"][i]))
                logger.error(f"Logit mismatch at step {i}: max_diff={max_diff}")
                ok = False
                break

    # Attentions
    if len(old["attentions"]) != len(new["attentions"]):
        logger.error(f"Attn count mismatch: {len(old['attentions'])} vs {len(new['attentions'])}")
        ok = False
    else:
        for i in range(len(old["attentions"])):
            if not np.allclose(old["attentions"][i], new["attentions"][i], atol=1e-6):
                max_diff = np.max(np.abs(old["attentions"][i] - new["attentions"][i]))
                logger.error(f"Attn mismatch at step {i}: max_diff={max_diff}")
                ok = False
                break

    # Chosen IDs
    if not np.array_equal(old["chosen_ids"], new["chosen_ids"]):
        logger.error("Chosen IDs mismatch")
        ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if "3.2" in args.model or "3B" in args.model:
        eos_token_ids = [128001, 128009]
        dtype = torch.float16
    else:
        eos_token_ids = [128001, 128008, 128009]
        dtype = torch.bfloat16

    logger.info(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    messages = [{"role": "user", "content": PROMPT}]
    result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_length = input_ids.shape[1]
    logger.info(f"Prompt length: {prompt_length}")

    for rep in range(args.repeats):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {rep + 1}/{args.repeats}")

        # Generate once, reuse outputs for both post-processing approaches
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.synchronize()

        t_gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=0.6, top_p=0.9, do_sample=True,
                eos_token_id=eos_token_ids,
                output_hidden_states=True, output_attentions=True,
                output_logits=True, return_dict_in_generate=True,
            )
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t_gen_start

        n_tokens = outputs.sequences.shape[1] - prompt_length
        n_hs_steps = len(outputs.hidden_states) - 1
        n_layers = len(outputs.hidden_states[1]) if n_hs_steps > 0 else 0
        logger.info(f"Generated {n_tokens} tokens in {t_gen:.2f}s ({n_hs_steps} gen steps, {n_layers} layers)")
        logger.info(f"Total tensor objects: {n_hs_steps * n_layers} hs + {n_hs_steps * (n_layers-1)} attn + {len(outputs.logits)} logits")

        # OLD approach
        old_result = old_postprocess(outputs, prompt_length)
        logger.info(f"OLD postprocess: {old_result['time_total']:.3f}s "
                     f"(hs={old_result['time_hs']:.3f}s, attn={old_result['time_attn']:.3f}s, "
                     f"logits={old_result['time_logits']:.3f}s)")

        # NEW approach
        new_result = new_postprocess(outputs, prompt_length)
        logger.info(f"NEW postprocess: {new_result['time_total']:.3f}s "
                     f"(hs={new_result['time_hs']:.3f}s, attn={new_result['time_attn']:.3f}s, "
                     f"logits={new_result['time_logits']:.3f}s)")

        speedup = old_result["time_total"] / max(new_result["time_total"], 0.001)
        logger.info(f"Speedup: {speedup:.2f}×")

        # Verify outputs match
        match = verify_match(old_result, new_result)
        logger.info(f"Outputs match: {match}")

        logger.info(f"\nBreakdown (generate vs postprocess):")
        logger.info(f"  Generate:       {t_gen:.2f}s")
        logger.info(f"  OLD postprocess: {old_result['time_total']:.3f}s ({old_result['time_total']/t_gen*100:.0f}% of gen time)")
        logger.info(f"  NEW postprocess: {new_result['time_total']:.3f}s ({new_result['time_total']/t_gen*100:.0f}% of gen time)")

        del outputs, old_result, new_result
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
