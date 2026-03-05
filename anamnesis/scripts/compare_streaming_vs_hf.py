#!/usr/bin/env python3
"""A/B comparison: streaming_generate vs HF generate on the same model.

Tests that our custom streaming loop produces valid outputs by comparing
against HuggingFace's generate() on the same prompt and seed.

Works with any causal LM — defaults to 3B for cheap/fast testing.

Usage:
    python -m anamnesis.scripts.compare_streaming_vs_hf
    python -m anamnesis.scripts.compare_streaming_vs_hf --model meta-llama/Llama-3.1-8B-Instruct
    python -m anamnesis.scripts.compare_streaming_vs_hf --num-prompts 3
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

from anamnesis.extraction.streaming_generate import streaming_generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEST_PROMPTS = [
    "Explain how photosynthesis works in plants.",
    "What are the main causes of the French Revolution?",
    "Describe the process of making traditional Japanese ramen.",
    "How do electric vehicles compare to gasoline cars?",
    "What is the significance of the Rosetta Stone?",
]


def run_hf_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_ids: list[int],
    seed: int,
) -> dict:
    """Run HF's generate() and collect outputs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=eos_token_ids,
            output_hidden_states=True,
            output_attentions=True,
            output_logits=True,
            return_dict_in_generate=True,
        )
    t_gen = time.time() - t0

    prompt_length = input_ids.shape[1]

    # Extract hidden states (skip prefill, last token only)
    hidden_states = []
    for t in range(1, len(outputs.hidden_states)):
        layers = []
        for l_tensor in outputs.hidden_states[t]:
            layers.append(l_tensor[0, -1].cpu().float().numpy())
        hidden_states.append(np.stack(layers))

    # Extract attentions (skip prefill)
    attentions = []
    for t in range(1, len(outputs.attentions)):
        layers = []
        for l_tensor in outputs.attentions[t]:
            layers.append(l_tensor[0, :, -1, :].cpu().float().numpy())
        attentions.append(np.stack(layers))

    # Extract logits (skip prefill — in recent HF, logits[0] IS prefill)
    logits = []
    for t in range(1, len(outputs.logits)):
        logits.append(outputs.logits[t][0].cpu().float().numpy())

    # Chosen token IDs (skip first gen token)
    chosen_ids = outputs.sequences[0, prompt_length + 1:].cpu().numpy()

    num_tokens = len(outputs.sequences[0]) - prompt_length

    # Cleanup
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "time": t_gen,
        "num_tokens": num_tokens,
        "hidden_states": hidden_states,
        "attentions": attentions,
        "logits": logits,
        "chosen_ids": chosen_ids,
    }


def run_streaming(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_ids: list[int],
    seed: int,
) -> dict:
    """Run our streaming_generate() and collect outputs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    stream_out = streaming_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_ids=eos_token_ids,
        output_hidden_states=True,
        output_attentions=True,
    )
    t_gen = time.time() - t0

    # chosen_ids: skip first token (same alignment as HF path)
    chosen_ids = np.array(stream_out.generated_token_ids[1:], dtype=np.int64)

    result = {
        "time": t_gen,
        "num_tokens": len(stream_out.generated_token_ids),
        "hidden_states": stream_out.hidden_states,
        "attentions": stream_out.attentions,
        "logits": stream_out.logits,
        "chosen_ids": chosen_ids,
    }

    del stream_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result


def compare_outputs(hf_result: dict, stream_result: dict, prompt_idx: int) -> dict:
    """Compare HF vs streaming outputs. Returns comparison metrics."""
    report: dict = {"prompt_idx": prompt_idx}

    # Speed comparison
    report["hf_time"] = hf_result["time"]
    report["stream_time"] = stream_result["time"]
    report["speedup"] = hf_result["time"] / max(stream_result["time"], 0.001)

    # Token count (won't match — different sampling paths)
    report["hf_tokens"] = hf_result["num_tokens"]
    report["stream_tokens"] = stream_result["num_tokens"]

    # Hidden states shape comparison
    hf_hs = hf_result["hidden_states"]
    st_hs = stream_result["hidden_states"]
    report["hf_hs_count"] = len(hf_hs)
    report["stream_hs_count"] = len(st_hs)

    if hf_hs and st_hs:
        report["hf_hs_shape"] = hf_hs[0].shape
        report["stream_hs_shape"] = st_hs[0].shape
        report["hs_shape_match"] = hf_hs[0].shape == st_hs[0].shape

        # Check for NaN/inf
        report["stream_hs_has_nan"] = any(np.isnan(h).any() for h in st_hs)
        report["stream_hs_has_inf"] = any(np.isinf(h).any() for h in st_hs)

        # Compare value ranges (should be similar order of magnitude)
        hf_norms = [np.linalg.norm(h) for h in hf_hs[:5]]
        st_norms = [np.linalg.norm(h) for h in st_hs[:5]]
        report["hf_hs_norm_range"] = (min(hf_norms), max(hf_norms))
        report["stream_hs_norm_range"] = (min(st_norms), max(st_norms))

    # Attentions shape
    hf_at = hf_result["attentions"]
    st_at = stream_result["attentions"]
    report["hf_attn_count"] = len(hf_at)
    report["stream_attn_count"] = len(st_at)
    if hf_at and st_at:
        report["attn_shape_match"] = hf_at[0].shape == st_at[0].shape
        report["stream_attn_has_nan"] = any(np.isnan(a).any() for a in st_at)

    # Logits shape
    hf_lg = hf_result["logits"]
    st_lg = stream_result["logits"]
    report["hf_logit_count"] = len(hf_lg)
    report["stream_logit_count"] = len(st_lg)
    if hf_lg and st_lg:
        report["logit_shape_match"] = hf_lg[0].shape == st_lg[0].shape
        report["stream_logit_has_nan"] = any(np.isnan(l).any() for l in st_lg)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare streaming vs HF generate")
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model ID (default: 3B for cheap testing)",
    )
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Short generations for fast comparison")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    # Determine EOS tokens based on model
    if "3.2" in args.model or "3B" in args.model:
        eos_token_ids = [128001, 128009]
        dtype = torch.float16
    else:
        eos_token_ids = [128001, 128008, 128009]
        dtype = torch.bfloat16

    logger.info(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info("Model loaded")

    prompts = TEST_PROMPTS[:args.num_prompts]
    all_reports = []

    for i, prompt_text in enumerate(prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt {i}: {prompt_text[:50]}...")

        messages = [{"role": "user", "content": prompt_text}]
        result = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        input_ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        prompt_length = input_ids.shape[1]
        seed = 42 + i

        # Run HF generate
        logger.info("Running HF generate()...")
        hf_result = run_hf_generate(
            model, input_ids, args.max_new_tokens,
            args.temperature, args.top_p, eos_token_ids, seed,
        )
        logger.info(f"  HF: {hf_result['num_tokens']} tokens in {hf_result['time']:.2f}s")

        # Run streaming generate
        logger.info("Running streaming_generate()...")
        stream_result = run_streaming(
            model, input_ids, args.max_new_tokens,
            args.temperature, args.top_p, eos_token_ids, seed,
        )
        logger.info(f"  Stream: {stream_result['num_tokens']} tokens in {stream_result['time']:.2f}s")

        # Compare
        report = compare_outputs(hf_result, stream_result, i)
        all_reports.append(report)

        logger.info(f"  Speedup: {report['speedup']:.2f}x")
        logger.info(f"  HS shape match: {report.get('hs_shape_match', 'N/A')}")
        logger.info(f"  HS NaN: {report.get('stream_hs_has_nan', 'N/A')}")
        logger.info(f"  Attn shape match: {report.get('attn_shape_match', 'N/A')}")
        logger.info(f"  Logit shape match: {report.get('logit_shape_match', 'N/A')}")
        logger.info(f"  HF HS norms: {report.get('hf_hs_norm_range', 'N/A')}")
        logger.info(f"  Stream HS norms: {report.get('stream_hs_norm_range', 'N/A')}")

        del hf_result, stream_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    avg_speedup = np.mean([r["speedup"] for r in all_reports])
    avg_hf_time = np.mean([r["hf_time"] for r in all_reports])
    avg_stream_time = np.mean([r["stream_time"] for r in all_reports])
    all_shapes_ok = all(r.get("hs_shape_match", False) for r in all_reports)
    any_nan = any(r.get("stream_hs_has_nan", False) for r in all_reports)

    logger.info(f"Prompts tested: {len(all_reports)}")
    logger.info(f"Avg HF time: {avg_hf_time:.2f}s")
    logger.info(f"Avg streaming time: {avg_stream_time:.2f}s")
    logger.info(f"Avg speedup: {avg_speedup:.2f}x")
    logger.info(f"All HS shapes match: {all_shapes_ok}")
    logger.info(f"Any NaN in streaming: {any_nan}")

    if all_shapes_ok and not any_nan and avg_speedup > 1.0:
        logger.info("\nVERDICT: PASS — streaming_generate is faster and produces valid outputs")
    elif all_shapes_ok and not any_nan:
        logger.info("\nVERDICT: PASS (shapes OK, no NaN) but no speedup observed")
    else:
        logger.info("\nVERDICT: FAIL — check reports above")


if __name__ == "__main__":
    main()
