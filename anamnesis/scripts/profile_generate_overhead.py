#!/usr/bin/env python3
"""Profile the overhead of output_hidden_states/attentions/logits in HF generate.

Isolates per-flag cost by running the same prompt+seed through generate() with
different flag combinations. Also measures GPU utilization and memory.

Usage:
    python -m anamnesis.scripts.profile_generate_overhead
    python -m anamnesis.scripts.profile_generate_overhead --model meta-llama/Llama-3.1-8B-Instruct
    python -m anamnesis.scripts.profile_generate_overhead --max-new-tokens 256
    python -m anamnesis.scripts.profile_generate_overhead --deep  # torch.profiler trace
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROMPT = "Explain how photosynthesis works in plants."


def get_gpu_stats() -> dict:
    """Get current GPU memory and utilization."""
    if not torch.cuda.is_available():
        return {}
    return {
        "memory_allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "memory_reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def reset_gpu_stats() -> None:
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def run_generation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_ids: list[int],
    seed: int,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
    output_logits: bool = False,
    label: str = "",
) -> dict:
    """Run a single generation with specific flags and measure performance."""
    reset_gpu_stats()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Warmup GPU
    torch.cuda.synchronize()

    t_start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=eos_token_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_logits=output_logits,
            return_dict_in_generate=True,
        )

    torch.cuda.synchronize()
    t_generate = time.perf_counter() - t_start

    prompt_length = input_ids.shape[1]
    num_tokens = outputs.sequences.shape[1] - prompt_length

    gpu_after = get_gpu_stats()

    # Count output objects (to measure tuple overhead)
    n_hs_objects = 0
    if output_hidden_states and hasattr(outputs, "hidden_states") and outputs.hidden_states:
        for step in outputs.hidden_states:
            n_hs_objects += len(step)

    n_attn_objects = 0
    if output_attentions and hasattr(outputs, "attentions") and outputs.attentions:
        for step in outputs.attentions:
            if step is not None:
                n_attn_objects += len(step)

    n_logit_objects = 0
    if output_logits and hasattr(outputs, "logits") and outputs.logits:
        n_logit_objects = len(outputs.logits)

    # Measure cleanup time
    t_cleanup_start = time.perf_counter()
    del outputs
    torch.cuda.empty_cache()
    gc.collect()
    t_cleanup = time.perf_counter() - t_cleanup_start

    result = {
        "label": label,
        "output_hidden_states": output_hidden_states,
        "output_attentions": output_attentions,
        "output_logits": output_logits,
        "num_tokens": num_tokens,
        "time_seconds": round(t_generate, 3),
        "time_per_token_ms": round(t_generate / max(num_tokens, 1) * 1000, 2),
        "cleanup_seconds": round(t_cleanup, 3),
        "peak_memory_mb": round(gpu_after.get("max_memory_allocated_mb", 0), 1),
        "n_hidden_state_objects": n_hs_objects,
        "n_attention_objects": n_attn_objects,
        "n_logit_objects": n_logit_objects,
        "total_output_objects": n_hs_objects + n_attn_objects + n_logit_objects,
    }

    return result


def run_flag_isolation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_ids: list[int],
    seed: int,
    repeats: int = 3,
) -> list[dict]:
    """Run generate() with different flag combinations to isolate overhead."""

    configs = [
        {"label": "A: baseline (no outputs)", "hs": False, "attn": False, "logits": False},
        {"label": "B: +hidden_states only", "hs": True, "attn": False, "logits": False},
        {"label": "C: +attentions only", "hs": False, "attn": True, "logits": False},
        {"label": "D: +logits only", "hs": False, "attn": False, "logits": True},
        {"label": "E: hidden_states+attentions", "hs": True, "attn": True, "logits": False},
        {"label": "F: all three (production)", "hs": True, "attn": True, "logits": True},
    ]

    all_results = []

    for config in configs:
        logger.info(f"\n--- {config['label']} ---")
        times = []
        last_result = None

        for r in range(repeats):
            result = run_generation(
                model, input_ids, max_new_tokens,
                temperature, top_p, eos_token_ids, seed,
                output_hidden_states=config["hs"],
                output_attentions=config["attn"],
                output_logits=config["logits"],
                label=config["label"],
            )
            times.append(result["time_seconds"])
            last_result = result
            logger.info(
                f"  Run {r+1}: {result['time_seconds']:.3f}s "
                f"({result['num_tokens']} tokens, "
                f"{result['time_per_token_ms']:.1f}ms/tok, "
                f"peak {result['peak_memory_mb']:.0f}MB, "
                f"{result['total_output_objects']} objects)"
            )

        # Aggregate
        assert last_result is not None
        summary = last_result.copy()
        summary["times"] = times
        summary["time_mean"] = round(np.mean(times), 3)
        summary["time_std"] = round(np.std(times), 3)
        summary["time_per_token_ms"] = round(np.mean(times) / max(last_result["num_tokens"], 1) * 1000, 2)
        all_results.append(summary)

        logger.info(
            f"  Mean: {summary['time_mean']:.3f}s ± {summary['time_std']:.3f}s"
        )

    return all_results


def run_deep_profile(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_ids: list[int],
    seed: int,
) -> None:
    """Run torch.profiler trace for a single generation with all flags."""
    logger.info("\n--- Deep profile (torch.profiler) ---")
    logger.info("Running with all output flags enabled...")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Limit tokens for manageable trace
    profile_tokens = min(max_new_tokens, 64)
    logger.info(f"Profiling {profile_tokens} tokens (capped for trace size)")

    try:
        from torch.profiler import profile, ProfilerActivity, schedule

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            # Profile everything (no warmup/active split for simplicity)
        ) as prof:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=profile_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    eos_token_id=eos_token_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                    output_logits=True,
                    return_dict_in_generate=True,
                )

        # Print key table
        logger.info("\nTop 20 operations by CPU time:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

        logger.info("\nTop 20 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        # Save trace
        trace_path = "/workspace/profile_trace.json"
        prof.export_chrome_trace(trace_path)
        logger.info(f"\nFull trace saved to {trace_path}")
        logger.info("View with chrome://tracing or https://ui.perfetto.dev/")

        del outputs
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        logger.error(f"Profiler failed: {e}")
        logger.info("Falling back to manual timing...")


def print_summary(results: list[dict]) -> None:
    """Print a comparison table."""
    logger.info("\n" + "=" * 80)
    logger.info("PROFILING SUMMARY")
    logger.info("=" * 80)

    baseline_time = results[0]["time_mean"] if results else 1.0

    logger.info(f"\n{'Config':<35} {'Mean(s)':>8} {'±':>3} {'Std':>6} "
                f"{'ms/tok':>7} {'Overhead':>8} {'PeakMB':>8} {'Objects':>8}")
    logger.info("-" * 95)

    for r in results:
        overhead = r["time_mean"] / baseline_time
        logger.info(
            f"{r['label']:<35} {r['time_mean']:>8.3f}   ± {r['time_std']:>5.3f} "
            f"{r['time_per_token_ms']:>7.1f} {overhead:>7.1f}× "
            f"{r['peak_memory_mb']:>8.0f} {r['total_output_objects']:>8}"
        )

    logger.info("\nKey insights:")
    if len(results) >= 2:
        hs_overhead = results[1]["time_mean"] / baseline_time
        logger.info(f"  output_hidden_states overhead: {hs_overhead:.1f}× baseline")
    if len(results) >= 3:
        attn_overhead = results[2]["time_mean"] / baseline_time
        logger.info(f"  output_attentions overhead: {attn_overhead:.1f}× baseline")
    if len(results) >= 4:
        logit_overhead = results[3]["time_mean"] / baseline_time
        logger.info(f"  output_logits overhead: {logit_overhead:.1f}× baseline")
    if len(results) >= 6:
        full_overhead = results[5]["time_mean"] / baseline_time
        logger.info(f"  All flags combined overhead: {full_overhead:.1f}× baseline")

        # Check if overhead is additive or super-linear
        sum_individual = (
            (results[1]["time_mean"] - baseline_time) +
            (results[2]["time_mean"] - baseline_time) +
            (results[3]["time_mean"] - baseline_time)
        )
        combined_delta = results[5]["time_mean"] - baseline_time
        if sum_individual > 0:
            interaction = combined_delta / sum_individual
            logger.info(f"  Interaction factor: {interaction:.2f}× "
                        f"({'super-linear' if interaction > 1.1 else 'roughly additive'})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile generate() overhead")
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model ID",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=3, help="Runs per config")
    parser.add_argument("--deep", action="store_true", help="Run torch.profiler trace")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Model setup
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

    # Prepare input
    messages = [{"role": "user", "content": PROMPT}]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    logger.info(f"Prompt length: {input_ids.shape[1]} tokens")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Repeats per config: {args.repeats}")

    # Warmup run (compile kernels, etc.)
    logger.info("\nWarmup run...")
    _ = run_generation(
        model, input_ids, min(args.max_new_tokens, 32),
        args.temperature, args.top_p, eos_token_ids, args.seed,
        label="warmup",
    )
    logger.info("Warmup done\n")

    # Flag isolation experiment
    results = run_flag_isolation(
        model, input_ids, args.max_new_tokens,
        args.temperature, args.top_p, eos_token_ids, args.seed,
        repeats=args.repeats,
    )

    print_summary(results)

    # Optional deep profiling
    if args.deep:
        run_deep_profile(
            model, input_ids, args.max_new_tokens,
            args.temperature, args.top_p, eos_token_ids, args.seed,
        )

    # Save results
    output_path = Path("/workspace/profile_results.json")
    try:
        serializable = []
        for r in results:
            sr = {k: v for k, v in r.items()}
            sr["times"] = [float(t) for t in sr.get("times", [])]
            serializable.append(sr)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save results: {e}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
