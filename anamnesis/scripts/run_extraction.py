"""Unified extraction script — single entry point for all extraction runs.

Replaces per-model experiment scripts with a parameterized interface
supporting model selection, mode sets, sample counts, raw tensor saving,
and prompt-swap validation.

Usage:
    # Full mixed-mode extraction with raw tensors
    python -m anamnesis.scripts.run_extraction \\
        --model 8b --modes mixed --n-samples 4 \\
        --save-raw --include-prompt-swap \\
        --run-name feature_iter_01

    # Quick Run 4 reproduction (backward compatible)
    python -m anamnesis.scripts.run_extraction \\
        --model 8b --modes run4 --n-samples 20 \\
        --run-name 8b_run4_repro

    # Smoke test
    python -m anamnesis.scripts.run_extraction \\
        --model 8b --modes run4 --n-samples 1 \\
        --save-raw --smoke-test \\
        --run-name smoke_test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DATA_ROOT = Path(os.environ.get(
    "ANAMNESIS_LEGACY_DATA",
    str(PROJECT_ROOT.parent / "phase_0"),
))
OUTPUTS_BASE = PROJECT_ROOT / "outputs"


# ── Model configs ─────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "torch_dtype": "bfloat16",
        "sampled_layers": [0, 8, 16, 20, 24, 28, 31],
        "pca_layers": [8, 16, 20, 24, 28],
        "early_layer_cutoff": 8,
        "late_layer_cutoff": 24,
        "temperature": 0.6,
        "eos_token_ids": [128001, 128008, 128009],
        "calibration_dir": OUTPUTS_BASE / "calibration" / "llama31_8b",
    },
    "3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "num_layers": 28,
        "hidden_dim": 3072,
        "num_attention_heads": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
        "torch_dtype": "float16",
        "sampled_layers": [0, 7, 14, 18, 21, 24, 27],
        "pca_layers": [7, 14, 18, 21, 24],
        "early_layer_cutoff": 7,
        "late_layer_cutoff": 21,
        "temperature": 0.7,
        "eos_token_ids": [128001, 128009],
        "calibration_dir": LEGACY_DATA_ROOT / "outputs" / "calibration",
    },
}


# ── Spec builders ─────────────────────────────────────────────────────────────

def _load_topics() -> list[str]:
    """Load the standard 20 topics."""
    prompts_path = PROJECT_ROOT / "prompts" / "prompt_sets.json"
    if not prompts_path.exists():
        # Fallback to legacy data root
        prompts_path = LEGACY_DATA_ROOT / "prompts" / "prompt_sets.json"

    with open(prompts_path) as f:
        prompts = json.load(f)

    topics: list[str] = []
    topics.extend(prompts["topics"]["set_a"])
    topics.extend(prompts["topics"]["set_b"])
    return topics


def build_specs(
    mode_set: str,
    n_samples_per_mode: int,
    include_prompt_swap: bool,
    prompt_set_prefix: str = "EXT",
) -> list[dict]:
    """Build generation specs for the requested mode set.

    Returns list of dicts with keys: mode, mode_idx, topic, topic_idx,
    system_prompt, user_prompt, generation_id, seed, repetition, condition.
    """
    from anamnesis.modes.run4_modes import RUN4_MODES, RUN4_MODE_INDEX
    from anamnesis.modes.extended_modes import EXTENDED_MODES, EXTENDED_MODE_INDEX
    from anamnesis.modes.prompt_swap import PROMPT_SWAP_PAIRS
    from anamnesis.extraction.generation_runner import make_seed

    topics = _load_topics()

    # Select mode set
    if mode_set == "run4":
        modes = RUN4_MODES
        mode_index = RUN4_MODE_INDEX
    elif mode_set in ("mixed", "extended"):
        modes = EXTENDED_MODES
        mode_index = EXTENDED_MODE_INDEX
    else:
        raise ValueError(f"Unknown mode set: {mode_set}. Use 'run4' or 'mixed'.")

    user_template = "Write about: {topic}"

    specs: list[dict] = []
    gen_id = 0

    # Determine how many topics × reps to cover n_samples_per_mode
    n_topics = len(topics)
    reps_needed = max(1, (n_samples_per_mode + n_topics - 1) // n_topics)

    for rep in range(reps_needed):
        for topic_idx, topic in enumerate(topics):
            if sum(1 for s in specs if s["mode"] == list(modes.keys())[0]) >= n_samples_per_mode:
                break
            for mode_name in modes:
                specs.append({
                    "generation_id": gen_id,
                    "prompt_set": prompt_set_prefix,
                    "topic": topic,
                    "topic_idx": topic_idx,
                    "mode": mode_name,
                    "mode_idx": mode_index[mode_name],
                    "system_prompt": modes[mode_name],
                    "user_prompt": user_template.format(topic=topic),
                    "seed": make_seed(topic_idx, mode_index[mode_name], rep, prompt_set_prefix),
                    "repetition": rep,
                    "condition": "standard",
                })
                gen_id += 1

    # Trim to exactly n_samples_per_mode per mode
    mode_counts: dict[str, int] = {}
    trimmed: list[dict] = []
    for spec in specs:
        mode = spec["mode"]
        count = mode_counts.get(mode, 0)
        if count < n_samples_per_mode:
            trimmed.append(spec)
            mode_counts[mode] = count + 1
    specs = trimmed

    # Add prompt-swap generations
    if include_prompt_swap:
        swap_gen_id = 10000  # High IDs to avoid collision
        n_swap_topics = min(10, n_topics)
        for pair in PROMPT_SWAP_PAIRS:
            for topic_idx in range(n_swap_topics):
                topic = topics[topic_idx]
                specs.append({
                    "generation_id": swap_gen_id,
                    "prompt_set": prompt_set_prefix,
                    "topic": topic,
                    "topic_idx": topic_idx,
                    "mode": f"swap_{pair.label}",
                    "mode_idx": -1,  # Not a standard mode
                    "system_prompt": pair.get_system_prompt(),
                    "user_prompt": pair.format_user_prompt(topic, user_template),
                    "seed": make_seed(topic_idx, 99, 0, f"SWAP_{pair.label}"),
                    "repetition": 0,
                    "condition": f"prompt_swap:{pair.label}",
                })
                swap_gen_id += 1

    return specs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Unified extraction script")
    parser.add_argument(
        "--model", choices=list(MODEL_CONFIGS.keys()), required=True,
        help="Model to use (3b or 8b)",
    )
    parser.add_argument(
        "--modes", choices=["run4", "mixed"], default="run4",
        help="Mode set (run4: original 5, mixed: 8 modes including Run 3 adapted)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Number of samples per mode (default: 20)",
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Save raw per-token tensors for GPU-free feature iteration",
    )
    parser.add_argument(
        "--include-prompt-swap", action="store_true",
        help="Include prompt-swap generations for confound testing",
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Name for this extraction run (determines output directory)",
    )
    parser.add_argument(
        "--no-tier3", action="store_true",
        help="Disable Tier 3 PCA features",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick test with 1 sample, reduced tokens",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print specs without generating",
    )
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model]

    if args.smoke_test:
        args.n_samples = 1

    # Build output paths
    outputs_dir = OUTPUTS_BASE / "runs" / args.run_name
    signatures_dir = outputs_dir / "signatures"
    raw_dir = outputs_dir / "raw_tensors" if args.save_raw else None

    # Build specs
    specs = build_specs(
        mode_set=args.modes,
        n_samples_per_mode=args.n_samples,
        include_prompt_swap=args.include_prompt_swap,
        prompt_set_prefix=f"{args.model.upper()}_{args.modes}",
    )

    # Summary
    mode_counts: dict[str, int] = {}
    for s in specs:
        mode_counts[s["mode"]] = mode_counts.get(s["mode"], 0) + 1

    print(f"Extraction run: {args.run_name}")
    print(f"  Model: {model_cfg['model_id']}")
    print(f"  Modes: {args.modes} ({len(mode_counts)} modes)")
    for mode, count in sorted(mode_counts.items()):
        cond = "swap" if mode.startswith("swap_") else "standard"
        print(f"    {mode}: {count} samples [{cond}]")
    print(f"  Total: {len(specs)} generations")
    print(f"  Save raw: {args.save_raw}")
    print(f"  Output: {outputs_dir}")

    if args.dry_run:
        print("\n[DRY RUN] Would generate the above specs. Exiting.")
        return

    # ── Load model and calibration ──
    from anamnesis.config import (
        ModelConfig,
        GenerationConfig,
        ExtractionConfig,
        CalibrationConfig,
        ExperimentConfig,
        GenerationSpec,
    )
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.generation_runner import run_experiment

    # Build config
    extraction_config = ExtractionConfig(
        sampled_layers=model_cfg["sampled_layers"],
        pca_layers=model_cfg["pca_layers"],
        early_layer_cutoff=model_cfg["early_layer_cutoff"],
        late_layer_cutoff=model_cfg["late_layer_cutoff"],
        enable_tier3=not args.no_tier3,
        save_raw_tensors=args.save_raw,
    )

    gen_config = GenerationConfig(
        temperature=model_cfg["temperature"],
        eos_token_ids=model_cfg["eos_token_ids"],
        max_new_tokens=100 if args.smoke_test else 512,
    )

    model_config = ModelConfig(
        model_id=model_cfg["model_id"],
        num_layers=model_cfg["num_layers"],
        hidden_dim=model_cfg["hidden_dim"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_kv_heads=model_cfg["num_kv_heads"],
        head_dim=model_cfg["head_dim"],
        torch_dtype=model_cfg["torch_dtype"],
    )

    cal_dir = model_cfg["calibration_dir"]
    cal_config = CalibrationConfig(
        positional_means_path=cal_dir / "positional_means.npz",
        pca_model_path=cal_dir / "pca_model.pkl",
    )

    experiment_config = ExperimentConfig(
        model=model_config,
        generation=gen_config,
        extraction=extraction_config,
        calibration=cal_config,
        outputs_dir=outputs_dir,
        signatures_dir=signatures_dir,
        metadata_path=outputs_dir / "metadata.json",
        results_path=outputs_dir / "results.json",
    )

    experiment_config.ensure_dirs()

    # Load calibration data
    positional_means: F32 | None = None
    pca_components: F32 | None = None
    pca_mean: F32 | None = None

    if cal_config.positional_means_path.exists():
        pm_data = np.load(cal_config.positional_means_path)
        positional_means = pm_data["positional_means"].astype(np.float32)
        logger.info(f"Loaded positional means: {positional_means.shape}")

    if cal_config.pca_model_path.exists() and not args.no_tier3:
        with open(cal_config.pca_model_path, "rb") as f:
            pca_data = pickle.load(f)
        if isinstance(pca_data, dict):
            pca_components = pca_data["components"].astype(np.float32)
            pca_mean = pca_data["mean"].astype(np.float32)
        else:
            pca_components = pca_data.components_.astype(np.float32)
            pca_mean = pca_data.mean_.astype(np.float32)
        logger.info(f"Loaded PCA model: {pca_components.shape}")

    # Load model
    logger.info(f"Loading model: {model_config.model_id}")
    loaded = load_model(
        model_config,
        sampled_layers=extraction_config.sampled_layers,
        register_gate_hooks=args.save_raw,  # capture SwiGLU gates when doing fat extraction
    )

    # Verify architecture
    n_layers = len(loaded.model.model.layers)
    assert n_layers == model_config.num_layers, (
        f"Model has {n_layers} layers but config says {model_config.num_layers}"
    )

    # Convert specs to GenerationSpec objects
    gen_specs = []
    for s in specs:
        gen_specs.append(GenerationSpec(
            generation_id=s["generation_id"],
            prompt_set=s["prompt_set"],
            topic=s["topic"],
            topic_idx=s["topic_idx"],
            mode=s["mode"],
            mode_idx=s["mode_idx"],
            system_prompt=s["system_prompt"],
            user_prompt=s["user_prompt"],
            seed=s["seed"],
            repetition=s["repetition"],
        ))

    # Run
    logger.info(f"Starting extraction: {len(gen_specs)} generations")
    metadata_list = run_experiment(
        loaded=loaded,
        config=experiment_config,
        positional_means=positional_means,
        pca_components=pca_components,
        pca_mean=pca_mean,
        specs=gen_specs,
        save_raw=args.save_raw,
    )

    # Cleanup
    loaded.remove_hooks()

    logger.info(f"Extraction complete: {len(metadata_list)} generations saved to {outputs_dir}")


if __name__ == "__main__":
    main()
