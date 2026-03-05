#!/usr/bin/env python3
"""Positional decomposition calibration for Llama 3.1 8B Instruct.

Adapted from Phase 0. Must run before the main experiment.
Produces model-specific positional means and PCA basis.

Outputs:
  - outputs/calibration/llama31_8b/positional_means.npz
  - outputs/calibration/llama31_8b/pca_model.pkl

Usage:
    python -m anamnesis.scripts.run_8b_calibration
    python -m anamnesis.scripts.run_8b_calibration --dry-run  # check config only
"""

from __future__ import annotations

import argparse
import gc
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anamnesis.config import ExperimentConfig
from anamnesis.extraction.model_loader import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Diverse calibration prompts — same as Phase 0 (content-washing)
CALIBRATION_PROMPTS = [
    "Explain how photosynthesis works in plants.",
    "What are the main causes of the French Revolution?",
    "Describe the process of making traditional Japanese ramen.",
    "How do electric vehicles compare to gasoline cars?",
    "What is the significance of the Rosetta Stone?",
    "Explain the concept of supply and demand in economics.",
    "How does the human immune system fight infections?",
    "Describe the architecture of Gothic cathedrals.",
    "What are the principles of object-oriented programming?",
    "How do tides work and what causes them?",
    "Explain the theory of plate tectonics.",
    "What makes a good leader?",
    "How do birds navigate during migration?",
    "Describe the water cycle and its importance.",
    "What is quantum entanglement?",
    "How do vaccines work?",
    "Explain the causes and effects of inflation.",
    "What are the different types of clouds?",
    "How does a combustion engine work?",
    "Describe the life cycle of a star.",
    "What is machine learning and how does it differ from traditional programming?",
    "How do earthquakes happen?",
    "Explain the basics of music theory.",
    "What are renewable energy sources?",
    "How does the stock market work?",
    "Describe the process of fermentation.",
    "What are the effects of sleep deprivation?",
    "How do submarines work?",
    "Explain the concept of natural selection.",
    "What is the significance of pi in mathematics?",
    "How do 3D printers work?",
    "Describe the history of the internet.",
    "What causes aurora borealis?",
    "How do computers store and retrieve data?",
    "Explain the process of osmosis.",
    "What are the major types of rocks?",
    "How do airplanes fly?",
    "Describe the structure of DNA.",
    "What is cryptocurrency and how does blockchain work?",
    "How do telescopes work?",
    "Explain the greenhouse effect.",
    "What are the stages of grief?",
    "How does sonar work?",
    "Describe the Silk Road and its importance.",
    "What is dark matter?",
    "How do coral reefs form?",
    "Explain the basics of game theory.",
    "What are the layers of the atmosphere?",
    "How does a nuclear reactor work?",
    "Describe the process of cheese making.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 8B calibration")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Override number of calibration prompts (for quick testing)",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    config.ensure_dirs()

    if args.dry_run:
        print(f"Model: {config.model.model_id}")
        print(f"Layers: {config.model.num_layers}")
        print(f"Hidden dim: {config.model.hidden_dim}")
        print(f"Attention heads: {config.model.num_attention_heads}")
        print(f"KV heads: {config.model.num_kv_heads}")
        print(f"dtype: {config.model.torch_dtype}")
        print(f"Sampled layers: {config.extraction.sampled_layers}")
        print(f"PCA layers: {config.extraction.pca_layers}")
        print(f"EOS tokens: {config.generation.eos_token_ids}")
        print(f"Temperature: {config.generation.temperature}")
        print(f"Calibration output: {config.calibration.positional_means_path}")
        print(f"PCA output: {config.calibration.pca_model_path}")
        print(f"Calibration prompts: {len(CALIBRATION_PROMPTS)}")
        return

    prompts = CALIBRATION_PROMPTS
    if args.num_prompts is not None:
        prompts = prompts[:args.num_prompts]

    logger.info(f"Calibration with {len(prompts)} prompts for {config.model.model_id}")

    # Load model without hooks (not needed for calibration)
    loaded = load_model(config.model, sampled_layers=[])
    loaded.disable_hooks()

    num_layers_plus_embed = config.model.num_layers + 1  # 33 for 8B
    hidden_dim = config.model.hidden_dim
    max_positions = config.generation.max_new_tokens + 200

    # Accumulators
    pos_sums = np.zeros((num_layers_plus_embed, max_positions, hidden_dim), dtype=np.float64)
    pos_counts = np.zeros((num_layers_plus_embed, max_positions), dtype=np.int64)
    pca_samples: list[np.ndarray] = []

    for prompt_idx, prompt_text in enumerate(tqdm(prompts, desc="Calibration")):
        messages = [{"role": "user", "content": prompt_text}]
        result = loaded.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        if isinstance(result, torch.Tensor):
            input_ids = result
        else:
            input_ids = result["input_ids"]
        prompt_length = input_ids.shape[1]
        device = next(loaded.model.parameters()).device
        input_ids = input_ids.to(device)

        torch.manual_seed(prompt_idx)

        with torch.no_grad():
            outputs = loaded.model.generate(
                input_ids,
                max_new_tokens=config.calibration.calibration_max_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                do_sample=True,
                eos_token_id=config.generation.eos_token_ids,
                output_hidden_states=True,
                output_attentions=False,
                return_dict_in_generate=True,
            )

        # Process prefill hidden states
        prefill = outputs.hidden_states[0]
        for l in range(min(len(prefill), num_layers_plus_embed)):
            h = prefill[l][0].cpu().float().numpy()
            for pos in range(min(h.shape[0], max_positions)):
                pos_sums[l, pos] += h[pos].astype(np.float64)
                pos_counts[l, pos] += 1

        # Process generation step hidden states
        for t in range(1, len(outputs.hidden_states)):
            step = outputs.hidden_states[t]
            abs_pos = prompt_length + t - 1
            if abs_pos >= max_positions:
                break
            for l in range(min(len(step), num_layers_plus_embed)):
                h = step[l][0, -1].cpu().float().numpy()
                pos_sums[l, abs_pos] += h.astype(np.float64)
                pos_counts[l, abs_pos] += 1

        # Collect PCA samples
        num_gen_steps = len(outputs.hidden_states) - 1
        if num_gen_steps > 0:
            sample_positions = [1, max(1, num_gen_steps // 2), num_gen_steps]
            for t in sample_positions:
                if t < len(outputs.hidden_states):
                    for l_idx in config.extraction.pca_layers:
                        if l_idx + 1 < len(outputs.hidden_states[t]):
                            h = outputs.hidden_states[t][l_idx + 1][0, -1].cpu().float().numpy()
                            pca_samples.append(h)

        # Cleanup
        del outputs
        del input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if (prompt_idx + 1) % 10 == 0:
            logger.info(f"Calibration progress: {prompt_idx + 1}/{len(prompts)}")

    # ── Compute positional means — vectorized ──
    mask = pos_counts > 5
    positional_means = np.zeros_like(pos_sums, dtype=np.float32)
    # Broadcast division: pos_counts needs expanding to match hidden_dim axis
    safe_counts = np.where(mask, pos_counts, 1)  # avoid div-by-zero
    positional_means = np.where(
        mask[:, :, np.newaxis],
        (pos_sums / safe_counts[:, :, np.newaxis]).astype(np.float32),
        0.0,
    ).astype(np.float32)

    calib_path = config.calibration.positional_means_path
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        calib_path,
        positional_means=positional_means,
        pos_counts=pos_counts,
    )
    logger.info(f"Positional means saved to {calib_path}")
    logger.info(f"  Shape: {positional_means.shape}")
    max_calibrated = np.max(np.where(pos_counts.sum(axis=0) > 0)) if pos_counts.sum() > 0 else 0
    logger.info(f"  Max calibrated position: {max_calibrated}")

    # ── Fit PCA ──
    if pca_samples:
        from sklearn.decomposition import PCA

        pca_matrix = np.stack(pca_samples).astype(np.float64)
        logger.info(f"Fitting PCA on {pca_matrix.shape[0]} samples of dim {pca_matrix.shape[1]}")

        n_components = min(config.extraction.pca_components, pca_matrix.shape[0], pca_matrix.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(pca_matrix)

        pca_data = {
            "components": pca.components_.astype(np.float32),
            "mean": pca.mean_.astype(np.float32),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }

        pca_path = config.calibration.pca_model_path
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pca_path, "wb") as f:
            pickle.dump(pca_data, f)

        logger.info(f"PCA model saved to {pca_path}")
        logger.info(f"  Components: {pca.components_.shape}")
        logger.info(f"  Explained variance (top 10): {pca.explained_variance_ratio_[:10]}")
        logger.info(f"  Total explained: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        logger.warning("No PCA samples collected — Tier 3 features will be empty")

    # Cleanup model
    loaded.remove_hooks()
    del loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info("Calibration complete!")


if __name__ == "__main__":
    main()
