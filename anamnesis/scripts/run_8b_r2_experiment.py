#!/usr/bin/env python3
"""Run the 8B paper-Run-2-equivalent experiment.

Generates 8B outputs under the five original Phase-0 Run-2 (paper numbering)
process-mode prompts (associative, compressed, deliberative, pedagogical,
structured), using the 15 topics the 3B Run-2 experiment used. These prompts
are NOT format-controlled by design — the cross-format transfer finding at 3B
(pedagogical → socratic 76%, dialectical → deliberative 85%, sil=-0.156 LDA
destructive interference) depends on this format variation. See
`anamnesis/modes/run3_original_modes.py` for provenance.

Default scope: 15 topics × 5 modes × 2 reps = 150 generations.
Output directory: `outputs/runs/run_8b_r2_equivalent/` (override via
ANAMNESIS_RUN_NAME env var if desired).

Usage:
    python -m anamnesis.scripts.run_8b_r2_experiment
    python -m anamnesis.scripts.run_8b_r2_experiment --dry-run
    python -m anamnesis.scripts.run_8b_r2_experiment --smoke-test
    python -m anamnesis.scripts.run_8b_r2_experiment --num-reps 1   # 75 gens, matches 3B Run-2 exactly
    python -m anamnesis.scripts.run_8b_r2_experiment --no-tier3     # skip PCA features

Must run calibration first (`run_8b_calibration.py`) — this script reuses the
same positional-means and PCA calibration as the baseline because the extraction
pipeline is model-dependent, not mode-dependent.

Safe to interrupt; signatures are saved per-generation and resumed on restart.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Set the run name BEFORE importing config so RUN_NAME is picked up at module load.
# (Users can override via ANAMNESIS_RUN_NAME env var.)
os.environ.setdefault("ANAMNESIS_RUN_NAME", "run_8b_r2_equivalent")

from anamnesis.config import ExperimentConfig  # noqa: E402
from anamnesis.extraction.generation_runner import build_generation_specs, run_experiment  # noqa: E402
from anamnesis.extraction.model_loader import load_model  # noqa: E402
from anamnesis.modes.run3_original_modes import (  # noqa: E402
    RUN3_ORIGINAL_MODES,
    RUN3_ORIGINAL_TOPICS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 8B paper-Run-2-equivalent experiment (cross-format transfer test-side data)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print specs without running")
    parser.add_argument("--no-tier3", action="store_true", help="Skip Tier 3 PCA features")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run only 3 generations to verify pipeline works",
    )
    parser.add_argument(
        "--num-reps", type=int, default=2,
        help="Repetitions per (topic, mode) cell. 2 matches 8B-baseline rep-structure (150 gens). "
             "1 matches 3B Run-2 data exactly (75 gens).",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    config.ensure_dirs()

    # Run-2 mode dict, Run-2 topics, distinct seed namespace
    specs = build_generation_specs(
        config,
        mode_dict=RUN3_ORIGINAL_MODES,
        topics=RUN3_ORIGINAL_TOPICS,
        num_reps=args.num_reps,
        prompt_set="8B_r2",
    )
    logger.info(f"Total generation specs: {len(specs)}")
    logger.info(
        f"  {len(RUN3_ORIGINAL_TOPICS)} topics × "
        f"{len(RUN3_ORIGINAL_MODES)} modes × "
        f"{args.num_reps} reps"
    )
    logger.info(f"  Modes: {list(RUN3_ORIGINAL_MODES.keys())}")

    if args.smoke_test:
        # One spec per mode from the first few topics — covers all 5 modes
        smoke_specs = [specs[i] for i in range(len(RUN3_ORIGINAL_MODES))]
        logger.info(f"Smoke test: running {len(smoke_specs)} samples (one per mode)")
        specs = smoke_specs

    if args.dry_run:
        for s in specs:
            print(
                f"  Gen {s.generation_id:3d}: {s.mode:13s} | "
                f"{s.topic[:40]:40s} | rep={s.repetition} | seed={s.seed}"
            )
        print(f"\nTotal: {len(specs)} generations")
        print(f"\nConfig:")
        print(f"  Model: {config.model.model_id}")
        print(f"  Output dir: {config.outputs_dir}")
        print(f"  Temperature: {config.generation.temperature}")
        print(f"  EOS tokens: {config.generation.eos_token_ids}")
        print(f"  dtype: {config.model.torch_dtype}")
        print(f"  Sampled layers: {config.extraction.sampled_layers}")
        print(f"  PCA layers: {config.extraction.pca_layers}")
        print(f"  Prompt set tag: {specs[0].prompt_set if specs else '-'}")
        return

    # --- Load calibration (shared with baseline; model-dependent not mode-dependent) ---
    positional_means = None
    calib_path = config.calibration.positional_means_path
    if calib_path.exists():
        calib = np.load(calib_path)
        positional_means = calib["positional_means"]
        logger.info(f"Loaded positional means: {positional_means.shape}")
    else:
        logger.warning(
            f"No positional calibration found at {calib_path} — "
            "running without correction. Run run_8b_calibration.py first!"
        )

    pca_components = None
    pca_mean = None
    if not args.no_tier3:
        pca_path = config.calibration.pca_model_path
        if pca_path.exists():
            with open(pca_path, "rb") as f:
                pca_data = pickle.load(f)
            pca_components = pca_data["components"]
            pca_mean = pca_data["mean"]
            logger.info(f"Loaded PCA model: {pca_components.shape[0]} components")
        else:
            logger.warning(f"No PCA model found at {pca_path} — Tier 3 will be empty")
            config.extraction.enable_tier3 = False
    else:
        config.extraction.enable_tier3 = False
        logger.info("Tier 3 disabled")

    # --- Load model ---
    logger.info(f"Loading {config.model.model_id}...")
    loaded = load_model(config.model, sampled_layers=config.extraction.sampled_layers)
    logger.info("Model loaded successfully")

    actual_layers = len(loaded.model.model.layers)
    if actual_layers != config.model.num_layers:
        logger.error(
            f"Model has {actual_layers} layers but config expects {config.model.num_layers}!"
        )
        sys.exit(1)
    logger.info(f"Architecture verified: {actual_layers} layers")

    # --- Run ---
    metadata = run_experiment(
        loaded=loaded,
        config=config,
        positional_means=positional_means,
        pca_components=pca_components,
        pca_mean=pca_mean,
        specs=specs,
    )

    logger.info(f"Done! {len(metadata)} generations completed.")
    loaded.remove_hooks()


if __name__ == "__main__":
    main()
