#!/usr/bin/env python3
"""Run the 8B baseline experiment: 200 generations (20 topics × 5 modes × 2 reps).

Supports robust resume — checks disk for completed generations and skips them.
Run calibration first (run_8b_calibration.py).

Usage:
    python -m anamnesis.scripts.run_8b_experiment
    python -m anamnesis.scripts.run_8b_experiment --no-tier3     # skip PCA features
    python -m anamnesis.scripts.run_8b_experiment --dry-run      # print specs only
    python -m anamnesis.scripts.run_8b_experiment --smoke-test   # run 3 samples to verify pipeline

The experiment saves each generation independently (.npz + .json), so it is
safe to interrupt and resume at any time.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anamnesis.config import ExperimentConfig
from anamnesis.extraction.model_loader import load_model
from anamnesis.extraction.generation_runner import build_generation_specs, run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 8B baseline experiment")
    parser.add_argument("--dry-run", action="store_true", help="Print specs without running")
    parser.add_argument("--no-tier3", action="store_true", help="Skip Tier 3 PCA features")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run only 3 generations to verify pipeline works",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    config.ensure_dirs()

    # Build specs
    specs = build_generation_specs(config)
    logger.info(f"Total generation specs: {len(specs)}")

    if args.smoke_test:
        # Pick 3 diverse specs: different modes, different topics
        smoke_specs = [specs[0], specs[5], specs[10]]
        logger.info(f"Smoke test: running {len(smoke_specs)} samples")
        specs = smoke_specs

    if args.dry_run:
        for s in specs:
            print(
                f"  Gen {s.generation_id:3d}: {s.mode:12s} | "
                f"{s.topic[:40]:40s} | rep={s.repetition} | seed={s.seed}"
            )
        print(f"\nTotal: {len(specs)} generations")
        print(f"\nConfig:")
        print(f"  Model: {config.model.model_id}")
        print(f"  Layers: {config.model.num_layers}")
        print(f"  Heads: {config.model.num_attention_heads} query, {config.model.num_kv_heads} KV")
        print(f"  Hidden dim: {config.model.hidden_dim}")
        print(f"  dtype: {config.model.torch_dtype}")
        print(f"  Temperature: {config.generation.temperature}")
        print(f"  EOS tokens: {config.generation.eos_token_ids}")
        print(f"  Sampled layers: {config.extraction.sampled_layers}")
        print(f"  PCA layers: {config.extraction.pca_layers}")
        print(f"  Output dir: {config.outputs_dir}")

        # Estimate feature counts
        n_layers = config.model.num_layers
        n_sampled = len(config.extraction.sampled_layers)
        n_pca_layers = len(config.extraction.pca_layers)
        t1 = n_layers * 7 + 25  # per-layer norms + logit stats + token dynamics
        t2 = n_layers * 2 + n_layers * 2 + (n_layers - 1) * 3 + n_sampled * 4
        t2_5 = n_sampled * 9 + n_sampled * 10 + 3 + 9
        t3 = n_pca_layers * 5 * 50 if not args.no_tier3 else 0
        print(f"\nExpected features: T1={t1}, T2={t2}, T2.5={t2_5}, T3={t3}")
        print(f"  Total: {t1 + t2 + t2_5 + t3}")
        return

    # Load calibration data
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

    # Load PCA model for Tier 3
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

    # Load model
    logger.info(f"Loading {config.model.model_id}...")
    loaded = load_model(config.model, sampled_layers=config.extraction.sampled_layers)
    logger.info("Model loaded successfully")

    # Verify model architecture matches config
    actual_layers = len(loaded.model.model.layers)
    if actual_layers != config.model.num_layers:
        logger.error(
            f"Model has {actual_layers} layers but config expects {config.model.num_layers}!"
        )
        sys.exit(1)
    logger.info(f"Architecture verified: {actual_layers} layers")

    # Run
    metadata = run_experiment(
        loaded=loaded,
        config=config,
        positional_means=positional_means,
        pca_components=pca_components,
        pca_mean=pca_mean,
        specs=specs,
    )

    logger.info(f"Done! {len(metadata)} generations completed.")

    # Cleanup
    loaded.remove_hooks()


if __name__ == "__main__":
    main()
