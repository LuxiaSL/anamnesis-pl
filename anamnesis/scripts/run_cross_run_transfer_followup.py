#!/usr/bin/env python3
"""Follow-up analyses for the 8B cross-run functional transfer experiment.

Three subcommands, each implementing one of the three experiments described in
`research/notes/HANDOFF-8b-cross-run-followup.md`:

  per-tier   Run forward + reverse transfer + LDA on feature subsets corresponding
             to the four tiers (T1, T2, T2.5, T3), plus T2+T2.5 combined.

  per-layer  Run forward + reverse transfer + LDA on feature subsets corresponding
             to layer depth buckets (early / middle / late), plus each of the 7
             sampled layers individually.

  joint-mlp  Train a single 10-class contrastive MLP on stacked R2+R3 samples
             (no attractor asymmetry between train/test), compute a 10x10
             centroid cosine matrix in the learned 32-d space, and flag the four
             pre-registered functional pairs.

Outputs land under `outputs/analysis/8b_cross_run_transfer_followup/<subcommand>/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from anamnesis.scripts.run_cross_run_transfer import (
    FORWARD_MAPPING,
    FORWARD_WILDCARD,
    OUTPUTS_BASE,
    REVERSE_MAPPING,
    REVERSE_WILDCARD,
    ProjectionNet,
    build_layer_indices,
    compute_centroids,
    compute_similarity_matrix,
    embed_with_model,
    load_feature_names,
    load_run_features,
    run_lda_direction_test,
    run_transfer_multi_seed,
    train_full_data_mlp,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

DEFAULT_TRAIN = "run_8b_baseline"       # paper Run 3 (format-controlled)
DEFAULT_TEST = "run_8b_r2_equivalent"   # paper Run 2 (format-free)
DEFAULT_OUTPUT_ROOT = OUTPUTS_BASE / "analysis" / "8b_cross_run_transfer_followup"


# ── Shared: run forward + reverse + LDA for a given feature config ────────────


@dataclass
class FeatureConfig:
    """Describes how to load features for one analysis run."""

    name: str
    feature_key: str | list[str] = "features"
    feature_indices: list[int] | None = None
    description: str = ""


def run_full_transfer_analysis(
    cfg: FeatureConfig,
    train_run: str,
    test_run: str,
    n_seeds: int,
    bottleneck_dim: int,
    n_epochs: int,
) -> dict[str, Any]:
    """Forward + reverse transfer + LDA for one feature config. Returns a dict."""
    logger.info(f"\n=== [{cfg.name}] {cfg.description} ===")

    X_r3, y_r3, modes_r3, _ = load_run_features(
        train_run, feature_key=cfg.feature_key, feature_indices=cfg.feature_indices,
    )
    X_r2, y_r2, modes_r2, _ = load_run_features(
        test_run, feature_key=cfg.feature_key, feature_indices=cfg.feature_indices,
    )
    if X_r3.shape[1] != X_r2.shape[1]:
        raise RuntimeError(
            f"Feature dim mismatch: {train_run}={X_r3.shape[1]}, {test_run}={X_r2.shape[1]}"
        )

    # --- Forward: train R3, embed R2 ---
    scaler_fwd = StandardScaler()
    X_r3_fwd = scaler_fwd.fit_transform(X_r3)
    X_r2_fwd = scaler_fwd.transform(X_r2)
    fwd = run_transfer_multi_seed(
        X_train=X_r3_fwd.astype(np.float32), y_train=y_r3, modes_train=modes_r3,
        X_test=X_r2_fwd.astype(np.float32), y_test=y_r2, modes_test=modes_r2,
        predicted=FORWARD_MAPPING, wildcard_mode=FORWARD_WILDCARD,
        direction_label=f"forward[{cfg.name}]",
        n_seeds=n_seeds, bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
    )

    # --- Reverse: train R2, embed R3 ---
    scaler_rev = StandardScaler()
    X_r2_rev = scaler_rev.fit_transform(X_r2)
    X_r3_rev = scaler_rev.transform(X_r3)
    rev = run_transfer_multi_seed(
        X_train=X_r2_rev.astype(np.float32), y_train=y_r2, modes_train=modes_r2,
        X_test=X_r3_rev.astype(np.float32), y_test=y_r3, modes_test=modes_r3,
        predicted=REVERSE_MAPPING, wildcard_mode=REVERSE_WILDCARD,
        direction_label=f"reverse[{cfg.name}]",
        n_seeds=n_seeds, bottleneck_dim=bottleneck_dim, n_epochs=n_epochs,
    )

    # --- LDA direction test ---
    lda_results = run_lda_direction_test(
        X_r2.astype(np.float32), y_r2, modes_r2,
        X_r3.astype(np.float32), y_r3, modes_r3,
    )

    return {
        "config_name": cfg.name,
        "description": cfg.description,
        "feature_key": cfg.feature_key,
        "n_features": int(X_r3.shape[1]),
        "train_run": train_run,
        "test_run": test_run,
        "n_train": int(X_r3.shape[0]),
        "n_test": int(X_r2.shape[0]),
        "n_seeds": n_seeds,
        "bottleneck_dim": bottleneck_dim,
        "n_epochs": n_epochs,
        "transfer_forward": fwd,
        "transfer_reverse": rev,
        "lda_direction_test": lda_results,
    }


def extract_summary_row(analysis: dict[str, Any]) -> dict[str, Any]:
    """Pull out headline numbers for a per-config summary table."""
    fwd = analysis["transfer_forward"]
    rev = analysis["transfer_reverse"]
    lda = analysis["lda_direction_test"]
    return {
        "name": analysis["config_name"],
        "n_features": analysis["n_features"],
        "fwd_aggregate_mean": fwd["mapping_accuracy_mean"],
        "fwd_aggregate_std": fwd["mapping_accuracy_std"],
        "rev_aggregate_mean": rev["mapping_accuracy_mean"],
        "rev_aggregate_std": rev["mapping_accuracy_std"],
        "fwd_pedagogical_to_socratic_mean": (
            fwd["per_mode_accuracy"].get("pedagogical", {}).get("mean", float("nan"))
        ),
        "fwd_pedagogical_to_socratic_std": (
            fwd["per_mode_accuracy"].get("pedagogical", {}).get("std", float("nan"))
        ),
        "rev_dialectical_to_deliberative_mean": (
            rev["per_mode_accuracy"].get("dialectical", {}).get("mean", float("nan"))
        ),
        "rev_dialectical_to_deliberative_std": (
            rev["per_mode_accuracy"].get("dialectical", {}).get("std", float("nan"))
        ),
        "fwd_associative_to_analogical_mean": (
            fwd["per_mode_accuracy"].get("associative", {}).get("mean", float("nan"))
        ),
        "rev_analogical_to_associative_mean": (
            rev["per_mode_accuracy"].get("analogical", {}).get("mean", float("nan"))
        ),
        "fwd_structured_to_linear_mean": (
            fwd["per_mode_accuracy"].get("structured", {}).get("mean", float("nan"))
        ),
        "rev_socratic_to_pedagogical_mean": (
            rev["per_mode_accuracy"].get("socratic", {}).get("mean", float("nan"))
        ),
        "lda_r2_sil_own": lda["r2_sil_own_lda"],
        "lda_r3_sil_in_r2": lda["r3_sil_in_r2_lda"],
        "wildcard_compressed_fwd": fwd.get("wildcard_assignments", {}),
        "wildcard_contrastive_rev": rev.get("wildcard_assignments", {}),
    }


# ── Subcommand: per-tier ──────────────────────────────────────────────────────


TIER_CONFIGS: list[FeatureConfig] = [
    FeatureConfig(
        name="tier1", feature_key="features_tier1",
        description="T1 (249 features, activation norms + logit statistics)",
    ),
    FeatureConfig(
        name="tier2", feature_key="features_tier2",
        description="T2 (249 features, attention entropy / head agreement / residual deltas)",
    ),
    FeatureConfig(
        name="tier2_5", feature_key="features_tier2_5",
        description="T2.5 (145 features, KV-cache dynamics)",
    ),
    FeatureConfig(
        name="tier3", feature_key="features_tier3",
        description="T3 (1250 features, PCA projections of residual stream)",
    ),
    FeatureConfig(
        name="tier2_plus_tier2_5",
        feature_key=["features_tier2", "features_tier2_5"],
        description="T2 + T2.5 concatenated (394 features, super-additive pair at 3B)",
    ),
    FeatureConfig(
        name="full_baseline", feature_key="features",
        description="Full 1893-feature baseline (sanity check against main results)",
    ),
]


def cmd_per_tier(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / "per_tier"
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    summaries: list[dict[str, Any]] = []
    for cfg in TIER_CONFIGS:
        analysis = run_full_transfer_analysis(
            cfg, train_run=args.train_run, test_run=args.test_run,
            n_seeds=args.n_seeds, bottleneck_dim=args.bottleneck_dim,
            n_epochs=args.n_epochs,
        )
        out_path = output_dir / f"{cfg.name}.json"
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"  → saved {out_path}")
        summaries.append(extract_summary_row(analysis))
    # Summary table
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "experiment": "per_tier_transfer",
        "elapsed_seconds": float(time.time() - t_start),
        "train_run": args.train_run,
        "test_run": args.test_run,
        "rows": summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n✓ per-tier summary → {summary_path}")
    _print_summary_table(summaries)


# ── Subcommand: per-layer ─────────────────────────────────────────────────────


# All 32 Llama 3.1 8B layers; sampled layers are those with T2.5+T3 density.
# Buckets roughly align with the 8B sampled-layer grid:
#   early   = L0-7   (sampled: 0)            + T1 global features
#   middle  = L8-20  (sampled: 8, 16, 20)
#   late    = L21-31 (sampled: 24, 28, 31)
EARLY_LAYERS = set(range(0, 8))
MIDDLE_LAYERS = set(range(8, 21))
LATE_LAYERS = set(range(21, 32))
SAMPLED_LAYERS = [0, 8, 16, 20, 24, 28, 31]


def build_layer_configs(feature_names: NDArray[np.str_]) -> list[FeatureConfig]:
    configs: list[FeatureConfig] = [
        FeatureConfig(
            name="early_L0_7_with_T1",
            feature_indices=build_layer_indices(
                feature_names, layers=EARLY_LAYERS, include_unlayered=True,
            ),
            description="Early layers 0-7, includes global T1 logit stats",
        ),
        FeatureConfig(
            name="middle_L8_20",
            feature_indices=build_layer_indices(
                feature_names, layers=MIDDLE_LAYERS, include_unlayered=False,
            ),
            description="Middle layers 8-20 (includes sampled 8, 16, 20)",
        ),
        FeatureConfig(
            name="late_L21_31",
            feature_indices=build_layer_indices(
                feature_names, layers=LATE_LAYERS, include_unlayered=False,
            ),
            description="Late layers 21-31 (includes sampled 24, 28, 31)",
        ),
    ]
    # Individual sampled-layer configs (dense T3/T2.5/T2 per layer)
    for lyr in SAMPLED_LAYERS:
        configs.append(
            FeatureConfig(
                name=f"sampled_L{lyr}",
                feature_indices=build_layer_indices(
                    feature_names, layers={lyr}, include_unlayered=False,
                ),
                description=f"Sampled layer {lyr} only (dense T3/T2.5/T2 features)",
            )
        )
    return configs


def cmd_per_layer(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / "per_layer"
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    # Use feature_names from the train run (both runs share pipeline + same layout).
    feature_names = load_feature_names(args.train_run)
    configs = build_layer_configs(feature_names)

    logger.info(f"Per-layer configs ({len(configs)} total):")
    for cfg in configs:
        idx_count = len(cfg.feature_indices) if cfg.feature_indices else 0
        logger.info(f"  {cfg.name:25s} {idx_count:5d} features — {cfg.description}")

    summaries: list[dict[str, Any]] = []
    for cfg in configs:
        analysis = run_full_transfer_analysis(
            cfg, train_run=args.train_run, test_run=args.test_run,
            n_seeds=args.n_seeds, bottleneck_dim=args.bottleneck_dim,
            n_epochs=args.n_epochs,
        )
        out_path = output_dir / f"{cfg.name}.json"
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"  → saved {out_path}")
        summaries.append(extract_summary_row(analysis))

    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "experiment": "per_layer_transfer",
        "elapsed_seconds": float(time.time() - t_start),
        "train_run": args.train_run,
        "test_run": args.test_run,
        "rows": summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n✓ per-layer summary → {summary_path}")
    _print_summary_table(summaries)


# ── Subcommand: joint-MLP ─────────────────────────────────────────────────────


# The 4 pre-registered functional pairs (name-based, direction-agnostic).
PAIR_LABELS = [
    ("pedagogical", "socratic", "interactive explanation"),
    ("deliberative", "dialectical", "propose-challenge-revise"),
    ("associative", "analogical", "connection-driven"),
    ("structured", "linear", "sequential exposition"),
]
WILDCARD_MODES = ["compressed", "contrastive"]


def compute_all_pair_cosines(
    centroids: dict[str, F32], mode_names: list[str]
) -> NDArray[np.float32]:
    """10x10 centroid-cosine matrix."""
    stack = np.stack([centroids[m] for m in mode_names])
    norms = np.linalg.norm(stack, axis=1, keepdims=True) + 1e-10
    unit = stack / norms
    return (unit @ unit.T).astype(np.float32)


def cmd_joint_mlp(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / "joint_mlp"
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Load both runs with full features.
    X_r3, y_r3_local, modes_r3, _ = load_run_features(args.train_run)
    X_r2, y_r2_local, modes_r2, _ = load_run_features(args.test_run)

    # Build joint 10-class label space. Preserve mode-name ordering:
    # R3 modes first (indices 0..4), then R2 modes (5..9).
    joint_mode_names: list[str] = list(modes_r3) + list(modes_r2)
    y_r3 = y_r3_local.astype(np.int64)  # 0..4 already
    y_r2 = y_r2_local.astype(np.int64) + len(modes_r3)  # offset to 5..9

    X = np.concatenate([X_r3, X_r2], axis=0).astype(np.float32)
    y = np.concatenate([y_r3, y_r2]).astype(np.int64)
    logger.info(
        f"Joint dataset: n={X.shape[0]} (R3={X_r3.shape[0]}, R2={X_r2.shape[0]}), "
        f"d={X.shape[1]}, modes={joint_mode_names}"
    )

    # Standardize in joint space (avoids either side dominating).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Multi-seed training
    all_cos_matrices: list[NDArray[np.float32]] = []
    per_seed_pair_cosines: list[dict[str, float]] = []

    for seed_i in range(args.n_seeds):
        torch_seed = seed_i * 7 + 42
        model, final_loss = train_full_data_mlp(
            X_scaled, y,
            bottleneck_dim=args.bottleneck_dim,
            n_epochs=args.n_epochs,
            torch_seed=torch_seed,
        )
        emb = embed_with_model(model, X_scaled)
        centroids = compute_centroids(emb, y, joint_mode_names)
        cos_matrix = compute_all_pair_cosines(centroids, joint_mode_names)
        all_cos_matrices.append(cos_matrix)

        # Per-seed snapshots of the pre-registered pair cosines
        pair_vals: dict[str, float] = {}
        for a, b, _desc in PAIR_LABELS:
            ai = joint_mode_names.index(a)
            bi = joint_mode_names.index(b)
            pair_vals[f"{a}↔{b}"] = float(cos_matrix[ai, bi])
        per_seed_pair_cosines.append(pair_vals)

        logger.info(
            f"  seed {seed_i}: loss={final_loss:.4f} | " +
            " ".join(f"{k}={v:+.3f}" for k, v in pair_vals.items())
        )

    stacked = np.stack(all_cos_matrices)  # (n_seeds, 10, 10)
    median_cos = np.median(stacked, axis=0)
    mean_cos = np.mean(stacked, axis=0)
    std_cos = np.std(stacked, axis=0)

    # Summarize pre-registered pairs + wildcards
    pair_summary: list[dict[str, Any]] = []
    for a, b, desc in PAIR_LABELS:
        ai = joint_mode_names.index(a)
        bi = joint_mode_names.index(b)
        vals = [m[ai, bi] for m in all_cos_matrices]
        pair_summary.append({
            "modes": [a, b],
            "description": desc,
            "cosine_mean": float(np.mean(vals)),
            "cosine_std": float(np.std(vals)),
            "cosine_median": float(np.median(vals)),
            "per_seed": [float(v) for v in vals],
        })

    # Wildcard cross-run neighbors: at which mode does compressed / contrastive
    # sit in the joint space? Report top-3 neighbors (excluding self).
    wildcard_details: dict[str, Any] = {}
    for wm in WILDCARD_MODES:
        if wm not in joint_mode_names:
            continue
        wi = joint_mode_names.index(wm)
        # Neighbors across seeds: median cosine to every other mode
        cos_to_others = median_cos[wi].copy()
        cos_to_others[wi] = -np.inf
        nearest_order = np.argsort(-cos_to_others)
        top3 = []
        for idx in nearest_order[:3]:
            top3.append({
                "mode": joint_mode_names[int(idx)],
                "median_cosine": float(cos_to_others[int(idx)]),
            })
        wildcard_details[wm] = {"top3_neighbors": top3}

    result: dict[str, Any] = {
        "experiment": "joint_mlp",
        "train_run": args.train_run,
        "test_run": args.test_run,
        "n_seeds": args.n_seeds,
        "bottleneck_dim": args.bottleneck_dim,
        "n_epochs": args.n_epochs,
        "joint_mode_names": joint_mode_names,
        "n_r3_samples": int(X_r3.shape[0]),
        "n_r2_samples": int(X_r2.shape[0]),
        "n_features": int(X.shape[1]),
        "cosine_matrix_median": median_cos.tolist(),
        "cosine_matrix_mean": mean_cos.tolist(),
        "cosine_matrix_std": std_cos.tolist(),
        "pair_summary": pair_summary,
        "wildcard_neighbors": wildcard_details,
        "per_seed_pair_cosines": per_seed_pair_cosines,
        "elapsed_seconds": float(time.time() - t_start),
    }
    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n✓ joint-MLP results → {out_path}")

    # Print compact pair table
    logger.info("\n" + "=" * 60)
    logger.info("Pre-registered pair cosines (joint 10-class MLP, median ± std over seeds)")
    logger.info("=" * 60)
    for p in pair_summary:
        a, b = p["modes"]
        logger.info(
            f"  {a:13s} ↔ {b:13s}: {p['cosine_mean']:+.3f} ± {p['cosine_std']:.3f} "
            f"({p['description']})"
        )
    for wm, det in wildcard_details.items():
        logger.info(f"\n  wildcard {wm}: nearest neighbors (median cosine):")
        for nb in det["top3_neighbors"]:
            logger.info(f"    → {nb['mode']:13s}: {nb['median_cosine']:+.3f}")


# ── Utility: compact summary table ────────────────────────────────────────────


def _print_summary_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    logger.info("\n" + "=" * 96)
    logger.info(
        f"{'config':25s} {'#feat':>6s} "
        f"{'fwd':>8s} {'ped→soc':>10s} "
        f"{'rev':>8s} {'dia→del':>10s} "
        f"{'lda_r3_sil':>12s}"
    )
    logger.info("=" * 96)
    for r in rows:
        logger.info(
            f"{r['name']:25s} {r['n_features']:>6d} "
            f"{r['fwd_aggregate_mean']:>7.1%} "
            f"{r['fwd_pedagogical_to_socratic_mean']:>9.1%} "
            f"{r['rev_aggregate_mean']:>7.1%} "
            f"{r['rev_dialectical_to_deliberative_mean']:>9.1%} "
            f"{r['lda_r3_sil_in_r2']:>+12.4f}"
        )


# ── Entry ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Follow-up analyses for 8B cross-run transfer"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for follow-up outputs",
    )
    parser.add_argument("--train-run", type=str, default=DEFAULT_TRAIN)
    parser.add_argument("--test-run", type=str, default=DEFAULT_TEST)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--bottleneck-dim", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=200)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("per-tier", help="Per-tier feature-subset transfer (Experiment 1.3)")
    sub.add_parser("per-layer", help="Per-layer-bucket transfer (Experiment 1.1)")
    sub.add_parser("joint-mlp", help="Joint-trained 10-class MLP (Experiment 1.2)")
    args = parser.parse_args()

    if args.cmd == "per-tier":
        cmd_per_tier(args)
    elif args.cmd == "per-layer":
        cmd_per_layer(args)
    elif args.cmd == "joint-mlp":
        cmd_joint_mlp(args)
    else:
        parser.error(f"Unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
