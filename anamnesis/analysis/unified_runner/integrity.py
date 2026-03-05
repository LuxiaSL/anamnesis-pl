"""Section 1: Data integrity and descriptive statistics."""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import NDArray

from .data_loading import AnalysisData
from .utils import ALL_TIERS, get_available_tiers


def run_integrity_checks(data: AnalysisData) -> dict:
    """Run all data integrity and descriptive checks."""
    results: dict = {}

    # Basic counts
    results["n_samples"] = data.n_samples
    results["n_modes"] = len(data.unique_modes)
    results["n_topics"] = len(data.unique_topics)
    results["modes"] = data.unique_modes
    results["topics"] = data.unique_topics
    results["samples_per_mode"] = {
        m: int(np.sum(data.mode_mask(m))) for m in data.unique_modes
    }
    results["samples_per_topic"] = {
        t: int(np.sum(data.topic_mask(t))) for t in data.unique_topics
    }
    results["balanced"] = len(set(results["samples_per_mode"].values())) == 1

    available_tiers, _ = get_available_tiers(data)

    # Feature dimensions
    tier_dims: dict[str, int] = {}
    for tier in available_tiers:
        try:
            tier_dims[tier] = data.get_tier(tier).shape[1]
        except KeyError:
            tier_dims[tier] = 0
    results["tier_dims"] = tier_dims
    results["total_features"] = sum(
        tier_dims.get(t, 0) for t in ["T1", "T2", "T2.5", "T3"]
    )

    # NaN / Inf checks
    nan_inf: dict[str, dict[str, int]] = {}
    all_clean = True
    for tier in available_tiers:
        try:
            X = data.get_tier(tier)
            nan_count = int(np.sum(np.isnan(X)))
            inf_count = int(np.sum(np.isinf(X)))
            nan_inf[tier] = {"nan": nan_count, "inf": inf_count}
            if nan_count > 0 or inf_count > 0:
                all_clean = False
        except KeyError:
            nan_inf[tier] = {"nan": -1, "inf": -1, "error": "tier not found"}
    results["nan_inf"] = nan_inf
    results["all_clean"] = all_clean

    # Per-feature variance (detect constant features)
    variance_report: dict[str, dict[str, int]] = {}
    for tier in available_tiers:
        try:
            X = data.get_tier(tier)
            var = X.var(axis=0)
            n_const = int(np.sum(var < 1e-12))
            n_near_const = int(np.sum(var < 1e-6))
            variance_report[tier] = {
                "n_features": X.shape[1],
                "n_constant": n_const,
                "n_near_constant": n_near_const,
            }
        except KeyError:
            pass
    results["variance_report"] = variance_report

    # Generation length distribution by mode
    if data.generation_lengths is not None:
        length_by_mode: dict[str, dict[str, float]] = {}
        for mode in data.unique_modes:
            mask = data.mode_mask(mode)
            lengths = data.generation_lengths[mask]
            length_by_mode[mode] = {
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "median": float(np.median(lengths)),
            }
        results["length_by_mode"] = length_by_mode

        # Overall length stats
        all_lengths = data.generation_lengths
        results["length_overall"] = {
            "mean": float(np.mean(all_lengths)),
            "std": float(np.std(all_lengths)),
            "min": int(np.min(all_lengths)),
            "max": int(np.max(all_lengths)),
        }

    # Feature value range summary per tier
    value_ranges: dict[str, dict[str, float]] = {}
    for tier in available_tiers:
        try:
            X = data.get_tier(tier)
            value_ranges[tier] = {
                "global_mean": float(np.mean(X)),
                "global_std": float(np.std(X)),
                "global_min": float(np.min(X)),
                "global_max": float(np.max(X)),
                "feature_mean_range": [
                    float(np.min(X.mean(axis=0))),
                    float(np.max(X.mean(axis=0))),
                ],
            }
        except KeyError:
            pass
    results["value_ranges"] = value_ranges

    return results
