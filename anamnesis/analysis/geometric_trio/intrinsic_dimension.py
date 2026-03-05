"""
Intrinsic dimension profiling of Run 4 computational signatures.

Estimates the intrinsic dimensionality of the mode-signature manifold
using TwoNN (Facco et al. 2017) and GRIDE multiscale (Denti et al. 2022).

Analyses:
  1. Global ID: all 100 samples pooled, per tier/group
  2. Per-mode ID: 20 samples per mode, per tier/group
  3. Bootstrap confidence intervals for per-mode estimates
  4. GRIDE multiscale: ID as a function of neighborhood size

Literature context:
  - Ansuini et al. 2019: PCA-based ID completely fails; only nonlinear
    estimators (TwoNN) capture manifold structure
  - Yin et al. 2024: LID differs between truthful/hallucinated outputs
  - Viswanathan et al. 2025: ID correlates with cross-entropy loss
  - Song et al. 2025: early layers expand, late layers compress

Pre-registered predictions:
  P1. T2.5 ID < T1 ID (structured strategy < noisy surface stats)
  P2. Per-mode ID should differ across modes
  P3. Analogical ID might be lowest (most geometrically distinct)
  P4. T2+T2.5 pooled ID < combined pooled ID
  P5. Overall ID in T2+T2.5 space: ~5-15
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dadapy import Data as DADAData
from numpy.typing import NDArray
from skdim.id import MLE, TwoNN

from .data_loader import Run4Data, load_run4

# All feature spaces to evaluate
TIERS_AND_GROUPS = ["T1", "T2", "T2.5", "T3", "T2+T2.5", "combined"]


@dataclass
class IDEstimate:
    """Single intrinsic dimension estimate with metadata."""
    subset_label: str  # e.g., "analogical" or "all"
    feature_space: str  # e.g., "T2+T2.5"
    n_samples: int
    ambient_dim: int
    # TwoNN
    twonn_id: float
    twonn_err: float  # from DADApy (analytical)
    # MLE (Levina-Bickel)
    mle_id: float
    # Bootstrap CIs (TwoNN)
    bootstrap_mean: float | None = None
    bootstrap_std: float | None = None
    bootstrap_ci_lo: float | None = None
    bootstrap_ci_hi: float | None = None
    # GRIDE multiscale profile
    gride_ids: NDArray | None = None  # ID at each scale
    gride_errs: NDArray | None = None  # errors at each scale
    gride_scales: NDArray | None = None  # neighborhood fractions


@dataclass
class IDResults:
    """Complete ID profiling results."""
    global_estimates: list[IDEstimate]
    per_mode_estimates: dict[str, list[IDEstimate]]  # mode -> list over tiers
    n_bootstrap: int
    predictions: dict[str, dict]  # prediction outcomes


def _standardize(X: NDArray[np.float32]) -> NDArray[np.float64]:
    """Z-score standardize features. Required for meaningful distance-based ID."""
    X = X.astype(np.float64)
    std = X.std(axis=0)
    # Avoid division by zero for constant features
    std[std < 1e-12] = 1.0
    return (X - X.mean(axis=0)) / std


def _remove_constant_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove features with zero variance (constant across samples)."""
    variance = X.var(axis=0)
    mask = variance > 1e-12
    return X[:, mask]


def estimate_id(
    X: NDArray[np.float32],
    label: str,
    feature_space: str,
    n_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
) -> IDEstimate:
    """
    Estimate intrinsic dimension using TwoNN, MLE, and GRIDE.

    Parameters
    ----------
    X : (N, D) feature matrix
    label : subset identifier (e.g., mode name or "all")
    feature_space : tier/group name
    n_bootstrap : number of bootstrap resamples for CIs
    rng : random number generator
    """
    if rng is None:
        rng = np.random.default_rng(42)

    X_std = _standardize(X)
    X_clean = _remove_constant_features(X_std)
    n, d = X_clean.shape

    # --- TwoNN via DADApy (more robust, gives analytical error) ---
    try:
        dada = DADAData(X_clean)
        dada.compute_id_2NN()
        twonn_id = float(dada.intrinsic_dim)
        twonn_err = float(dada.intrinsic_dim_err) if dada.intrinsic_dim_err is not None else 0.0
    except Exception as e:
        print(f"  DADApy TwoNN failed for {label}/{feature_space} (N={n}, D={d}): {e}")
        twonn_id = float("nan")
        twonn_err = float("nan")

    # --- MLE (Levina-Bickel) via scikit-dimension ---
    try:
        mle = MLE()
        mle.fit(X_clean)
        mle_id = float(mle.dimension_)
    except Exception as e:
        print(f"  MLE failed for {label}/{feature_space} (N={n}, D={d}): {e}")
        mle_id = float("nan")

    # --- Subsampling TwoNN for confidence intervals ---
    # NOTE: We use subsampling WITHOUT replacement (jackknife-style) rather
    # than bootstrap WITH replacement because TwoNN depends on nearest-neighbor
    # distances. Duplicate points from bootstrap resampling collapse distances
    # to zero, producing degenerate ID estimates (often near 0). Subsampling
    # 80% of points avoids this while still estimating variance.
    boot_ids: list[float] = []
    subsample_frac = 0.8
    if n >= 10 and n_bootstrap > 0:
        subsample_size = max(8, int(n * subsample_frac))
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=subsample_size, replace=False)
            X_boot = X_clean[idx]
            try:
                dada_boot = DADAData(X_boot)
                dada_boot.compute_id_2NN()
                if dada_boot.intrinsic_dim is not None:
                    boot_ids.append(float(dada_boot.intrinsic_dim))
            except Exception:
                continue

    bootstrap_mean = float(np.mean(boot_ids)) if boot_ids else None
    bootstrap_std = float(np.std(boot_ids)) if boot_ids else None
    bootstrap_ci_lo = float(np.percentile(boot_ids, 2.5)) if boot_ids else None
    bootstrap_ci_hi = float(np.percentile(boot_ids, 97.5)) if boot_ids else None

    # --- GRIDE multiscale ---
    gride_ids = None
    gride_errs = None
    gride_scales = None
    if n >= 15:
        try:
            dada_g = DADAData(X_clean)
            result = dada_g.return_id_scaling_gride()
            if result is not None and len(result) == 3:
                gride_ids = np.array(result[0])
                gride_errs = np.array(result[1])
                gride_scales = np.array(result[2])
        except Exception as e:
            print(f"  GRIDE failed for {label}/{feature_space}: {e}")

    return IDEstimate(
        subset_label=label,
        feature_space=feature_space,
        n_samples=n,
        ambient_dim=d,
        twonn_id=twonn_id,
        twonn_err=twonn_err,
        mle_id=mle_id,
        bootstrap_mean=bootstrap_mean,
        bootstrap_std=bootstrap_std,
        bootstrap_ci_lo=bootstrap_ci_lo,
        bootstrap_ci_hi=bootstrap_ci_hi,
        gride_ids=gride_ids,
        gride_errs=gride_errs,
        gride_scales=gride_scales,
    )


def run_id_profiling(
    data: Run4Data,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> IDResults:
    """
    Run full intrinsic dimension profiling.

    Computes global and per-mode ID estimates across all tiers and groups.
    """
    rng = np.random.default_rng(seed)
    global_estimates: list[IDEstimate] = []
    per_mode_estimates: dict[str, list[IDEstimate]] = {}

    for tier in TIERS_AND_GROUPS:
        X = data.get_tier(tier)
        print(f"\n{'='*60}")
        print(f"Feature space: {tier} (D={X.shape[1]})")
        print(f"{'='*60}")

        # Global estimate (all 100 samples)
        print(f"  Global (N={X.shape[0]})...")
        est = estimate_id(X, "all", tier, n_bootstrap=n_bootstrap, rng=rng)
        global_estimates.append(est)
        _print_estimate(est)

        # Per-mode estimates (20 samples each)
        for mode in data.unique_modes:
            mask = data.mode_mask(mode)
            X_mode = X[mask]
            print(f"  {mode} (N={X_mode.shape[0]})...")
            est = estimate_id(X_mode, mode, tier, n_bootstrap=n_bootstrap, rng=rng)
            per_mode_estimates.setdefault(mode, []).append(est)
            _print_estimate(est)

    # Evaluate predictions
    predictions = _evaluate_predictions(global_estimates, per_mode_estimates)

    return IDResults(
        global_estimates=global_estimates,
        per_mode_estimates=per_mode_estimates,
        n_bootstrap=n_bootstrap,
        predictions=predictions,
    )


def _print_estimate(est: IDEstimate) -> None:
    """Print a single ID estimate."""
    ci_str = ""
    if est.bootstrap_ci_lo is not None:
        ci_str = f"  95% CI: [{est.bootstrap_ci_lo:.1f}, {est.bootstrap_ci_hi:.1f}]"
    gride_str = ""
    if est.gride_ids is not None:
        gride_str = f"  GRIDE: {np.array2string(est.gride_ids, precision=1)}"
    print(f"    TwoNN: {est.twonn_id:.2f} ± {est.twonn_err:.2f} | "
          f"MLE: {est.mle_id:.2f} | "
          f"N={est.n_samples}, D={est.ambient_dim}"
          f"{ci_str}{gride_str}")


def _evaluate_predictions(
    global_ests: list[IDEstimate],
    per_mode_ests: dict[str, list[IDEstimate]],
) -> dict[str, dict]:
    """
    Evaluate pre-registered predictions against results.

    Returns dict with prediction outcomes, evidence, and assessment.
    """
    predictions: dict[str, dict] = {}

    # Helper: get global estimate for a tier
    def global_for(tier: str) -> IDEstimate | None:
        return next((e for e in global_ests if e.feature_space == tier), None)

    # Helper: get per-mode estimate for a mode in a tier
    def mode_for(mode: str, tier: str) -> IDEstimate | None:
        return next(
            (e for e in per_mode_ests.get(mode, []) if e.feature_space == tier),
            None,
        )

    # P1: T2.5 ID < T1 ID
    t25_global = global_for("T2.5")
    t1_global = global_for("T1")
    if t25_global and t1_global:
        predictions["P1_T25_lt_T1"] = {
            "prediction": "T2.5 global ID < T1 global ID",
            "T2.5_ID": t25_global.twonn_id,
            "T1_ID": t1_global.twonn_id,
            "confirmed": t25_global.twonn_id < t1_global.twonn_id,
            "margin": t1_global.twonn_id - t25_global.twonn_id,
        }

    # P2: Per-mode ID should differ
    t2t25_key = "T2+T2.5"
    mode_ids = {}
    for mode in per_mode_ests:
        est = mode_for(mode, t2t25_key)
        if est:
            mode_ids[mode] = est.twonn_id
    if mode_ids:
        id_values = list(mode_ids.values())
        predictions["P2_mode_ID_differs"] = {
            "prediction": "Per-mode IDs differ in T2+T2.5 space",
            "mode_IDs": mode_ids,
            "range": max(id_values) - min(id_values),
            "std": float(np.std(id_values)),
            "assessment": "Need bootstrap CIs to assess significance",
        }

    # P3: Analogical ID is lowest
    if mode_ids:
        anal_id = mode_ids.get("analogical")
        if anal_id is not None:
            min_mode = min(mode_ids, key=mode_ids.get)  # type: ignore[arg-type]
            predictions["P3_analogical_lowest"] = {
                "prediction": "Analogical has lowest ID in T2+T2.5",
                "analogical_ID": anal_id,
                "lowest_mode": min_mode,
                "lowest_ID": mode_ids[min_mode],
                "confirmed": min_mode == "analogical",
            }

    # P4: T2+T2.5 ID < combined ID
    t2t25_global = global_for("T2+T2.5")
    combined_global = global_for("combined")
    if t2t25_global and combined_global:
        predictions["P4_T2T25_lt_combined"] = {
            "prediction": "T2+T2.5 global ID < combined global ID",
            "T2+T2.5_ID": t2t25_global.twonn_id,
            "combined_ID": combined_global.twonn_id,
            "confirmed": t2t25_global.twonn_id < combined_global.twonn_id,
            "margin": combined_global.twonn_id - t2t25_global.twonn_id,
        }

    # P5: Overall ID in T2+T2.5 ~ 5-15
    if t2t25_global:
        predictions["P5_T2T25_range"] = {
            "prediction": "T2+T2.5 global ID in range [5, 15]",
            "actual_ID": t2t25_global.twonn_id,
            "in_range": 5 <= t2t25_global.twonn_id <= 15,
        }

    return predictions


def save_results(results: IDResults, output_dir: Path) -> None:
    """Save results to JSON and numpy files."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    # Global estimates summary
    global_summary = []
    for est in results.global_estimates:
        entry = {
            "subset": est.subset_label,
            "feature_space": est.feature_space,
            "n_samples": est.n_samples,
            "ambient_dim": est.ambient_dim,
            "twonn_id": est.twonn_id,
            "twonn_err": est.twonn_err,
            "mle_id": est.mle_id,
            "bootstrap_mean": est.bootstrap_mean,
            "bootstrap_std": est.bootstrap_std,
            "bootstrap_ci_lo": est.bootstrap_ci_lo,
            "bootstrap_ci_hi": est.bootstrap_ci_hi,
        }
        if est.gride_ids is not None:
            entry["gride_ids"] = est.gride_ids.tolist()
            entry["gride_errs"] = est.gride_errs.tolist()
            entry["gride_scales"] = est.gride_scales.tolist()
        global_summary.append(entry)

    # Per-mode estimates
    per_mode_summary: dict[str, list[dict]] = {}
    for mode, ests in results.per_mode_estimates.items():
        per_mode_summary[mode] = []
        for est in ests:
            entry = {
                "feature_space": est.feature_space,
                "n_samples": est.n_samples,
                "ambient_dim": est.ambient_dim,
                "twonn_id": est.twonn_id,
                "twonn_err": est.twonn_err,
                "mle_id": est.mle_id,
                "bootstrap_mean": est.bootstrap_mean,
                "bootstrap_std": est.bootstrap_std,
                "bootstrap_ci_lo": est.bootstrap_ci_lo,
                "bootstrap_ci_hi": est.bootstrap_ci_hi,
            }
            if est.gride_ids is not None:
                entry["gride_ids"] = est.gride_ids.tolist()
                entry["gride_errs"] = est.gride_errs.tolist()
                entry["gride_scales"] = est.gride_scales.tolist()
            per_mode_summary[mode].append(entry)

    # Convert NaN to None for JSON serialization
    def _clean_for_json(obj: object) -> object:
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_for_json(v) for v in obj]
        return obj

    full_output = {
        "n_bootstrap": results.n_bootstrap,
        "global_estimates": _clean_for_json(global_summary),
        "per_mode_estimates": _clean_for_json(per_mode_summary),
        "predictions": _clean_for_json(results.predictions),
    }

    with open(output_dir / "intrinsic_dimension_results.json", "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'intrinsic_dimension_results.json'}")


if __name__ == "__main__":
    print("Loading Run 4 data...")
    data = load_run4()
    print(f"Loaded {data.n_samples} samples, {len(data.unique_modes)} modes")

    print("\nRunning intrinsic dimension profiling...")
    results = run_id_profiling(data, n_bootstrap=200, seed=42)

    # Print prediction outcomes
    print("\n" + "=" * 60)
    print("PREDICTION OUTCOMES")
    print("=" * 60)
    for name, pred in results.predictions.items():
        print(f"\n{name}:")
        for k, v in pred.items():
            print(f"  {k}: {v}")

    # Save
    output_dir = Path(__file__).parent / "results"
    save_results(results, output_dir)
