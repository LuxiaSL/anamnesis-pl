"""T3 Investigation: Why is PCA alive at 8B but dead at 3B?

Runs entirely on existing extracted data (no GPU needed).

Analyses:
  1. Per-layer T3 accuracy breakdown
  2. Per-temporal-sample T3 accuracy
  3. PCA variance structure comparison (3B vs 8B)
  4. PCA direction interpretability (mode-discriminative components)
  5. T3 contrastive projection (nonlinear headroom within T3)
  6. T3 vs T2+T2.5 information overlap

Usage:
    python -m anamnesis.analysis.t3_investigation [--run 8b_baseline] [--run 3b_run4]
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .unified_runner.data_loading import AnalysisData, load_analysis_data
from .unified_runner.utils import clean_for_json


# ── Constants ──────────────────────────────────────────────────────────────────

_RF_KWARGS = dict(n_estimators=100, n_jobs=1)
_N_FOLDS = 5

# Run config: maps run name → (signature_dir, pca_model_path)
_RUN_CONFIGS: dict[str, dict[str, Path]] = {
    "8b_baseline": {
        "sig_dir": Path(__file__).resolve().parents[1]
        / "outputs"
        / "runs"
        / "run_8b_baseline"
        / "signatures",
        "pca_path": Path(__file__).resolve().parents[1]
        / "outputs"
        / "calibration"
        / "llama31_8b"
        / "pca_model.pkl",
    },
    "3b_run4": {
        "sig_dir": Path(__file__).resolve().parents[2]
        / "phase_0"
        / "outputs"
        / "runs"
        / "run4_format_controlled"
        / "signatures",
        "pca_path": Path(__file__).resolve().parents[2]
        / "phase_0"
        / "outputs"
        / "calibration"
        / "pca_model.pkl",
    },
}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "analysis" / "t3_investigation"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rf_accuracy(X: NDArray, y: NDArray, seed: int = 42) -> float:
    """Quick RF 5-fold stratified CV accuracy."""
    skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=seed)
    accs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = RandomForestClassifier(**_RF_KWARGS, random_state=seed)
        clf.fit(X_tr, y[train_idx])
        accs.append(float(accuracy_score(y[test_idx], clf.predict(X_te))))
    return float(np.mean(accs))


def _rf_predictions(
    X: NDArray, y: NDArray, seed: int = 42,
) -> NDArray:
    """Get out-of-fold RF predictions for every sample."""
    skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = RandomForestClassifier(**_RF_KWARGS, random_state=seed)
        clf.fit(X_tr, y[train_idx])
        preds[test_idx] = clf.predict(X_te)
    return preds


def _parse_t3_feature_names(
    names: NDArray,
) -> tuple[list[int], list[int], list[int]]:
    """Parse T3 feature names like 'pca_L8_t0_c0' → (layers, timesteps, components)."""
    layers: list[int] = []
    timesteps: list[int] = []
    components: list[int] = []
    pattern = re.compile(r"pca_L(\d+)_t(\d+)_c(\d+)")
    for name in names:
        m = pattern.match(str(name))
        if m:
            layers.append(int(m.group(1)))
            timesteps.append(int(m.group(2)))
            components.append(int(m.group(3)))
        else:
            layers.append(-1)
            timesteps.append(-1)
            components.append(-1)
    return layers, timesteps, components


# ── Analysis 1: Per-layer T3 accuracy ─────────────────────────────────────────

def analysis_1_per_layer(data: AnalysisData) -> dict:
    """RF accuracy using T3 features from each PCA layer separately."""
    print("  [1/6] Per-layer T3 accuracy...")
    X_t3 = data.get_tier("T3")
    y = data.modes
    names = data.run4.tier_feature_names.get("T3", np.array([]))

    if len(names) != X_t3.shape[1]:
        return {"error": "feature name mismatch", "n_names": len(names), "n_features": X_t3.shape[1]}

    layers_arr, _, _ = _parse_t3_feature_names(names)
    unique_layers = sorted(set(l for l in layers_arr if l >= 0))

    per_layer: dict[str, dict] = {}
    for layer in unique_layers:
        mask = np.array([l == layer for l in layers_arr])
        X_layer = X_t3[:, mask]
        acc = _rf_accuracy(X_layer, y)
        per_layer[f"L{layer}"] = {
            "accuracy": acc,
            "n_features": int(np.sum(mask)),
        }
        print(f"    L{layer}: {acc:.1%} ({int(np.sum(mask))} features)")

    # Also T3 combined for reference
    t3_acc = _rf_accuracy(X_t3, y)
    print(f"    T3 combined: {t3_acc:.1%} ({X_t3.shape[1]} features)")

    # Rank layers
    ranked = sorted(per_layer.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    return {
        "per_layer": per_layer,
        "t3_combined_accuracy": t3_acc,
        "layer_ranking": [{"layer": k, "accuracy": v["accuracy"]} for k, v in ranked],
    }


# ── Analysis 2: Per-temporal-sample T3 accuracy ──────────────────────────────

def analysis_2_temporal(data: AnalysisData) -> dict:
    """RF accuracy using T3 features from each temporal sample separately."""
    print("  [2/6] Per-temporal-sample T3 accuracy...")
    X_t3 = data.get_tier("T3")
    y = data.modes
    names = data.run4.tier_feature_names.get("T3", np.array([]))

    if len(names) != X_t3.shape[1]:
        return {"error": "feature name mismatch"}

    _, timesteps_arr, _ = _parse_t3_feature_names(names)
    unique_timesteps = sorted(set(t for t in timesteps_arr if t >= 0))

    per_timestep: dict[str, dict] = {}
    for ts in unique_timesteps:
        mask = np.array([t == ts for t in timesteps_arr])
        X_ts = X_t3[:, mask]
        acc = _rf_accuracy(X_ts, y)
        per_timestep[f"t{ts}"] = {
            "accuracy": acc,
            "n_features": int(np.sum(mask)),
        }
        print(f"    t{ts}: {acc:.1%} ({int(np.sum(mask))} features)")

    # t0-only vs full T3
    t0_mask = np.array([t == 0 for t in timesteps_arr])
    t0_acc = _rf_accuracy(X_t3[:, t0_mask], y)

    # non-t0 combined
    non_t0_mask = np.array([t > 0 for t in timesteps_arr])
    non_t0_acc = _rf_accuracy(X_t3[:, non_t0_mask], y) if np.sum(non_t0_mask) > 0 else None

    print(f"    t0-only: {t0_acc:.1%}")
    if non_t0_acc is not None:
        print(f"    non-t0 combined: {non_t0_acc:.1%}")

    return {
        "per_timestep": per_timestep,
        "t0_only_accuracy": t0_acc,
        "non_t0_accuracy": non_t0_acc,
        "t0_dominance": t0_acc >= _rf_accuracy(X_t3, y) - 0.03,  # within 3pp
    }


# ── Analysis 3: PCA variance structure ────────────────────────────────────────

def analysis_3_pca_variance(pca_path: Path) -> dict:
    """Analyze PCA model variance structure."""
    print("  [3/6] PCA variance structure...")

    if not pca_path.exists():
        return {"error": f"PCA model not found: {pca_path}"}

    with open(pca_path, "rb") as f:
        pca_data = pickle.load(f)

    # Handle different pkl formats
    if isinstance(pca_data, dict):
        components = pca_data.get("components")
        explained_variance = pca_data.get("explained_variance_ratio")
        mean = pca_data.get("mean")
    else:
        # sklearn PCA object
        components = getattr(pca_data, "components_", None)
        explained_variance = getattr(pca_data, "explained_variance_ratio_", None)
        mean = getattr(pca_data, "mean_", None)

    result: dict = {
        "pca_path": str(pca_path),
    }

    if components is not None:
        result["n_components"] = int(components.shape[0])
        result["input_dim"] = int(components.shape[1])

    if explained_variance is not None:
        ev = np.array(explained_variance, dtype=np.float64)
        result["variance_explained"] = {
            "first_component": float(ev[0]),
            "top_5_cumulative": float(np.sum(ev[:5])),
            "top_10_cumulative": float(np.sum(ev[:10])),
            "top_20_cumulative": float(np.sum(ev[:20])),
            "top_50_cumulative": float(np.sum(ev[:50])) if len(ev) >= 50 else float(np.sum(ev)),
            "per_component": [float(v) for v in ev[:50]],
        }
        print(f"    First component: {ev[0]:.1%}")
        print(f"    Top 10 cumulative: {np.sum(ev[:10]):.1%}")
        print(f"    Top 50 cumulative: {np.sum(ev[:min(50, len(ev))]):.1%}")
    else:
        # If no explained_variance_ratio, compute singular values from components
        result["note"] = "No explained_variance_ratio in pkl; only components available"
        if components is not None:
            # Relative magnitude of components as proxy
            norms = np.linalg.norm(components, axis=1)
            norms_rel = norms / norms.sum()
            result["component_norms_relative"] = [float(v) for v in norms_rel[:50]]
            print(f"    First component norm fraction: {norms_rel[0]:.4f}")

    return result


# ── Analysis 4: PCA direction interpretability ────────────────────────────────

def analysis_4_pca_directions(data: AnalysisData) -> dict:
    """Which PCA components separate which modes?"""
    print("  [4/6] PCA direction interpretability...")
    X_t3 = data.get_tier("T3")
    y = data.modes
    names = data.run4.tier_feature_names.get("T3", np.array([]))

    if len(names) != X_t3.shape[1]:
        return {"error": "feature name mismatch"}

    layers_arr, timesteps_arr, components_arr = _parse_t3_feature_names(names)
    unique_layers = sorted(set(l for l in layers_arr if l >= 0))
    unique_modes = sorted(set(y))

    # For each layer, analyze t0 features (since t0 dominates)
    per_layer: dict[str, dict] = {}
    for layer in unique_layers:
        layer_t0_mask = np.array(
            [l == layer and t == 0 for l, t in zip(layers_arr, timesteps_arr)]
        )
        X_lt0 = X_t3[:, layer_t0_mask]
        n_components = X_lt0.shape[1]

        if n_components == 0:
            continue

        # Mode centroids in PCA space
        centroids: dict[str, NDArray] = {}
        for mode in unique_modes:
            mode_mask = y == mode
            centroids[mode] = np.mean(X_lt0[mode_mask], axis=0)

        # Per-component mode variance (between-class variance)
        overall_mean = np.mean(X_lt0, axis=0)
        between_var = np.zeros(n_components)
        for mode in unique_modes:
            n_mode = np.sum(y == mode)
            diff = centroids[mode] - overall_mean
            between_var += n_mode * diff ** 2
        between_var /= len(y)

        # Total variance per component
        total_var = np.var(X_lt0, axis=0)

        # Mode discrimination ratio: between / total
        # High ratio = this component separates modes
        with np.errstate(divide="ignore", invalid="ignore"):
            disc_ratio = np.where(total_var > 1e-10, between_var / total_var, 0.0)

        # Top discriminative components
        top_disc_idx = np.argsort(disc_ratio)[::-1][:10]

        per_layer[f"L{layer}"] = {
            "n_components": n_components,
            "top_discriminative_components": [
                {
                    "component": int(idx),
                    "discrimination_ratio": float(disc_ratio[idx]),
                    "total_variance": float(total_var[idx]),
                    "between_class_variance": float(between_var[idx]),
                }
                for idx in top_disc_idx
            ],
            "mean_discrimination_ratio": float(np.mean(disc_ratio)),
            "max_discrimination_ratio": float(np.max(disc_ratio)),
            # How many components have discrimination > 0.1?
            "n_discriminative_above_0.1": int(np.sum(disc_ratio > 0.1)),
            "n_discriminative_above_0.2": int(np.sum(disc_ratio > 0.2)),
        }

        # Per-mode spread in top component
        best_comp = top_disc_idx[0]
        mode_vals = {
            mode: float(centroids[mode][best_comp])
            for mode in unique_modes
        }
        per_layer[f"L{layer}"]["best_component_mode_centroids"] = mode_vals

        print(f"    L{layer}: max disc ratio={disc_ratio[top_disc_idx[0]]:.3f}, "
              f"n>0.1={int(np.sum(disc_ratio > 0.1))}/{n_components}")

    return {"per_layer_t0": per_layer}


# ── Analysis 5: T3 contrastive projection ─────────────────────────────────────

def analysis_5_contrastive(data: AnalysisData) -> dict:
    """Run contrastive MLP on T3 features alone."""
    print("  [5/6] T3 contrastive projection...")

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return {"error": "PyTorch not installed — skipping contrastive analysis"}

    from sklearn.neighbors import KNeighborsClassifier as _KNN
    from .unified_runner.contrastive import (
        _train_contrastive_mlp,
        _embed,
        _build_topic_folds,
    )

    X_t3 = data.get_tier("T3")
    y = data.modes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t3)

    folds = _build_topic_folds(data.topics, n_folds=5, seed=42)

    fold_accs: list[float] = []
    fold_sils: list[float] = []

    for i, (train_mask, test_mask) in enumerate(folds):
        X_train = X_scaled[train_mask]
        y_train = y[train_mask]
        X_test = X_scaled[test_mask]
        y_test = y[test_mask]

        try:
            model = _train_contrastive_mlp(
                X_train, y_train,
                hidden_dim=256, bottleneck_dim=32,
                n_epochs=200, seed=42 + i,
            )
            emb_train = _embed(model, X_train)
            emb_test = _embed(model, X_test)

            knn = _KNN(n_neighbors=5, metric="cosine")
            knn.fit(emb_train, y_train)
            preds = knn.predict(emb_test)
            acc = float(accuracy_score(y_test, preds))
            fold_accs.append(acc)

            from sklearn.metrics import silhouette_score
            if len(set(y_test)) > 1:
                sil = float(silhouette_score(emb_test, y_test, metric="cosine"))
                fold_sils.append(sil)
        except Exception as e:
            print(f"    Fold {i} failed: {e}")
            continue

    # Compare to RF
    t3_rf_acc = _rf_accuracy(X_t3, y)

    result = {
        "t3_contrastive_knn": {
            "mean": float(np.mean(fold_accs)) if fold_accs else None,
            "std": float(np.std(fold_accs)) if fold_accs else None,
            "per_fold": fold_accs,
        },
        "t3_contrastive_silhouette": {
            "mean": float(np.mean(fold_sils)) if fold_sils else None,
        },
        "t3_rf_accuracy": t3_rf_acc,
        "nonlinear_headroom": (
            float(np.mean(fold_accs) - t3_rf_acc) if fold_accs else None
        ),
    }

    if fold_accs:
        mlp_acc = np.mean(fold_accs)
        print(f"    T3 contrastive kNN: {mlp_acc:.1%} (vs RF {t3_rf_acc:.1%})")
        print(f"    Nonlinear headroom: {mlp_acc - t3_rf_acc:+.1%}")
    else:
        print("    T3 contrastive failed")

    return result


# ── Analysis 6: T3 vs T2+T2.5 information overlap ────────────────────────────

def analysis_6_information_overlap(data: AnalysisData) -> dict:
    """Do T3 and T2+T2.5 capture the same or different mode information?"""
    print("  [6/6] T3 vs T2+T2.5 information overlap...")
    y = data.modes

    # Get out-of-fold predictions from both
    X_t3 = data.get_tier("T3")
    X_t2t25 = data.get_tier("T2+T2.5")

    preds_t3 = _rf_predictions(X_t3, y)
    preds_t2t25 = _rf_predictions(X_t2t25, y)

    # Agreement metrics
    agreement = float(np.mean(preds_t3 == preds_t2t25))
    kappa = float(cohen_kappa_score(preds_t3, preds_t2t25))

    # Per-mode analysis: which modes does each get right?
    unique_modes = sorted(set(y))
    per_mode: dict[str, dict] = {}
    for mode in unique_modes:
        mode_mask = y == mode
        t3_correct = preds_t3[mode_mask] == y[mode_mask]
        t2t25_correct = preds_t2t25[mode_mask] == y[mode_mask]

        # Both correct, only T3, only T2+T2.5, neither
        both = np.sum(t3_correct & t2t25_correct)
        only_t3 = np.sum(t3_correct & ~t2t25_correct)
        only_t2t25 = np.sum(~t3_correct & t2t25_correct)
        neither = np.sum(~t3_correct & ~t2t25_correct)
        n = int(np.sum(mode_mask))

        per_mode[mode] = {
            "n": n,
            "both_correct": int(both),
            "only_t3_correct": int(only_t3),
            "only_t2t25_correct": int(only_t2t25),
            "neither_correct": int(neither),
            "t3_recall": float(np.mean(t3_correct)),
            "t2t25_recall": float(np.mean(t2t25_correct)),
        }
        print(f"    {mode}: T3={np.mean(t3_correct):.0%}, T2+T2.5={np.mean(t2t25_correct):.0%}, "
              f"both={int(both)}, only_T3={int(only_t3)}, only_T2T25={int(only_t2t25)}")

    # Complementarity: how many samples does T3 get right that T2+T2.5 misses?
    t3_correct_all = preds_t3 == y
    t2t25_correct_all = preds_t2t25 == y
    only_t3_total = int(np.sum(t3_correct_all & ~t2t25_correct_all))
    only_t2t25_total = int(np.sum(~t3_correct_all & t2t25_correct_all))

    # Confusion between T3 and T2+T2.5 predictions
    # (what does T3 predict when T2+T2.5 is wrong, and vice versa?)
    t3_when_t2t25_wrong = preds_t3[~t2t25_correct_all]
    true_when_t2t25_wrong = y[~t2t25_correct_all]
    t3_rescues = int(np.sum(t3_when_t2t25_wrong == true_when_t2t25_wrong))

    t2t25_when_t3_wrong = preds_t2t25[~t3_correct_all]
    true_when_t3_wrong = y[~t3_correct_all]
    t2t25_rescues = int(np.sum(t2t25_when_t3_wrong == true_when_t3_wrong))

    result = {
        "prediction_agreement": agreement,
        "cohen_kappa": kappa,
        "per_mode": per_mode,
        "complementarity": {
            "only_t3_correct": only_t3_total,
            "only_t2t25_correct": only_t2t25_total,
            "t3_rescues_t2t25_errors": t3_rescues,
            "t2t25_rescues_t3_errors": t2t25_rescues,
            "n_samples": int(len(y)),
        },
        "interpretation": (
            "complementary" if only_t3_total >= 5 and only_t2t25_total >= 5
            else "redundant" if kappa > 0.7
            else "weakly_complementary"
        ),
    }

    print(f"    Agreement: {agreement:.1%}, Cohen's kappa: {kappa:.3f}")
    print(f"    Only T3 correct: {only_t3_total}, Only T2+T2.5 correct: {only_t2t25_total}")
    print(f"    T3 rescues T2+T2.5 errors: {t3_rescues}, T2+T2.5 rescues T3 errors: {t2t25_rescues}")

    return result


# ── Main runner ────────────────────────────────────────────────────────────────

def run_t3_investigation(run_name: str) -> dict:
    """Run the full T3 investigation for one model."""
    config = _RUN_CONFIGS.get(run_name)
    if config is None:
        raise ValueError(f"Unknown run: {run_name}. Available: {list(_RUN_CONFIGS)}")

    sig_dir = config["sig_dir"]
    pca_path = config["pca_path"]

    print("=" * 60)
    print(f"T3 INVESTIGATION: {run_name}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    data = load_analysis_data(
        signature_dir=sig_dir,
        run_name=run_name,
        core_only=True,
        load_text=False,
    )
    print(f"  {data.n_samples} samples, {len(data.unique_modes)} modes")

    results: dict = {
        "run_name": run_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": data.n_samples,
    }

    section_times: dict[str, float] = {}

    # 1. Per-layer
    t0 = time.perf_counter()
    results["per_layer_accuracy"] = analysis_1_per_layer(data)
    section_times["per_layer"] = time.perf_counter() - t0

    # 2. Temporal
    t0 = time.perf_counter()
    results["temporal_analysis"] = analysis_2_temporal(data)
    section_times["temporal"] = time.perf_counter() - t0

    # 3. PCA variance
    t0 = time.perf_counter()
    results["pca_variance"] = analysis_3_pca_variance(pca_path)
    section_times["pca_variance"] = time.perf_counter() - t0

    # 4. PCA directions
    t0 = time.perf_counter()
    results["pca_directions"] = analysis_4_pca_directions(data)
    section_times["pca_directions"] = time.perf_counter() - t0

    # 5. Contrastive
    t0 = time.perf_counter()
    results["contrastive"] = analysis_5_contrastive(data)
    section_times["contrastive"] = time.perf_counter() - t0

    # 6. Information overlap
    t0 = time.perf_counter()
    results["information_overlap"] = analysis_6_information_overlap(data)
    section_times["overlap"] = time.perf_counter() - t0

    results["section_times"] = section_times
    total_time = sum(section_times.values())
    print(f"\nTotal time: {total_time:.1f}s")

    return results


def _print_comparison(results_8b: dict | None, results_3b: dict | None) -> None:
    """Print side-by-side comparison if both runs are available."""
    if results_8b is None or results_3b is None:
        return

    print("\n" + "=" * 60)
    print("CROSS-MODEL COMPARISON")
    print("=" * 60)

    # Per-layer comparison
    print("\n--- Per-layer T3 accuracy ---")
    print(f"{'Layer':<10} {'3B':<12} {'8B':<12} {'Delta':<10}")
    layers_3b = results_3b.get("per_layer_accuracy", {}).get("per_layer", {})
    layers_8b = results_8b.get("per_layer_accuracy", {}).get("per_layer", {})
    all_layers = sorted(set(list(layers_3b.keys()) + list(layers_8b.keys())))
    for layer in all_layers:
        acc_3b = layers_3b.get(layer, {}).get("accuracy")
        acc_8b = layers_8b.get(layer, {}).get("accuracy")
        s_3b = f"{acc_3b:.1%}" if acc_3b is not None else "—"
        s_8b = f"{acc_8b:.1%}" if acc_8b is not None else "—"
        delta = ""
        if acc_3b is not None and acc_8b is not None:
            d = acc_8b - acc_3b
            delta = f"{d:+.1%}"
        print(f"{layer:<10} {s_3b:<12} {s_8b:<12} {delta:<10}")

    # Temporal comparison
    print("\n--- Temporal sample accuracy ---")
    ts_3b = results_3b.get("temporal_analysis", {})
    ts_8b = results_8b.get("temporal_analysis", {})
    print(f"  3B t0-only: {ts_3b.get('t0_only_accuracy', '?')}")
    print(f"  8B t0-only: {ts_8b.get('t0_only_accuracy', '?')}")

    # PCA variance
    print("\n--- PCA variance structure ---")
    pv_3b = results_3b.get("pca_variance", {}).get("variance_explained", {})
    pv_8b = results_8b.get("pca_variance", {}).get("variance_explained", {})
    for key in ["first_component", "top_10_cumulative", "top_50_cumulative"]:
        v_3b = pv_3b.get(key)
        v_8b = pv_8b.get(key)
        s_3b = f"{v_3b:.1%}" if v_3b else "?"
        s_8b = f"{v_8b:.1%}" if v_8b else "?"
        print(f"  {key}: 3B={s_3b}, 8B={s_8b}")

    # Contrastive
    print("\n--- T3 contrastive projection ---")
    c_3b = results_3b.get("contrastive", {})
    c_8b = results_8b.get("contrastive", {})
    for label, c in [("3B", c_3b), ("8B", c_8b)]:
        knn = c.get("t3_contrastive_knn", {}).get("mean")
        rf = c.get("t3_rf_accuracy")
        headroom = c.get("nonlinear_headroom")
        if knn is not None and rf is not None:
            print(f"  {label}: MLP kNN={knn:.1%}, RF={rf:.1%}, headroom={headroom:+.1%}")

    # Information overlap
    print("\n--- T3 vs T2+T2.5 complementarity ---")
    for label, r in [("3B", results_3b), ("8B", results_8b)]:
        io = r.get("information_overlap", {})
        comp = io.get("complementarity", {})
        print(f"  {label}: kappa={io.get('cohen_kappa', '?'):.3f}, "
              f"only_T3={comp.get('only_t3_correct', '?')}, "
              f"only_T2T25={comp.get('only_t2t25_correct', '?')}, "
              f"interpretation={io.get('interpretation', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="T3 Investigation")
    parser.add_argument(
        "--run",
        nargs="+",
        default=["8b_baseline", "3b_run4"],
        choices=list(_RUN_CONFIGS.keys()),
        help="Which runs to analyze",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for run_name in args.run:
        try:
            results = run_t3_investigation(run_name)
            all_results[run_name] = results

            # Save per-run results
            out_path = OUTPUT_DIR / f"{run_name}.json"
            with open(out_path, "w") as f:
                json.dump(clean_for_json(results), f, indent=2)
            print(f"\nResults saved to {out_path}")
        except Exception as e:
            print(f"\nERROR running {run_name}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-model comparison
    if len(all_results) >= 2:
        _print_comparison(
            all_results.get("8b_baseline"),
            all_results.get("3b_run4"),
        )

    # Save combined results
    if all_results:
        combined_path = OUTPUT_DIR / "combined_results.json"
        with open(combined_path, "w") as f:
            json.dump(clean_for_json(all_results), f, indent=2)
        print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
