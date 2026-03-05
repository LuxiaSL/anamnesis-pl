"""Section 2: Classification — 5-way mode discrimination."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .data_loading import AnalysisData
from .utils import ALL_TIERS, KEY_TIERS, get_available_tiers

# n_jobs=1 on RF to avoid joblib thread pool deadlocks.
# We parallelize at the outer loop level instead.
_RF_KWARGS = dict(n_estimators=100, n_jobs=1)


def _run_rf_cv(
    X: NDArray, y: NDArray, n_splits: int = 5, seed: int = 42,
    return_confusion: bool = True,
) -> dict:
    """Single RF cross-validation run."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs = []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = RandomForestClassifier(**_RF_KWARGS, random_state=seed)
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)

        fold_accs.append(float(accuracy_score(y[test_idx], y_pred)))
        if return_confusion:
            all_y_true.extend(y[test_idx].tolist())
            all_y_pred.extend(y_pred.tolist())

    result: dict = {
        "accuracy": float(np.mean(fold_accs)),
        "fold_accuracies": fold_accs,
    }
    if return_confusion and all_y_true:
        labels = sorted(set(all_y_true))
        result["confusion_matrix"] = confusion_matrix(
            all_y_true, all_y_pred, labels=labels,
        ).tolist()
        result["labels"] = labels

    return result


def _run_topic_heldout(
    X: NDArray, y: NDArray, topics: NDArray, n_splits: int = 5, seed: int = 42,
) -> dict:
    """GroupKFold by topic."""
    unique_topics = sorted(set(topics))
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    actual_splits = min(n_splits, len(unique_topics))
    gkf = GroupKFold(n_splits=actual_splits)

    fold_accs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = RandomForestClassifier(**_RF_KWARGS, random_state=seed)
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)
        fold_accs.append(float(accuracy_score(y[test_idx], y_pred)))

    return {
        "accuracy": float(np.mean(fold_accs)),
        "fold_accuracies": fold_accs,
        "n_groups": len(unique_topics),
    }


# ── Parallelized helpers for expensive loops ──

def _cv_single_seed(args: tuple) -> float:
    """Worker for CV stability — one seed, returns accuracy."""
    X, y, seed = args
    return _run_rf_cv(X, y, seed=seed, return_confusion=False)["accuracy"]


def _perm_single(args: tuple) -> float:
    """Worker for permutation test — one permuted-label RF, returns accuracy."""
    X, y_perm, seed = args
    return _run_rf_cv(X, y_perm, seed=seed, return_confusion=False)["accuracy"]


def _run_cv_stability(
    X: NDArray, y: NDArray, n_seeds: int = 50,
) -> dict:
    """RF accuracy across multiple CV seeds, parallelized."""
    n_workers = max(1, mp.cpu_count() - 1)
    args_list = [(X, y, seed) for seed in range(n_seeds)]

    accs = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_cv_single_seed, a) for a in args_list]
        for f in as_completed(futures):
            accs.append(f.result())

    return {
        "mean": float(np.mean(accs)),
        "median": float(np.median(accs)),
        "std": float(np.std(accs)),
        "ci_lo": float(np.percentile(accs, 2.5)),
        "ci_hi": float(np.percentile(accs, 97.5)),
        "min": float(np.min(accs)),
        "max": float(np.max(accs)),
        "all_accuracies": sorted(accs),
        "n_seeds": n_seeds,
    }


def _run_permutation_test(
    X: NDArray, y: NDArray, n_permutations: int = 500, seed: int = 42,
) -> dict:
    """Permutation test for RF accuracy, parallelized."""
    observed = _run_rf_cv(X, y, seed=seed, return_confusion=False)["accuracy"]

    rng = np.random.default_rng(seed)
    perm_args = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_args.append((X, y_perm, seed))

    n_workers = max(1, mp.cpu_count() - 1)
    null_accs = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_perm_single, a) for a in perm_args]
        for f in as_completed(futures):
            null_accs.append(f.result())

    null_arr = np.array(null_accs)
    p_value = float(np.mean(null_arr >= observed))

    return {
        "observed_accuracy": observed,
        "p_value": max(p_value, 1.0 / (n_permutations + 1)),
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
        "null_max": float(np.max(null_arr)),
        "null_p95": float(np.percentile(null_arr, 95)),
        "null_p99": float(np.percentile(null_arr, 99)),
        "n_permutations": n_permutations,
    }


def _run_pairwise_binary(
    X: NDArray, y: NDArray, seed: int = 42,
) -> dict:
    """Binary RF for all mode pairs."""
    modes = sorted(set(y))
    results: dict[str, dict] = {}
    for i, m1 in enumerate(modes):
        for m2 in modes[i + 1:]:
            mask = np.isin(y, [m1, m2])
            X_sub = X[mask]
            y_sub = (y[mask] == m1).astype(int)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_accs = []
            for train_idx, test_idx in skf.split(X_sub, y_sub):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_sub[train_idx])
                X_te = scaler.transform(X_sub[test_idx])

                clf = RandomForestClassifier(**_RF_KWARGS, random_state=seed)
                clf.fit(X_tr, y_sub[train_idx])
                y_pred = clf.predict(X_te)
                fold_accs.append(float(accuracy_score(y_sub[test_idx], y_pred)))

            results[f"{m1}_vs_{m2}"] = {
                "accuracy": float(np.mean(fold_accs)),
                "fold_accuracies": fold_accs,
            }

    return results


def _run_4way_no_analogical(
    X: NDArray, y: NDArray, seed: int = 42,
) -> dict:
    """4-way classification excluding analogical."""
    mask = y != "analogical"
    if np.sum(mask) == 0:
        return {"error": "no analogical mode found"}
    return _run_rf_cv(X[mask], y[mask], seed=seed)


def _run_linear_probe(
    X: NDArray, y: NDArray, seed: int = 42,
) -> dict:
    """Logistic regression classification."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accs = []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)
        fold_accs.append(float(accuracy_score(y[test_idx], y_pred)))

    return {
        "accuracy": float(np.mean(fold_accs)),
        "fold_accuracies": fold_accs,
    }


def run_classification(data: AnalysisData) -> dict:
    """Run all classification analyses across tiers."""
    y = data.modes
    topics = data.topics
    results: dict = {}

    available_tiers, key_tiers = get_available_tiers(data)
    for tier in available_tiers:
        print(f"  Classification: {tier}")
        X = data.get_tier(tier)
        tier_results: dict = {}

        # Core: 5-way RF
        tier_results["rf_5way"] = _run_rf_cv(X, y)

        # Topic-heldout
        tier_results["topic_heldout"] = _run_topic_heldout(X, y, topics)

        # Linear probe
        tier_results["linear_probe"] = _run_linear_probe(X, y)

        # Pairwise binary
        tier_results["pairwise_binary"] = _run_pairwise_binary(X, y)

        # 4-way (no analogical)
        tier_results["rf_4way_no_analogical"] = _run_4way_no_analogical(X, y)

        # Expensive tests only on key tiers
        if tier in key_tiers:
            print(f"    CV stability ({tier})...")
            tier_results["cv_stability"] = _run_cv_stability(X, y, n_seeds=100)

            print(f"    Permutation test ({tier})...")
            tier_results["permutation_test"] = _run_permutation_test(
                X, y, n_permutations=1000,
            )

        results[tier] = tier_results
        print(f"    Done: {tier} RF={tier_results['rf_5way']['accuracy']:.1%}")

    # Length-only confound check
    print("  Length-only baseline...")
    results["length_only"] = _run_length_only_baseline(data, y)

    return results


def _run_length_only_baseline(data: AnalysisData, y: NDArray) -> dict:
    """RF with generation length as only feature — should be at chance (~20%).

    Critical confound check: if length alone predicts mode, feature-based
    classification is suspect.
    """
    # Try to get token counts from metadata, fall back to word count from text
    lengths: NDArray | None = None
    if hasattr(data, "token_counts") and data.token_counts is not None:
        lengths = np.array(data.token_counts, dtype=float)
    elif hasattr(data, "texts") and data.texts is not None:
        lengths = np.array([len(t.split()) for t in data.texts], dtype=float)
    else:
        # Try extracting from run4 metadata
        try:
            gen_lengths = []
            for i in range(data.n_samples):
                meta = data.run4.metadata[i] if hasattr(data.run4, "metadata") else None
                if meta and "n_tokens" in meta:
                    gen_lengths.append(float(meta["n_tokens"]))
                elif meta and "generation_length" in meta:
                    gen_lengths.append(float(meta["generation_length"]))
                else:
                    gen_lengths.append(float("nan"))
            if not any(np.isnan(gen_lengths)):
                lengths = np.array(gen_lengths)
        except Exception:
            pass

    if lengths is None or np.any(np.isnan(lengths)):
        return {"error": "no length data available", "accuracy": None}

    X_length = lengths.reshape(-1, 1)
    result = _run_rf_cv(X_length, y, return_confusion=True)

    # Per-mode length stats for context
    per_mode_lengths: dict[str, dict] = {}
    for mode in sorted(set(y)):
        mask = y == mode
        mode_lens = lengths[mask]
        per_mode_lengths[mode] = {
            "mean": float(np.mean(mode_lens)),
            "std": float(np.std(mode_lens)),
            "min": float(np.min(mode_lens)),
            "max": float(np.max(mode_lens)),
        }
    result["per_mode_lengths"] = per_mode_lengths

    return result
