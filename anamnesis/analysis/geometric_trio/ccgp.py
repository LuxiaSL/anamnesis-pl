"""
Cross-Condition Generalization Performance (CCGP) for Run 4 signatures.

Tests whether the mode axis is genuinely abstract (transfers across topics)
or topic-entangled. Follows Bernardi et al. 2020 (Cell) framework.

Design:
  - "Conditions" = topics (20)
  - "Categories" = modes (5)
  - Train mode classifier on half the topics, test on the other half
  - Test all 15 possible binary dichotomies of 5 modes
  - CCGP score = fraction of dichotomies that transfer

Pre-registered predictions:
  P1. Mode axis transfers across topics (already suggested by 78% topic-heldout)
  P2. Per-mode CCGP: analogical ~100%, linear/socratic 55-60%
  P3. CCGP in T2+T2.5 space > CCGP in combined space
  P4. The bimodal fold distribution is a T1/T3 artifact
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data_loader import Run4Data, load_run4

TIERS_AND_GROUPS = ["T1", "T2", "T2.5", "T3", "T2+T2.5", "combined"]


@dataclass
class DichotomyResult:
    """Result for a single binary dichotomy across topic splits."""
    group_a: tuple[str, ...]  # modes in group A
    group_b: tuple[str, ...]  # modes in group B
    # Per-fold accuracies (one per topic split)
    fold_accuracies: list[float]
    mean_accuracy: float
    std_accuracy: float
    # Is this dichotomy "decodable" (mean accuracy > threshold)?
    decodable: bool
    threshold: float


@dataclass
class CCGPResult:
    """CCGP results for one feature space."""
    feature_space: str
    ambient_dim: int
    n_samples: int
    n_topics: int
    n_modes: int
    # 5-way classification (topic-heldout)
    multiclass_fold_accuracies: list[float]
    multiclass_mean: float
    multiclass_std: float
    # Per-mode recall (topic-heldout, 5-way)
    per_mode_recall: dict[str, float]
    # Shattering dichotomies
    n_dichotomies: int
    n_decodable: int
    ccgp_score: float  # fraction decodable
    dichotomy_results: list[DichotomyResult]
    # Per-fold details for multiclass
    per_fold_confusion: list[NDArray]


@dataclass
class CCGPResults:
    """Complete CCGP analysis across all feature spaces."""
    results: dict[str, CCGPResult]  # feature_space -> result
    predictions: dict[str, dict]


def _generate_topic_folds(
    topics: list[str],
    n_folds: int = 5,
    rng: np.random.Generator | None = None,
) -> list[tuple[list[str], list[str]]]:
    """
    Generate train/test topic splits for CCGP.

    Returns list of (train_topics, test_topics) pairs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    topics_arr = np.array(topics)
    n = len(topics_arr)
    fold_size = n // n_folds

    # Shuffle and split
    perm = rng.permutation(n)
    folds: list[tuple[list[str], list[str]]] = []
    for i in range(n_folds):
        test_idx = perm[i * fold_size : (i + 1) * fold_size]
        train_idx = np.setdiff1d(perm, test_idx)
        folds.append(
            (topics_arr[train_idx].tolist(), topics_arr[test_idx].tolist())
        )
    return folds


def _generate_dichotomies(modes: list[str]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """
    Generate all non-trivial binary dichotomies of modes.

    For 5 modes: 2^5 / 2 - 1 = 15 dichotomies.
    Excludes trivial (all-in-one-group) splits.
    """
    n = len(modes)
    dichotomies: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    # Generate all subsets of size 1 to n//2
    for k in range(1, n // 2 + 1):
        for group_a in combinations(modes, k):
            group_b = tuple(m for m in modes if m not in group_a)
            # Avoid duplicates: for k = n//2, only take one ordering
            if k == n // 2 and group_a > group_b:
                continue
            dichotomies.append((group_a, group_b))
    return dichotomies


def run_ccgp_single(
    X: NDArray[np.float32],
    modes: NDArray,
    topics: NDArray,
    feature_space: str,
    n_folds: int = 5,
    decodable_threshold: float = 0.65,
    seed: int = 42,
) -> CCGPResult:
    """
    Run CCGP analysis for a single feature space.

    Uses kNN classifier (k=3) for robustness at low N.
    Topic-heldout: train on some topics, test on held-out topics.
    """
    rng = np.random.default_rng(seed)
    unique_modes = sorted(set(modes))
    unique_topics = sorted(set(topics))
    n_modes = len(unique_modes)

    topic_folds = _generate_topic_folds(unique_topics, n_folds=n_folds, rng=rng)

    # --- 5-way multiclass topic-heldout ---
    multiclass_accs: list[float] = []
    per_mode_correct: dict[str, int] = {m: 0 for m in unique_modes}
    per_mode_total: dict[str, int] = {m: 0 for m in unique_modes}
    per_fold_confusion: list[NDArray] = []

    for train_topics, test_topics in topic_folds:
        train_mask = np.isin(topics, train_topics)
        test_mask = np.isin(topics, test_topics)

        X_train, y_train = X[train_mask], modes[train_mask]
        X_test, y_test = X[test_mask], modes[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = KNeighborsClassifier(n_neighbors=min(3, len(X_train) // n_modes))
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = float(np.mean(y_pred == y_test))
        multiclass_accs.append(acc)

        # Per-mode recall
        for mode in unique_modes:
            mode_mask = y_test == mode
            per_mode_correct[mode] += int(np.sum(y_pred[mode_mask] == mode))
            per_mode_total[mode] += int(np.sum(mode_mask))

        # Confusion matrix
        cm = np.zeros((n_modes, n_modes), dtype=int)
        mode_to_idx = {m: i for i, m in enumerate(unique_modes)}
        for true, pred in zip(y_test, y_pred):
            cm[mode_to_idx[true], mode_to_idx[pred]] += 1
        per_fold_confusion.append(cm)

    per_mode_recall = {
        m: per_mode_correct[m] / max(per_mode_total[m], 1)
        for m in unique_modes
    }

    # --- Shattering dichotomies ---
    dichotomies = _generate_dichotomies(unique_modes)
    dichotomy_results: list[DichotomyResult] = []

    for group_a, group_b in dichotomies:
        # Binary task: group_a vs group_b
        fold_accs: list[float] = []

        for train_topics, test_topics in topic_folds:
            train_mask = np.isin(topics, train_topics)
            test_mask = np.isin(topics, test_topics)
            mode_mask = np.isin(modes, group_a + group_b)

            train_full = train_mask & mode_mask
            test_full = test_mask & mode_mask

            if np.sum(train_full) < 4 or np.sum(test_full) < 2:
                continue

            X_train, X_test = X[train_full], X[test_full]
            y_train = np.isin(modes[train_full], group_a).astype(int)
            y_test = np.isin(modes[test_full], group_a).astype(int)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_train) // 2)))
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)

            fold_accs.append(float(np.mean(y_pred == y_test)))

        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        std_acc = float(np.std(fold_accs)) if fold_accs else 0.0

        dichotomy_results.append(DichotomyResult(
            group_a=group_a,
            group_b=group_b,
            fold_accuracies=fold_accs,
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
            decodable=mean_acc > decodable_threshold,
            threshold=decodable_threshold,
        ))

    n_decodable = sum(1 for d in dichotomy_results if d.decodable)

    return CCGPResult(
        feature_space=feature_space,
        ambient_dim=X.shape[1],
        n_samples=X.shape[0],
        n_topics=len(unique_topics),
        n_modes=n_modes,
        multiclass_fold_accuracies=multiclass_accs,
        multiclass_mean=float(np.mean(multiclass_accs)),
        multiclass_std=float(np.std(multiclass_accs)),
        per_mode_recall=per_mode_recall,
        n_dichotomies=len(dichotomy_results),
        n_decodable=n_decodable,
        ccgp_score=n_decodable / max(len(dichotomy_results), 1),
        dichotomy_results=dichotomy_results,
        per_fold_confusion=per_fold_confusion,
    )


def run_ccgp(
    data: Run4Data,
    n_folds: int = 5,
    seed: int = 42,
) -> CCGPResults:
    """Run CCGP across all feature spaces."""
    results: dict[str, CCGPResult] = {}

    for tier in TIERS_AND_GROUPS:
        X = data.get_tier(tier)
        print(f"\n{'='*60}")
        print(f"CCGP: {tier} (D={X.shape[1]})")
        print(f"{'='*60}")

        result = run_ccgp_single(
            X, data.modes, data.topics, tier,
            n_folds=n_folds, seed=seed,
        )
        results[tier] = result

        # Print summary
        print(f"  5-way topic-heldout: {result.multiclass_mean:.1%} "
              f"± {result.multiclass_std:.1%}")
        print(f"  Per-fold: {[f'{a:.0%}' for a in result.multiclass_fold_accuracies]}")
        print(f"  Per-mode recall: ", end="")
        for m, r in sorted(result.per_mode_recall.items()):
            print(f"{m}={r:.0%} ", end="")
        print()
        print(f"  Dichotomies: {result.n_decodable}/{result.n_dichotomies} "
              f"decodable (CCGP={result.ccgp_score:.2f})")

        # Top and bottom dichotomies
        sorted_dich = sorted(result.dichotomy_results,
                             key=lambda d: d.mean_accuracy, reverse=True)
        print(f"  Best: {sorted_dich[0].group_a} vs {sorted_dich[0].group_b} "
              f"= {sorted_dich[0].mean_accuracy:.1%}")
        print(f"  Worst: {sorted_dich[-1].group_a} vs {sorted_dich[-1].group_b} "
              f"= {sorted_dich[-1].mean_accuracy:.1%}")

    # Evaluate predictions
    predictions = _evaluate_predictions(results)

    return CCGPResults(results=results, predictions=predictions)


def _evaluate_predictions(results: dict[str, CCGPResult]) -> dict[str, dict]:
    """Evaluate pre-registered CCGP predictions."""
    predictions: dict[str, dict] = {}

    # P1: Mode axis transfers (5-way > 60%)
    t2t25 = results.get("T2+T2.5")
    if t2t25:
        predictions["P1_mode_transfers"] = {
            "prediction": "5-way topic-heldout accuracy > 60% in T2+T2.5",
            "actual": t2t25.multiclass_mean,
            "confirmed": t2t25.multiclass_mean > 0.60,
        }

    # P2: Per-mode CCGP pattern
    if t2t25:
        predictions["P2_per_mode_pattern"] = {
            "prediction": "analogical ~100%, linear/socratic lowest (55-60%)",
            "actual": t2t25.per_mode_recall,
            "analogical_high": t2t25.per_mode_recall.get("analogical", 0) > 0.90,
            "linear_low": t2t25.per_mode_recall.get("linear", 1) < 0.65,
            "socratic_low": t2t25.per_mode_recall.get("socratic", 1) < 0.65,
        }

    # P3: T2+T2.5 CCGP > combined CCGP
    combined = results.get("combined")
    if t2t25 and combined:
        predictions["P3_T2T25_gt_combined"] = {
            "prediction": "CCGP score higher for T2+T2.5 than combined",
            "T2+T2.5_ccgp": t2t25.ccgp_score,
            "combined_ccgp": combined.ccgp_score,
            "T2+T2.5_5way": t2t25.multiclass_mean,
            "combined_5way": combined.multiclass_mean,
            "confirmed_ccgp": t2t25.ccgp_score >= combined.ccgp_score,
            "confirmed_5way": t2t25.multiclass_mean >= combined.multiclass_mean,
        }

    # P4: Bimodal fold distribution is T1/T3 artifact
    t1 = results.get("T1")
    if t2t25 and combined and t1:
        t2t25_range = (max(t2t25.multiclass_fold_accuracies)
                       - min(t2t25.multiclass_fold_accuracies))
        combined_range = (max(combined.multiclass_fold_accuracies)
                          - min(combined.multiclass_fold_accuracies))
        t1_range = (max(t1.multiclass_fold_accuracies)
                    - min(t1.multiclass_fold_accuracies))
        predictions["P4_bimodal_artifact"] = {
            "prediction": "T2+T2.5 fold range << combined/T1 fold range",
            "T2+T2.5_range": t2t25_range,
            "combined_range": combined_range,
            "T1_range": t1_range,
            "confirmed": t2t25_range < combined_range * 0.7,
        }

    return predictions


def save_results(results: CCGPResults, output_dir: Path) -> None:
    """Save CCGP results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output: dict = {"results": {}, "predictions": results.predictions}

    for tier, result in results.results.items():
        tier_out: dict = {
            "feature_space": result.feature_space,
            "ambient_dim": result.ambient_dim,
            "n_samples": result.n_samples,
            "n_topics": result.n_topics,
            "n_modes": result.n_modes,
            "multiclass_mean": result.multiclass_mean,
            "multiclass_std": result.multiclass_std,
            "multiclass_fold_accuracies": result.multiclass_fold_accuracies,
            "per_mode_recall": result.per_mode_recall,
            "n_dichotomies": result.n_dichotomies,
            "n_decodable": result.n_decodable,
            "ccgp_score": result.ccgp_score,
            "dichotomies": [],
        }
        for d in result.dichotomy_results:
            tier_out["dichotomies"].append({
                "group_a": list(d.group_a),
                "group_b": list(d.group_b),
                "mean_accuracy": d.mean_accuracy,
                "std_accuracy": d.std_accuracy,
                "fold_accuracies": d.fold_accuracies,
                "decodable": d.decodable,
            })
        output["results"][tier] = tier_out

    with open(output_dir / "ccgp_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'ccgp_results.json'}")


if __name__ == "__main__":
    print("Loading Run 4 data...")
    data = load_run4()
    print(f"Loaded {data.n_samples} samples, {len(data.unique_modes)} modes, "
          f"{len(data.unique_topics)} topics")

    print("\nRunning CCGP analysis...")
    results = run_ccgp(data, n_folds=5, seed=42)

    # Print predictions
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
