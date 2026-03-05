"""
Independent verification of geometric trio results.

Re-runs ID, CCGP, and delta-hyperbolicity analyses with varied parameters
to check stability of findings. Also performs additional sanity checks.

Outputs verification_run.json with original vs re-run comparisons.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data_loader import Run4Data, check_data_quality, load_run4

OUTPUT_DIR = Path(__file__).parent / "results"


# ──────────────────────────────────────────────────────────────
# Section 0: Data sanity checks
# ──────────────────────────────────────────────────────────────

def run_sanity_checks(data: Run4Data) -> dict:
    """Verify basic data properties match expectations."""
    quality = check_data_quality(data)
    results: dict = {
        "n_samples": data.n_samples,
        "n_samples_expected": 100,
        "samples_match": data.n_samples == 100,
        "n_modes": len(data.unique_modes),
        "modes": data.unique_modes,
        "n_topics": len(data.unique_topics),
        "samples_per_mode": quality["samples_per_mode"],
        "balanced": all(v == 20 for v in quality["samples_per_mode"].values()),
    }

    # Feature dimensions
    tier_dims = {
        name: feat.shape[1] for name, feat in data.tier_features.items()
    }
    expected_dims = {"T1": 221, "T2": 221, "T2.5": 145, "T3": 1250}
    results["tier_dims"] = tier_dims
    results["expected_dims"] = expected_dims
    results["dims_match"] = tier_dims == expected_dims

    # NaN/Inf checks
    nan_inf: dict = {}
    for name, feat in {**data.tier_features, **data.group_features}.items():
        nan_count = int(np.sum(np.isnan(feat)))
        inf_count = int(np.sum(np.isinf(feat)))
        nan_inf[name] = {"nan": nan_count, "inf": inf_count}
    results["nan_inf"] = nan_inf
    results["no_nan_inf"] = all(
        v["nan"] == 0 and v["inf"] == 0 for v in nan_inf.values()
    )

    # Verify combined dimension
    combined_dim = data.all_features.shape[1]
    results["combined_dim"] = combined_dim
    results["combined_dim_expected"] = 1837
    results["combined_dim_match"] = combined_dim == 1837

    return results


# ──────────────────────────────────────────────────────────────
# Section 1: Intrinsic Dimension verification
# ──────────────────────────────────────────────────────────────

def _standardize(X: NDArray) -> NDArray[np.float64]:
    X = X.astype(np.float64)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return (X - X.mean(axis=0)) / std


def _remove_constant(X: NDArray[np.float64]) -> NDArray[np.float64]:
    variance = X.var(axis=0)
    return X[:, variance > 1e-12]


def run_id_verification(data: Run4Data) -> dict:
    """
    Verify ID estimates with multiple seeds, subsample fractions,
    and a second TwoNN implementation (skdim).
    """
    from dadapy import Data as DADAData
    from skdim.id import TwoNN as SkdimTwoNN

    seeds = [42, 123, 777]
    subsample_fracs = [0.7, 0.8, 0.9]
    tiers = ["T1", "T2", "T2.5", "T3", "T2+T2.5", "combined"]
    modes = data.unique_modes

    results: dict = {
        "global_twonn_multi_seed": {},
        "global_skdim_twonn": {},
        "per_mode_twonn_multi_seed": {},
        "subsample_sensitivity": {},
    }

    # --- Global ID across seeds ---
    for tier in tiers:
        X = data.get_tier(tier)
        X_std = _standardize(X)
        X_clean = _remove_constant(X_std)

        # DADApy TwoNN (deterministic, no seed needed for point estimate)
        try:
            dada = DADAData(X_clean)
            dada.compute_id_2NN()
            dadapy_id = float(dada.intrinsic_dim)
        except Exception as e:
            dadapy_id = f"ERROR: {e}"

        # skdim TwoNN
        try:
            est = SkdimTwoNN()
            est.fit(X_clean)
            skdim_id = float(est.dimension_)
        except Exception as e:
            skdim_id = f"ERROR: {e}"

        results["global_twonn_multi_seed"][tier] = {"dadapy": dadapy_id}
        results["global_skdim_twonn"][tier] = skdim_id

        # Bootstrap with different seeds
        seed_boot_means = {}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            boot_ids = []
            n = X_clean.shape[0]
            subsample_size = max(8, int(n * 0.8))
            for _ in range(200):
                idx = rng.choice(n, size=subsample_size, replace=False)
                try:
                    dada_b = DADAData(X_clean[idx])
                    dada_b.compute_id_2NN()
                    if dada_b.intrinsic_dim is not None:
                        boot_ids.append(float(dada_b.intrinsic_dim))
                except Exception:
                    continue
            if boot_ids:
                seed_boot_means[seed] = {
                    "mean": float(np.mean(boot_ids)),
                    "std": float(np.std(boot_ids)),
                    "ci_lo": float(np.percentile(boot_ids, 2.5)),
                    "ci_hi": float(np.percentile(boot_ids, 97.5)),
                }
        results["global_twonn_multi_seed"][tier]["bootstrap_by_seed"] = seed_boot_means

    print("  Global ID done.")

    # --- Per-mode ID across seeds (for T2+T2.5 only, the key tier) ---
    tier = "T2+T2.5"
    X_full = data.get_tier(tier)
    for mode in modes:
        mask = data.mode_mask(mode)
        X_mode = X_full[mask]
        X_std = _standardize(X_mode)
        X_clean = _remove_constant(X_std)

        try:
            dada = DADAData(X_clean)
            dada.compute_id_2NN()
            point_id = float(dada.intrinsic_dim)
        except Exception as e:
            point_id = f"ERROR: {e}"

        try:
            est = SkdimTwoNN()
            est.fit(X_clean)
            skdim_mode_id = float(est.dimension_)
        except Exception as e:
            skdim_mode_id = f"ERROR: {e}"

        seed_boots = {}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            boot_ids = []
            n = X_clean.shape[0]
            subsample_size = max(8, int(n * 0.8))
            for _ in range(200):
                idx = rng.choice(n, size=subsample_size, replace=False)
                try:
                    dada_b = DADAData(X_clean[idx])
                    dada_b.compute_id_2NN()
                    if dada_b.intrinsic_dim is not None:
                        boot_ids.append(float(dada_b.intrinsic_dim))
                except Exception:
                    continue
            if boot_ids:
                seed_boots[seed] = {
                    "mean": float(np.mean(boot_ids)),
                    "std": float(np.std(boot_ids)),
                }
        results["per_mode_twonn_multi_seed"][mode] = {
            "dadapy_id": point_id,
            "skdim_id": skdim_mode_id,
            "bootstrap_by_seed": seed_boots,
        }

    print("  Per-mode ID done.")

    # --- Subsample fraction sensitivity (global T2+T2.5) ---
    X = data.get_tier("T2+T2.5")
    X_std = _standardize(X)
    X_clean = _remove_constant(X_std)
    n = X_clean.shape[0]

    for frac in subsample_fracs:
        subsample_size = max(8, int(n * frac))
        rng = np.random.default_rng(42)
        boot_ids = []
        for _ in range(200):
            idx = rng.choice(n, size=subsample_size, replace=False)
            try:
                dada_b = DADAData(X_clean[idx])
                dada_b.compute_id_2NN()
                if dada_b.intrinsic_dim is not None:
                    boot_ids.append(float(dada_b.intrinsic_dim))
            except Exception:
                continue
        results["subsample_sensitivity"][str(frac)] = {
            "subsample_size": subsample_size,
            "mean": float(np.mean(boot_ids)) if boot_ids else None,
            "std": float(np.std(boot_ids)) if boot_ids else None,
            "n_successful": len(boot_ids),
        }

    print("  Subsample sensitivity done.")

    return results


# ──────────────────────────────────────────────────────────────
# Section 2: CCGP verification
# ──────────────────────────────────────────────────────────────

def _generate_topic_folds(
    topics: list[str],
    n_folds: int,
    rng: np.random.Generator,
) -> list[tuple[list[str], list[str]]]:
    topics_arr = np.array(topics)
    n = len(topics_arr)
    fold_size = n // n_folds
    perm = rng.permutation(n)
    folds = []
    for i in range(n_folds):
        test_idx = perm[i * fold_size : (i + 1) * fold_size]
        train_idx = np.setdiff1d(perm, test_idx)
        folds.append((topics_arr[train_idx].tolist(), topics_arr[test_idx].tolist()))
    return folds


def _run_ccgp_variant(
    X: NDArray[np.float32],
    modes: NDArray,
    topics: NDArray,
    n_folds: int,
    seed: int,
    clf_name: str,
) -> dict:
    """Run CCGP with a specific classifier, fold count, and seed."""
    rng = np.random.default_rng(seed)
    unique_modes = sorted(set(modes))
    unique_topics = sorted(set(topics))
    n_modes = len(unique_modes)

    topic_folds = _generate_topic_folds(unique_topics, n_folds=n_folds, rng=rng)

    # 5-way multiclass
    accs = []
    per_mode_correct: dict[str, int] = {m: 0 for m in unique_modes}
    per_mode_total: dict[str, int] = {m: 0 for m in unique_modes}

    for train_topics, test_topics in topic_folds:
        train_mask = np.isin(topics, train_topics)
        test_mask = np.isin(topics, test_topics)
        X_train, y_train = X[train_mask], modes[train_mask]
        X_test, y_test = X[test_mask], modes[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if clf_name == "knn3":
            clf = KNeighborsClassifier(n_neighbors=min(3, len(X_train) // n_modes))
        elif clf_name == "knn5":
            clf = KNeighborsClassifier(n_neighbors=min(5, len(X_train) // n_modes))
        elif clf_name == "linearsvc":
            clf = LinearSVC(max_iter=5000, dual="auto")
        elif clf_name == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        accs.append(float(np.mean(y_pred == y_test)))

        for mode in unique_modes:
            mode_mask = y_test == mode
            per_mode_correct[mode] += int(np.sum(y_pred[mode_mask] == mode))
            per_mode_total[mode] += int(np.sum(mode_mask))

    per_mode_recall = {
        m: per_mode_correct[m] / max(per_mode_total[m], 1)
        for m in unique_modes
    }

    # Binary dichotomies
    from itertools import combinations

    dichotomies = []
    n = len(unique_modes)
    for k in range(1, n // 2 + 1):
        for group_a in combinations(unique_modes, k):
            group_b = tuple(m for m in unique_modes if m not in group_a)
            if k == n // 2 and group_a > group_b:
                continue
            dichotomies.append((group_a, group_b))

    n_decodable = 0
    dich_details = []
    for group_a, group_b in dichotomies:
        fold_accs = []
        for train_topics, test_topics in topic_folds:
            train_mask = np.isin(topics, train_topics)
            test_mask = np.isin(topics, test_topics)
            mode_mask = np.isin(modes, group_a + group_b)
            train_full = train_mask & mode_mask
            test_full = test_mask & mode_mask

            if np.sum(train_full) < 4 or np.sum(test_full) < 2:
                continue

            X_tr, X_te = X[train_full], X[test_full]
            y_tr = np.isin(modes[train_full], group_a).astype(int)
            y_te = np.isin(modes[test_full], group_a).astype(int)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            if clf_name == "knn3":
                clf = KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_tr) // 2)))
            elif clf_name == "knn5":
                clf = KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_tr) // 2)))
            elif clf_name == "linearsvc":
                clf = LinearSVC(max_iter=5000, dual="auto")
            elif clf_name == "rf":
                clf = RandomForestClassifier(n_estimators=100, random_state=seed)
            else:
                raise ValueError(f"Unknown classifier: {clf_name}")

            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_te_s)
            fold_accs.append(float(np.mean(y_pred == y_te)))

        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        decodable = mean_acc > 0.65
        if decodable:
            n_decodable += 1
        dich_details.append({
            "group_a": list(group_a),
            "group_b": list(group_b),
            "mean_accuracy": mean_acc,
            "decodable": decodable,
        })

    return {
        "multiclass_mean": float(np.mean(accs)) if accs else 0.0,
        "multiclass_std": float(np.std(accs)) if accs else 0.0,
        "multiclass_fold_accs": accs,
        "per_mode_recall": per_mode_recall,
        "n_decodable": n_decodable,
        "n_dichotomies": len(dichotomies),
        "ccgp_score": n_decodable / max(len(dichotomies), 1),
        "dichotomies": dich_details,
    }


def run_ccgp_verification(data: Run4Data) -> dict:
    """
    Verify CCGP with different seeds, classifiers, and fold counts.
    Focus on T2+T2.5 (the key tier).
    """
    tier = "T2+T2.5"
    X = data.get_tier(tier)
    results: dict = {"tier": tier, "variants": {}}

    # Different seeds with original classifier (kNN k=3)
    for seed in [42, 123, 777]:
        key = f"knn3_seed{seed}_5fold"
        print(f"    CCGP: {key}")
        results["variants"][key] = _run_ccgp_variant(
            X, data.modes, data.topics, n_folds=5, seed=seed, clf_name="knn3"
        )

    # Different classifiers with original seed
    for clf_name in ["knn5", "linearsvc", "rf"]:
        key = f"{clf_name}_seed42_5fold"
        print(f"    CCGP: {key}")
        results["variants"][key] = _run_ccgp_variant(
            X, data.modes, data.topics, n_folds=5, seed=42, clf_name=clf_name
        )

    # Different fold counts
    for n_folds in [4, 10, 20]:
        key = f"knn3_seed42_{n_folds}fold"
        print(f"    CCGP: {key}")
        results["variants"][key] = _run_ccgp_variant(
            X, data.modes, data.topics, n_folds=n_folds, seed=42, clf_name="knn3"
        )

    # Also run on T2 and T2.5 individually for reference
    for tier_name in ["T2", "T2.5", "combined"]:
        X_t = data.get_tier(tier_name)
        key = f"knn3_seed42_5fold_{tier_name}"
        print(f"    CCGP: {key}")
        results["variants"][key] = _run_ccgp_variant(
            X_t, data.modes, data.topics, n_folds=5, seed=42, clf_name="knn3"
        )

    return results


# ──────────────────────────────────────────────────────────────
# Section 3: Delta-hyperbolicity verification
# ──────────────────────────────────────────────────────────────

def _compute_centroids(
    X: NDArray, modes: NDArray, metric: str = "euclidean"
) -> tuple[dict[str, NDArray], dict[tuple[str, str], float]]:
    """Compute centroids and pairwise distances."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    unique_modes = sorted(set(modes))
    centroids = {}
    for mode in unique_modes:
        mask = modes == mode
        centroids[mode] = X_std[mask].mean(axis=0)

    distances: dict[tuple[str, str], float] = {}
    for i, m1 in enumerate(unique_modes):
        for m2 in unique_modes[i + 1 :]:
            if metric == "euclidean":
                d = float(np.linalg.norm(centroids[m1] - centroids[m2]))
            elif metric == "cosine":
                d = float(cosine(centroids[m1], centroids[m2]))
            elif metric == "manhattan":
                d = float(np.sum(np.abs(centroids[m1] - centroids[m2])))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            distances[(m1, m2)] = d

    return centroids, distances


def _hierarchical_clustering(
    centroid_dists: dict[tuple[str, str], float],
    modes: list[str],
    linkage_method: str = "average",
) -> str:
    """Build hierarchical clustering from centroid distance matrix and return Newick-like string."""
    n = len(modes)
    mode_idx = {m: i for i, m in enumerate(modes)}

    # Build condensed distance matrix
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            key = (modes[i], modes[j])
            if key not in centroid_dists:
                key = (modes[j], modes[i])
            condensed.append(centroid_dists[key])

    Z = linkage(condensed, method=linkage_method)
    root = to_tree(Z)

    def _to_newick(node) -> str:
        if node.is_leaf():
            return modes[node.id]
        left = _to_newick(node.get_left())
        right = _to_newick(node.get_right())
        return f"({left},{right})"

    return _to_newick(root)


def run_hyperbolicity_verification(data: Run4Data) -> dict:
    """Verify centroid topology with different metrics and clustering."""
    tier = "T2+T2.5"
    X = data.get_tier(tier)
    modes_list = sorted(set(data.modes))
    results: dict = {"tier": tier}

    # Euclidean centroids (original)
    _, euc_dists = _compute_centroids(X, data.modes, "euclidean")
    results["euclidean_centroid_distances"] = {
        f"{k[0]}__{k[1]}": v for k, v in euc_dists.items()
    }

    # Cosine centroids
    _, cos_dists = _compute_centroids(X, data.modes, "cosine")
    results["cosine_centroid_distances"] = {
        f"{k[0]}__{k[1]}": v for k, v in cos_dists.items()
    }

    # Manhattan centroids
    _, man_dists = _compute_centroids(X, data.modes, "manhattan")
    results["manhattan_centroid_distances"] = {
        f"{k[0]}__{k[1]}": v for k, v in man_dists.items()
    }

    # Hierarchical clustering with different methods and metrics
    results["hierarchical_clustering"] = {}
    for metric_name, dists in [
        ("euclidean", euc_dists),
        ("cosine", cos_dists),
        ("manhattan", man_dists),
    ]:
        for method in ["single", "average", "complete", "ward" if metric_name == "euclidean" else "average"]:
            if method == "average" and metric_name != "euclidean":
                # Already done for euclidean with average
                pass
            key = f"{metric_name}_{method}"
            if key in results["hierarchical_clustering"]:
                continue
            try:
                tree = _hierarchical_clustering(dists, modes_list, method)
                results["hierarchical_clustering"][key] = tree
            except Exception as e:
                results["hierarchical_clustering"][key] = f"ERROR: {e}"

    # Check NearestCentroid gives same centroids
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    nc = NearestCentroid()
    nc.fit(X_std, data.modes)
    sklearn_centroids = {}
    for i, mode in enumerate(nc.classes_):
        sklearn_centroids[mode] = nc.centroids_[i]

    # Compare sklearn NearestCentroid with manual
    manual_centroids, _ = _compute_centroids(X, data.modes, "euclidean")
    centroid_match = {}
    for mode in modes_list:
        diff = float(np.linalg.norm(sklearn_centroids[mode] - manual_centroids[mode]))
        centroid_match[mode] = {"l2_diff": diff, "match": diff < 1e-10}
    results["centroid_verification"] = centroid_match

    # Topology check: identify pairs and outgroup across all metrics
    topology_summary: dict = {}
    for metric_name, dists in [
        ("euclidean", euc_dists),
        ("cosine", cos_dists),
        ("manhattan", man_dists),
    ]:
        sorted_pairs = sorted(dists.items(), key=lambda x: x[1])
        nearest_pair = sorted_pairs[0]
        farthest_pair = sorted_pairs[-1]

        # Analogical outgroup ratio
        anal_dists = {k: v for k, v in dists.items() if "analogical" in k}
        non_anal = {k: v for k, v in dists.items() if "analogical" not in k}
        mean_anal = float(np.mean(list(anal_dists.values())))
        mean_non_anal = float(np.mean(list(non_anal.values())))

        topology_summary[metric_name] = {
            "nearest_pair": f"{nearest_pair[0][0]}-{nearest_pair[0][1]}",
            "nearest_dist": nearest_pair[1],
            "farthest_pair": f"{farthest_pair[0][0]}-{farthest_pair[0][1]}",
            "farthest_dist": farthest_pair[1],
            "analogical_outgroup_ratio": mean_anal / max(mean_non_anal, 1e-10),
            "lin_soc_dist": dists.get(("linear", "socratic"), dists.get(("socratic", "linear"))),
            "con_dia_dist": dists.get(("contrastive", "dialectical"), dists.get(("dialectical", "contrastive"))),
        }

    results["topology_summary"] = topology_summary

    # Delta-hyperbolicity with Manhattan distance matrix
    scaler2 = StandardScaler()
    X_std2 = scaler2.fit_transform(X).astype(np.float64)
    D_man = squareform(pdist(X_std2, metric="cityblock"))
    diameter_man = float(np.max(D_man))

    # Sample quadruples for Manhattan delta
    rng = np.random.default_rng(42)
    n_pts = D_man.shape[0]
    n_quads = 200_000
    quads = np.zeros((n_quads, 4), dtype=np.int32)
    for q in range(n_quads):
        quads[q] = rng.choice(n_pts, size=4, replace=False)
    i, j, k, l = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    s1 = D_man[i, j] + D_man[k, l]
    s2 = D_man[i, k] + D_man[j, l]
    s3 = D_man[i, l] + D_man[j, k]
    sums = np.stack([s1, s2, s3], axis=1)
    sums.sort(axis=1)
    deltas = (sums[:, 2] - sums[:, 1]) / 2

    results["manhattan_delta"] = {
        "delta_max": float(np.max(deltas)),
        "delta_relative": float(np.max(deltas)) / diameter_man if diameter_man > 0 else 0,
        "delta_mean": float(np.mean(deltas)),
        "diameter": diameter_man,
    }

    return results


# ──────────────────────────────────────────────────────────────
# Section 4: Silhouette scores
# ──────────────────────────────────────────────────────────────

def run_silhouette_analysis(data: Run4Data) -> dict:
    """Compute silhouette scores per tier."""
    results: dict = {}
    tiers = ["T1", "T2", "T2.5", "T3", "T2+T2.5", "combined"]

    for tier in tiers:
        X = data.get_tier(tier)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        try:
            score = float(silhouette_score(X_std, data.modes))
        except Exception as e:
            score = f"ERROR: {e}"

        results[tier] = {"silhouette_score": score}

    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def _clean_for_json(obj: object) -> object:
    """Make object JSON-serializable."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_clean_for_json(v) for v in obj]
    return obj


def _build_stability_summary(verification: dict) -> dict:
    """
    Synthesize all verification results into a stability assessment.
    """
    summary: dict = {
        "data_integrity": "PASS" if verification["sanity_checks"]["samples_match"]
        and verification["sanity_checks"]["dims_match"]
        and verification["sanity_checks"]["no_nan_inf"]
        else "FAIL",
    }

    # ID stability: compare bootstrap means across seeds
    id_data = verification["intrinsic_dimension"]
    id_stability = {}
    for tier, tier_data in id_data["global_twonn_multi_seed"].items():
        boots = tier_data.get("bootstrap_by_seed", {})
        if boots:
            means = [v["mean"] for v in boots.values()]
            id_stability[tier] = {
                "point_estimate": tier_data["dadapy"],
                "boot_means_across_seeds": means,
                "cross_seed_range": max(means) - min(means),
                "cross_seed_std": float(np.std(means)),
                "stable": (max(means) - min(means)) < 3.0,
            }
    summary["id_global_stability"] = id_stability

    # Per-mode ID stability
    per_mode_data = id_data["per_mode_twonn_multi_seed"]
    mode_stability = {}
    for mode, mode_data in per_mode_data.items():
        boots = mode_data.get("bootstrap_by_seed", {})
        if boots:
            means = [v["mean"] for v in boots.values()]
            mode_stability[mode] = {
                "point_estimate": mode_data["dadapy_id"],
                "skdim_estimate": mode_data["skdim_id"],
                "boot_means_across_seeds": means,
                "cross_seed_range": max(means) - min(means),
                "stable": (max(means) - min(means)) < 5.0,
            }
    summary["id_per_mode_stability"] = mode_stability

    # Key ID claims
    global_ids = {
        tier: data["dadapy"]
        for tier, data in id_data["global_twonn_multi_seed"].items()
        if isinstance(data["dadapy"], (int, float))
    }
    summary["id_key_claims"] = {
        "T1_approx_18": abs(global_ids.get("T1", 0) - 18) < 3,
        "T2_approx_16": abs(global_ids.get("T2", 0) - 16) < 3,
        "T2.5_approx_17": abs(global_ids.get("T2.5", 0) - 17) < 3,
        "T3_approx_26": abs(global_ids.get("T3", 0) - 26) < 5,
        "T2T25_approx_22": abs(global_ids.get("T2+T2.5", 0) - 22) < 4,
        "combined_approx_24": abs(global_ids.get("combined", 0) - 24) < 5,
        "T3_dramatically_higher": global_ids.get("T3", 0) > global_ids.get("T1", 999) + 5,
        "actual_values": global_ids,
    }

    # CCGP stability
    ccgp_data = verification["ccgp"]
    ccgp_stability: dict = {"ccgp_scores": {}, "multiclass_means": {}}
    for key, variant in ccgp_data["variants"].items():
        ccgp_stability["ccgp_scores"][key] = variant["ccgp_score"]
        ccgp_stability["multiclass_means"][key] = variant["multiclass_mean"]

    # Analogical recall across variants
    analog_recalls = {}
    for key, variant in ccgp_data["variants"].items():
        if "T2" not in key and "T2.5" not in key and "combined" not in key:
            analog_recalls[key] = variant["per_mode_recall"].get("analogical", 0)
    ccgp_stability["analogical_recall_across_variants"] = analog_recalls
    ccgp_stability["analogical_consistently_high"] = all(
        v >= 0.7 for v in analog_recalls.values()
    ) if analog_recalls else False

    # CCGP score stability
    t2t25_ccgp_scores = [
        v["ccgp_score"]
        for k, v in ccgp_data["variants"].items()
        if "T2" not in k.split("_")[-1] and "T2.5" not in k.split("_")[-1]
        and "combined" not in k
    ]
    ccgp_stability["ccgp_score_range"] = (
        max(t2t25_ccgp_scores) - min(t2t25_ccgp_scores) if t2t25_ccgp_scores else None
    )
    summary["ccgp_stability"] = ccgp_stability

    # Topology stability
    topo = verification["delta_hyperbolicity"]["topology_summary"]
    topology_stable = {}
    for metric, data in topo.items():
        topology_stable[metric] = {
            "nearest_pair": data["nearest_pair"],
            "analogical_outgroup_ratio": data["analogical_outgroup_ratio"],
            "analogical_is_outgroup": data["analogical_outgroup_ratio"] > 1.2,
        }

    # Check if nearest pairs are consistent
    nearest_pairs = set(d["nearest_pair"] for d in topo.values())
    topology_stable["consistent_nearest_pair"] = len(nearest_pairs) <= 2
    topology_stable["nearest_pairs_seen"] = list(nearest_pairs)

    summary["topology_stability"] = topology_stable

    # Hierarchical clustering consistency
    trees = verification["delta_hyperbolicity"]["hierarchical_clustering"]
    summary["clustering_trees"] = trees

    return summary


def main() -> None:
    print("=" * 60)
    print("GEOMETRIC TRIO VERIFICATION RUN")
    print("=" * 60)

    print("\nLoading Run 4 data...")
    data = load_run4()
    print(f"Loaded {data.n_samples} samples, {len(data.unique_modes)} modes, {len(data.unique_topics)} topics")

    verification: dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Section 0: Sanity checks
    print("\n--- Sanity Checks ---")
    verification["sanity_checks"] = run_sanity_checks(data)
    print(f"  Samples: {verification['sanity_checks']['n_samples']}")
    print(f"  Balanced: {verification['sanity_checks']['balanced']}")
    print(f"  Dims match: {verification['sanity_checks']['dims_match']}")
    print(f"  No NaN/Inf: {verification['sanity_checks']['no_nan_inf']}")

    # Section 1: Intrinsic dimension
    print("\n--- Intrinsic Dimension Verification ---")
    verification["intrinsic_dimension"] = run_id_verification(data)

    # Section 2: CCGP
    print("\n--- CCGP Verification ---")
    verification["ccgp"] = run_ccgp_verification(data)

    # Section 3: Delta-hyperbolicity
    print("\n--- Delta-Hyperbolicity Verification ---")
    verification["delta_hyperbolicity"] = run_hyperbolicity_verification(data)

    # Section 4: Silhouette scores
    print("\n--- Silhouette Scores ---")
    verification["silhouette"] = run_silhouette_analysis(data)
    for tier, scores in verification["silhouette"].items():
        print(f"  {tier}: {scores['silhouette_score']:.4f}" if isinstance(scores['silhouette_score'], float) else f"  {tier}: {scores['silhouette_score']}")

    # Section 5: Stability summary
    print("\n--- Building Stability Summary ---")
    verification["stability_summary"] = _build_stability_summary(verification)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "verification_run.json"
    with open(output_path, "w") as f:
        json.dump(_clean_for_json(verification), f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY VERIFICATION FINDINGS")
    print("=" * 60)
    s = verification["stability_summary"]

    print(f"\nData integrity: {s['data_integrity']}")

    print("\nGlobal ID stability (cross-seed range):")
    for tier, info in s["id_global_stability"].items():
        print(f"  {tier}: range={info['cross_seed_range']:.2f}, "
              f"stable={info['stable']}")

    print("\nID key claims:")
    for claim, ok in s["id_key_claims"].items():
        if claim != "actual_values":
            print(f"  {claim}: {ok}")

    print("\nCCGP scores across variants:")
    for key, score in s["ccgp_stability"]["ccgp_scores"].items():
        print(f"  {key}: {score:.2f}")

    print(f"\nAnalogical consistently high recall: "
          f"{s['ccgp_stability']['analogical_consistently_high']}")

    print("\nTopology stability:")
    for metric, info in s["topology_stability"].items():
        if isinstance(info, dict) and "nearest_pair" in info:
            print(f"  {metric}: nearest={info['nearest_pair']}, "
                  f"outgroup_ratio={info['analogical_outgroup_ratio']:.2f}")


if __name__ == "__main__":
    main()
